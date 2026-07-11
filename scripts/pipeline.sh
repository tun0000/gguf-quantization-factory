#!/usr/bin/env bash
# pipeline.sh — HF 模型 → GGUF → 量化 → PPL/速度評測 → summary 一條龍
#
# 用法:
#   ./pipeline.sh                              # 預設 Qwen/Qwen2.5-3B-Instruct,跑全部步驟
#   ./pipeline.sh <HF_MODEL_ID>                # 量化任意 HF 模型
#   ./pipeline.sh <HF_MODEL_ID> <step>         # 只跑單一步驟
#
# step: all | download | convert | quantize | wikitext | imatrix | ppl | bench | summary
#
# 環境變數:
#   GGUF_WORK_DIR  工作目錄(預設 ~/gguf-factory,模型與 GGUF 都放這,不進 git)
#   QUANTS_LIST    量化等級清單(預設 "Q4_K_M Q5_K_M Q8_0";IQ 系列或 *_IMAT 後綴
#                  會自動帶 --imatrix,需先跑過 imatrix step)
#   PPL_CTX        perplexity context 長度(預設 512,與社群慣例一致)
#   NGL            -ngl 層數(預設 99 全放 GPU)
#   VRAM_LIMIT_MB  跑 GPU 工作前允許的既有 VRAM 佔用上限(預設 4096)
#   FORCE          =1 時重跑已有結果的 ppl/bench(預設跳過已完成的)
set -euo pipefail

MODEL_ID="${1:-Qwen/Qwen2.5-3B-Instruct}"
STEP="${2:-all}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
WORK_DIR="${GGUF_WORK_DIR:-$HOME/gguf-factory}"
LLAMA_BIN="$WORK_DIR/llama.cpp/build/bin"
PY="$WORK_DIR/.venv/bin/python"
HF_CLI="$WORK_DIR/.venv/bin/hf"

read -ra QUANTS <<< "${QUANTS_LIST:-Q4_K_M Q5_K_M Q8_0}"
PPL_CTX="${PPL_CTX:-512}"
NGL="${NGL:-99}"
VRAM_LIMIT_MB="${VRAM_LIMIT_MB:-4096}"

MODEL_NAME="$(basename "$MODEL_ID")"
SLUG="$(echo "$MODEL_NAME" | tr '[:upper:]' '[:lower:]')"
MODEL_DIR="$WORK_DIR/models/$MODEL_NAME"
HF_DIR="$MODEL_DIR/hf"
F16_GGUF="$MODEL_DIR/${SLUG}-f16.gguf"
DATA_DIR="$WORK_DIR/data"
WIKITEXT="$DATA_DIR/wikitext-2-test.txt"
RESULTS_DIR="$REPO_DIR/results"
LOG_DIR="$RESULTS_DIR/logs"
mkdir -p "$MODEL_DIR" "$DATA_DIR" "$LOG_DIR"

log() { echo "[pipeline $(date +%H:%M:%S)] $*"; }

gguf_for() {  # gguf_for <F16|Q4_K_M|...> → 檔案路徑
    local lvl="$1"
    if [ "$lvl" = "F16" ]; then echo "$F16_GGUF"
    else echo "$MODEL_DIR/${SLUG}-$(echo "$lvl" | tr '[:upper:]' '[:lower:]').gguf"; fi
}

check_vram() {  # GPU 工作前確認沒有其他大程式佔 VRAM
    local used
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    log "目前 VRAM 佔用: ${used} MiB(上限 ${VRAM_LIMIT_MB} MiB)"
    if [ "$used" -gt "$VRAM_LIMIT_MB" ]; then
        nvidia-smi
        log "VRAM 已被其他程式大量佔用(${used} MiB),請先關閉再重跑;或設 VRAM_LIMIT_MB 放寬"
        exit 1
    fi
}

step_download() {
    log "下載 $MODEL_ID → $HF_DIR"
    "$HF_CLI" download "$MODEL_ID" --local-dir "$HF_DIR"
    ls -lh "$HF_DIR"
}

step_convert() {
    log "convert_hf_to_gguf.py → F16 GGUF"
    "$PY" "$WORK_DIR/llama.cpp/convert_hf_to_gguf.py" "$HF_DIR" \
        --outfile "$F16_GGUF" --outtype f16 2>&1 | tail -20
    ls -lh "$F16_GGUF"
}

record_size() {  # record_size <level> <path> — upsert 到 sizes.tsv
    local sizes="$LOG_DIR/${SLUG}-sizes.tsv"
    touch "$sizes"
    grep -vP "^$1\t" "$sizes" > "$sizes.tmp" || true
    printf '%s\t%s\n' "$1" "$(stat -c%s "$2")" >> "$sizes.tmp"
    mv "$sizes.tmp" "$sizes"
}

step_quantize() {
    record_size F16 "$F16_GGUF"
    for q in "${QUANTS[@]}"; do
        local out qtype args; out="$(gguf_for "$q")"
        qtype="${q%_IMAT}"   # Q4_K_M_IMAT → 量化型別仍是 Q4_K_M
        args=()
        # IQ 系列與 *_IMAT 需要 importance matrix(先跑 imatrix step)
        if [[ "$q" == IQ* || "$q" == *_IMAT ]]; then
            [ -f "$MODEL_DIR/imatrix.dat" ] || { log "$q 需要 $MODEL_DIR/imatrix.dat,先跑 imatrix step"; exit 1; }
            args+=(--imatrix "$MODEL_DIR/imatrix.dat")
        fi
        log "llama-quantize → $q(型別 $qtype ${args[*]:-})"
        "$LLAMA_BIN/llama-quantize" "${args[@]}" "$F16_GGUF" "$out" "$qtype" 2>&1 | tail -5
        record_size "$q" "$out"
    done
    log "檔案大小:"
    ls -lh "$MODEL_DIR"/*.gguf
}

step_imatrix() {
    local train="$DATA_DIR/wikitext-2-train.txt"
    local imat="$MODEL_DIR/imatrix.dat"
    if [ -f "$imat" ] && [ "${FORCE:-0}" != 1 ]; then log "imatrix 已存在,跳過"; return; fi
    # 校正語料一律用 train split(test split 留給 PPL 評測,避免資料污染)
    [ -f "$train" ] || "$PY" "$SCRIPT_DIR/prepare_wikitext.py" --out "$train" --split train --max-chars 8000000
    check_vram
    log "llama-imatrix(wikitext-2 train,-ngl $NGL)→ $imat"
    "$LLAMA_BIN/llama-imatrix" -m "$F16_GGUF" -f "$train" -ngl "$NGL" -o "$imat" </dev/null 2>&1 \
        | tail -5
    ls -lh "$imat"
}

step_wikitext() {
    log "準備 wikitext-2-raw-v1 test split"
    "$PY" "$SCRIPT_DIR/prepare_wikitext.py" --out "$WIKITEXT"
}

step_ppl() {
    check_vram
    for lvl in F16 "${QUANTS[@]}"; do
        local gguf plog; gguf="$(gguf_for "$lvl")"
        plog="$LOG_DIR/${SLUG}-ppl-${lvl}.log"
        if [ "${FORCE:-0}" != 1 ] && grep -qs 'Final estimate' "$plog"; then
            log "[$lvl] 已有 PPL 結果,跳過(FORCE=1 可重跑)"; continue
        fi
        log "llama-perplexity [$lvl] -c $PPL_CTX -ngl $NGL(log → $(basename "$plog"))"
        "$LLAMA_BIN/llama-perplexity" -m "$gguf" -f "$WIKITEXT" \
            -c "$PPL_CTX" -ngl "$NGL" </dev/null > "$plog" 2>&1
        grep -E 'Final estimate' "$plog" || { log "[$lvl] 沒抓到 Final estimate,請看 $plog"; exit 1; }
    done
}

step_bench() {
    check_vram
    for lvl in F16 "${QUANTS[@]}"; do
        local gguf bjson blog vram; gguf="$(gguf_for "$lvl")"
        bjson="$LOG_DIR/${SLUG}-bench-${lvl}.json"
        blog="$LOG_DIR/${SLUG}-bench-${lvl}.log"
        vram="$LOG_DIR/${SLUG}-vram-${lvl}.txt"
        if [ "${FORCE:-0}" != 1 ] && [ -s "$bjson" ]; then
            log "[$lvl] 已有 bench 結果,跳過(FORCE=1 可重跑)"; continue
        fi
        log "llama-bench [$lvl](VRAM 峰值同步量測)"
        "$SCRIPT_DIR/measure_vram.sh" "$vram" -- \
            "$LLAMA_BIN/llama-bench" -m "$gguf" -ngl "$NGL" -o json > "$bjson" 2> "$blog"
        "$LLAMA_BIN/llama-bench" -m "$gguf" -ngl "$NGL" 2>/dev/null | tee -a "$blog"
        cat "$vram"
    done
}

step_summary() {
    local commit=""
    commit=$(git -C "$WORK_DIR/llama.cpp" log -1 --format=%h 2>/dev/null || echo unknown)
    "$PY" "$SCRIPT_DIR/make_summary.py" \
        --slug "$SLUG" --model-id "$MODEL_ID" --logs "$LOG_DIR" \
        --llama-commit "$commit" --ppl-ctx "$PPL_CTX" \
        --out "$RESULTS_DIR/summary-$SLUG.md" --json-out "$RESULTS_DIR/summary-$SLUG.json"
    log "summary → $RESULTS_DIR/summary-$SLUG.md"
    cat "$RESULTS_DIR/summary-$SLUG.md"
}

case "$STEP" in
    all) step_download; step_convert; step_quantize; step_wikitext; step_ppl; step_bench; step_summary ;;
    download|convert|quantize|wikitext|imatrix|ppl|bench|summary) "step_$STEP" ;;
    *) echo "未知 step: $STEP" >&2; exit 1 ;;
esac
log "完成: $STEP"

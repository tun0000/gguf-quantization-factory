#!/usr/bin/env bash
# measure_vram.sh — 執行指令期間輪詢 nvidia-smi,記錄 VRAM 峰值
# 用法: measure_vram.sh <輸出檔> -- <指令...>
set -uo pipefail

OUT="$1"; shift
[ "${1:-}" = "--" ] && shift

query() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }

baseline=$(query)
samples="$OUT.samples"
: > "$samples"

( while :; do query >> "$samples" 2>/dev/null || break; sleep 0.5; done ) &
poller=$!

"$@"
rc=$?

kill "$poller" 2>/dev/null
wait "$poller" 2>/dev/null

peak=$(sort -n "$samples" | tail -1)
peak=${peak:-$baseline}
{
    echo "baseline_mib=$baseline"
    echo "peak_mib=$peak"
    echo "delta_mib=$((peak - baseline))"
} > "$OUT"
rm -f "$samples"
exit "$rc"

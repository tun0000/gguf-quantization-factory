#!/usr/bin/env python3
"""解析 pipeline 產出的 logs,彙整成 results/summary-<slug>.md 與 summary-<slug>.json。

輸入(位於 --logs 目錄):
  <slug>-sizes.tsv        量化等級 \t bytes
  <slug>-ppl-<LVL>.log    llama-perplexity 輸出(抓 "Final estimate: PPL = ...")
  <slug>-bench-<LVL>.json llama-bench -o json 輸出(抓 avg_ts)
  <slug>-vram-<LVL>.txt   measure_vram.sh 輸出(key=val)
"""
import argparse
import json
import re
import subprocess
from pathlib import Path

PREFERRED_ORDER = ["F16", "Q8_0", "Q5_K_M", "Q4_K_M", "Q4_K_M_IMAT", "Q4_K_M_ZHTW", "IQ4_XS", "IQ4_XS_ZHTW"]  # 依精度由高到低排

ADVICE = {
    "F16": "無損基準;僅供對照或 VRAM 非常充裕時使用",
    "Q8_0": "幾乎無損(ΔPPL 通常 <0.1%);對品質敏感的正式服務",
    "Q5_K_M": "品質/大小平衡佳;VRAM 稍緊但仍重視品質的部署",
    "Q4_K_M": "檔案最小、速度最快、品質損失可接受;日常使用與邊緣裝置的推薦預設",
    "Q4_K_M_IMAT": "同 Q4_K_M 大小,英文 wikitext imatrix 校正換取更低 PPL;要 4-bit 品質時優先選",
    "Q4_K_M_ZHTW": "同 Q4_K_M 大小,繁中維基 imatrix 校正;繁中任務比 Q4_K_M_IMAT 再略勝一籌",
    "IQ4_XS": "比 Q4_K_M 再小 ~10%(需 imatrix,英文 wikitext 校正);極度受限的 VRAM/磁碟環境",
    "IQ4_XS_ZHTW": "同 IQ4_XS 大小,繁中維基 imatrix 校正;偏好中文的極限壓縮場景",
}


def discover_levels(logs: Path, slug: str) -> list:
    """從 ppl log 檔自動偵測有哪些量化等級,依 PREFERRED_ORDER 排序,未知的排最後。"""
    found = {
        p.name[len(f"{slug}-ppl-"):-len(".log")]
        for p in logs.glob(f"{slug}-ppl-*.log")
    }
    ordered = [l for l in PREFERRED_ORDER if l in found]
    return ordered + sorted(found - set(ordered))


def human_size(n: int) -> str:
    return f"{n / 1024**3:.2f} GiB"


def gpu_name() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip().splitlines()[0]
    except Exception:
        return "unknown GPU"


def parse_ppl(path: Path):
    if not path.exists():
        return None, None
    m = re.search(r"Final estimate: PPL = ([\d.]+) \+/- ([\d.]+)", path.read_text(errors="replace"))
    return (float(m.group(1)), float(m.group(2))) if m else (None, None)


def parse_bench(path: Path):
    """回傳 (pp_ts, tg_ts)。llama-bench -o json:n_gen==0 的列是 prompt processing。"""
    if not path.exists():
        return None, None
    data = json.loads(path.read_text())
    pp = tg = None
    for row in data:
        if row.get("n_gen", 0) == 0 and row.get("n_prompt", 0) > 0:
            pp = row.get("avg_ts")
        elif row.get("n_prompt", 0) == 0 and row.get("n_gen", 0) > 0:
            tg = row.get("avg_ts")
    return pp, tg


def parse_vram(path: Path):
    if not path.exists():
        return None, None
    kv = dict(
        line.split("=", 1)
        for line in path.read_text().splitlines()
        if "=" in line
    )
    return (
        int(kv["peak_mib"]) if "peak_mib" in kv else None,
        int(kv["delta_mib"]) if "delta_mib" in kv else None,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--logs", required=True)
    ap.add_argument("--llama-commit", default="unknown")
    ap.add_argument("--ppl-ctx", default="512")
    ap.add_argument("--out", required=True)
    ap.add_argument("--json-out", required=True)
    args = ap.parse_args()

    logs = Path(args.logs)
    sizes = {}
    sizes_file = logs / f"{args.slug}-sizes.tsv"
    if sizes_file.exists():
        for line in sizes_file.read_text().splitlines():
            lvl, nbytes = line.split("\t")
            sizes[lvl] = int(nbytes)

    levels = discover_levels(logs, args.slug)
    rows = []
    for lvl in levels:
        ppl, ppl_err = parse_ppl(logs / f"{args.slug}-ppl-{lvl}.log")
        pp_ts, tg_ts = parse_bench(logs / f"{args.slug}-bench-{lvl}.json")
        vram_peak, vram_delta = parse_vram(logs / f"{args.slug}-vram-{lvl}.txt")
        rows.append({
            "level": lvl,
            "bytes": sizes.get(lvl),
            "ppl": ppl,
            "ppl_err": ppl_err,
            "pp_ts": pp_ts,
            "tg_ts": tg_ts,
            "vram_peak_mib": vram_peak,
            "vram_delta_mib": vram_delta,
            "advice": ADVICE.get(lvl, "—"),
        })

    f16_ppl = next((r["ppl"] for r in rows if r["level"] == "F16"), None)
    for r in rows:
        r["dppl_pct"] = (
            round((r["ppl"] - f16_ppl) / f16_ppl * 100, 3)
            if r["ppl"] is not None and f16_ppl else None
        )

    gpu = gpu_name()
    meta = {
        "model_id": args.model_id,
        "gpu": gpu,
        "llama_commit": args.llama_commit,
        "ppl_ctx": int(args.ppl_ctx),
        "dataset": "wikitext-2-raw-v1 (test split)",
        "levels": rows,
    }
    Path(args.json_out).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    def fmt(v, pattern="{:.2f}"):
        return pattern.format(v) if v is not None else "—"

    lines = [
        f"# {args.model_id} GGUF 量化評測結果",
        "",
        f"- **GPU**:{gpu}(`-ngl 99` 全部層放 GPU)",
        f"- **PPL 資料集**:wikitext-2-raw-v1 test split,context = {args.ppl_ctx}",
        f"- **速度**:llama-bench 預設情境(pp512 = 一次吃 512 token 的 prompt;tg128 = 生成 128 token)",
        f"- **llama.cpp commit**:`{args.llama_commit}`",
        "",
        "| 量化等級 | 檔案大小 | PPL ↓ | PPL vs F16 | pp512 tok/s | tg128 tok/s | 峰值 VRAM | 適用情境建議 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        ppl_cell = f"{r['ppl']:.4f} ± {r['ppl_err']:.5f}" if r["ppl"] is not None else "—"
        dppl = "基準" if r["level"] == "F16" else (
            f"{r['dppl_pct']:+.2f}%" if r["dppl_pct"] is not None else "—")
        size = human_size(r["bytes"]) if r["bytes"] else "—"
        vram = f"{r['vram_peak_mib']} MiB" if r["vram_peak_mib"] else "—"
        lines.append(
            f"| {r['level']} | {size} | {ppl_cell} | {dppl} | "
            f"{fmt(r['pp_ts'])} | {fmt(r['tg_ts'])} | {vram} | {r['advice']} |"
        )
    lines.append("")
    lines.append("> PPL(perplexity,困惑度)越低越好;`PPL vs F16` 為相對 F16 基準的變化百分比。")
    lines.append("> 峰值 VRAM 為 llama-bench 執行期間 `nvidia-smi` 輪詢到的整卡最高佔用(含桌面基礎佔用)。")
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out} and {args.json_out}")


if __name__ == "__main__":
    main()

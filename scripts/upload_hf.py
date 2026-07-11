#!/usr/bin/env python3
"""把量化 GGUF 上傳到 Hugging Face model repo(含自動產生 model card)。

- 用 huggingface_hub 的 upload_large_folder:分 chunk 上傳、可斷點續傳。
- token 來源:HF_TOKEN 環境變數或 ~/.cache/huggingface/token(hf auth login 的 cache)。
- 上傳檔名採 HF 社群慣例:<ModelName>-<QUANT>.gguf(hardlink 到 stage 目錄,不佔額外空間)。
"""
import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi

DEFAULT_QUANTS = ["Q4_K_M", "Q5_K_M", "Q8_0"]


def build_model_card(repo_id: str, model_id: str, summary: dict, quants: list) -> str:
    model_name = model_id.split("/")[-1]
    rows_md = []
    for r in summary["levels"]:
        ppl = f"{r['ppl']:.4f}" if r.get("ppl") is not None else "—"
        dppl = "基準 baseline" if r["level"] == "F16" else (
            f"{r['dppl_pct']:+.2f}%" if r.get("dppl_pct") is not None else "—")
        size = f"{r['bytes'] / 1024**3:.2f} GiB" if r.get("bytes") else "—"
        pp = f"{r['pp_ts']:.2f}" if r.get("pp_ts") is not None else "—"
        tg = f"{r['tg_ts']:.2f}" if r.get("tg_ts") is not None else "—"
        vram = f"{r['vram_peak_mib']} MiB" if r.get("vram_peak_mib") else "—"
        uploaded = f"`{model_name}-{r['level']}.gguf`" if r["level"] in quants else "—(未上傳 not uploaded)"
        rows_md.append(
            f"| {r['level']} | {size} | {ppl} | {dppl} | {pp} | {tg} | {vram} | {uploaded} |")
    table = "\n".join(rows_md)

    return f"""---
license: apache-2.0
base_model: {model_id}
language:
- en
- zh
pipeline_tag: text-generation
tags:
- gguf
- quantized
- llama.cpp
- ollama
- qwen2.5
---

# {model_name} — GGUF 量化版

[{model_id}](https://huggingface.co/{model_id}) 的 GGUF 量化版本,
由 [llama.cpp](https://github.com/ggml-org/llama.cpp)(commit `{summary["llama_commit"]}`)
的 `convert_hf_to_gguf.py` + `llama-quantize` 產生。
量化 pipeline 原始碼:[gguf-quantization-factory](https://github.com/tun0000/gguf-quantization-factory)。

GGUF quantized versions of [{model_id}](https://huggingface.co/{model_id}),
produced with llama.cpp. All benchmark numbers below were measured on a real
{summary["gpu"]} (all layers offloaded, `-ngl 99`).

## 量化等級比較 Quantization comparison

- **PPL**:wikitext-2-raw-v1 test split,context = {summary["ppl_ctx"]}(越低越好 / lower is better)
- **速度 Speed**:llama-bench pp512 / tg128,單位 tokens/s

| 量化 Quant | 大小 Size | PPL ↓ | ΔPPL vs F16 | pp512 tok/s | tg128 tok/s | 峰值 VRAM | 檔案 File |
|---|---|---|---|---|---|---|---|
{table}

## 怎麼選 Which one should I pick?

- **Q4_K_M**(推薦預設 / recommended default):檔案最小、速度最快,品質損失通常可接受。
  適合桌機/筆電日常使用、VRAM 有限的裝置。
- **Q5_K_M**:比 Q4_K_M 更貼近原模型,只多一點大小。想要更好品質時選這個。
- **Q8_0**:幾乎無損(ΔPPL 通常 <0.1%),但檔案接近 F16 的一半大小。對品質敏感的正式服務。
- **F16**:未上傳(檔案過大);想要 F16 請用 pipeline 自行轉檔。

## 使用方式 Usage

### Ollama

```bash
# 直接從 Hugging Face 跑(Ollama ≥ 0.3.34)
ollama run hf.co/{repo_id}:Q4_K_M
```

或自己寫 Modelfile(含 Qwen ChatML template 與 stop tokens),見
[gguf-quantization-factory](https://github.com/tun0000/gguf-quantization-factory) 的 `ollama/Modelfile`。

### llama.cpp

```bash
# 下載
hf download {repo_id} {model_name}-Q4_K_M.gguf --local-dir .

# 互動對話(全部層放 GPU)
llama-cli -m {model_name}-Q4_K_M.gguf -ngl 99 -cnv
```

## 來源模型與授權 License

- 來源模型 Base model:[{model_id}](https://huggingface.co/{model_id})
- 授權 License:**Apache-2.0**(依來源模型)
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True, help="來源 HF model id")
    ap.add_argument("--model-dir", required=True, help="放 GGUF 的目錄")
    ap.add_argument("--summary-json", required=True, help="make_summary.py 產出的 summary.json")
    ap.add_argument("--repo-name", default=None, help="HF repo 名(預設 <ModelName>-GGUF)")
    ap.add_argument("--card-only", action="store_true", help="只更新 model card,不傳 GGUF")
    ap.add_argument("--quants", nargs="+", default=DEFAULT_QUANTS, help="要上傳的量化等級")
    args = ap.parse_args()

    model_name = args.model_id.split("/")[-1]
    slug = model_name.lower()
    repo_name = args.repo_name or f"{model_name}-GGUF"

    api = HfApi()
    user = api.whoami()["name"]
    repo_id = f"{user}/{repo_name}"
    print(f"HF user: {user} → repo: {repo_id}")

    api.create_repo(repo_id, repo_type="model", exist_ok=True)

    summary = json.loads(Path(args.summary_json).read_text(encoding="utf-8"))
    card = build_model_card(repo_id, args.model_id, summary, args.quants)
    api.upload_file(
        path_or_fileobj=card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Add model card with real benchmark results",
    )
    print("model card uploaded")

    if not args.card_only:
        model_dir = Path(args.model_dir)
        stage = model_dir / "hf-stage"
        stage.mkdir(exist_ok=True)
        for q in args.quants:
            src = model_dir / f"{slug}-{q.lower()}.gguf"
            dst = stage / f"{model_name}-{q}.gguf"
            if not dst.exists():
                os.link(src, dst)  # hardlink,不多佔磁碟
        print(f"staged: {[p.name for p in stage.glob('*.gguf')]}")
        api.upload_large_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=str(stage),
            allow_patterns=["*.gguf"],
        )

    files = api.list_repo_files(repo_id)
    print("repo files:", files)
    print(f"https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()

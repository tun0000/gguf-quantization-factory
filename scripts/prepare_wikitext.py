#!/usr/bin/env python3
"""下載 wikitext-2-raw-v1 的 test split,存成 llama-perplexity 可用的純文字檔。"""
import argparse
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="輸出純文字檔路徑")
    args = ap.parse_args()

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    rows = ds["text"]
    # raw 版每列通常已含結尾換行;若沒有則以換行相接
    if rows and rows[0].endswith("\n"):
        text = "".join(rows)
    else:
        text = "\n".join(rows)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"rows={len(rows)} chars={len(text)} → {out}")


if __name__ == "__main__":
    main()

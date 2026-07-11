#!/usr/bin/env python3
"""下載 wikitext-2-raw-v1 指定 split,存成 llama-perplexity / llama-imatrix 可用的純文字檔。

test split 供 PPL 評測;train split 供 imatrix 校正(兩者必須分開,避免資料污染)。
"""
import argparse
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="輸出純文字檔路徑")
    ap.add_argument("--split", default="test", choices=["test", "train", "validation"])
    ap.add_argument("--max-chars", type=int, default=0, help="裁切字元數上限(0 = 不裁切)")
    args = ap.parse_args()

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=args.split)
    rows = ds["text"]
    # raw 版每列通常已含結尾換行;若沒有則以換行相接
    if rows and rows[0].endswith("\n"):
        text = "".join(rows)
    else:
        text = "\n".join(rows)
    if args.max_chars and len(text) > args.max_chars:
        text = text[: args.max_chars]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"rows={len(rows)} chars={len(text)} → {out}")


if __name__ == "__main__":
    main()

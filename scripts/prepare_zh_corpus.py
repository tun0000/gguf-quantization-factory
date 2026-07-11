#!/usr/bin/env python3
"""下載中文維基百科(串流),用 OpenCC 強制轉換為台灣繁體,存成純文字檔。

供 llama-imatrix 校正語料或 llama-perplexity 評測語料使用。
用 --skip-articles / 印出的 articles_consumed 可組出互不重疊的 calibration / test 語料,
避免校正語料與評測語料污染(同一批文章不能同時當校正又當考題)。
"""
import argparse
import sys
from pathlib import Path

from datasets import load_dataset
from opencc import OpenCC


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="輸出純文字檔路徑")
    ap.add_argument("--max-chars", type=int, required=True, help="輸出字元數上限(轉繁體後計)")
    ap.add_argument("--skip-articles", type=int, default=0, help="跳過前 N 篇文章(避開已用於校正/其他 split 的文章)")
    ap.add_argument("--config", default="20231101.zh", help="wikimedia/wikipedia 的 config 名")
    args = ap.parse_args()

    cc = OpenCC("s2twp")  # 簡體 → 台灣正體(含慣用詞轉換,如 软件→軟體)
    ds = load_dataset("wikimedia/wikipedia", args.config, split="train", streaming=True)

    chunks = []
    total = 0
    articles_seen = 0
    articles_used = 0
    for row in ds:
        articles_seen += 1
        if articles_seen <= args.skip_articles:
            continue
        text = cc.convert(row["text"])
        chunks.append(text)
        total += len(text)
        articles_used += 1
        if total >= args.max_chars:
            break

    out_text = "\n\n".join(chunks)[: args.max_chars]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(out_text, encoding="utf-8")

    # 印出下一段可以從第幾篇文章開始跳過(給呼叫端組 train/test 不重疊區間)
    next_skip = args.skip_articles + articles_used
    print(f"articles_used={articles_used} chars={len(out_text)} next_skip={next_skip} → {out}")


if __name__ == "__main__":
    main()

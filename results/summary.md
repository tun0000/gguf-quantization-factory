# Qwen/Qwen2.5-3B-Instruct GGUF 量化評測結果

- **GPU**:NVIDIA GeForce RTX 4090(`-ngl 99` 全部層放 GPU)
- **PPL 資料集**:wikitext-2-raw-v1 test split,context = 512
- **速度**:llama-bench 預設情境(pp512 = 一次吃 512 token 的 prompt;tg128 = 生成 128 token)
- **llama.cpp commit**:`4f37f51`

| 量化等級 | 檔案大小 | PPL ↓ | PPL vs F16 | pp512 tok/s | tg128 tok/s | 峰值 VRAM | 適用情境建議 |
|---|---|---|---|---|---|---|---|
| F16 | 5.75 GiB | 9.0631 ± 0.06417 | 基準 | 19999.59 | 127.83 | 8698 MiB | 無損基準;僅供對照或 VRAM 非常充裕時使用 |
| Q8_0 | 3.06 GiB | 9.0806 ± 0.06432 | +0.19% | 22389.26 | 206.78 | 5888 MiB | 幾乎無損(ΔPPL 通常 <0.1%);對品質敏感的正式服務 |
| Q5_K_M | 2.07 GiB | 9.1883 ± 0.06519 | +1.38% | 20976.91 | 270.39 | 4630 MiB | 品質/大小平衡佳;VRAM 稍緊但仍重視品質的部署 |
| Q4_K_M | 1.80 GiB | 9.5741 ± 0.06843 | +5.64% | 19691.23 | 300.21 | 4348 MiB | 檔案最小、速度最快、品質損失可接受;日常使用與邊緣裝置的推薦預設 |

> PPL(perplexity,困惑度)越低越好;`PPL vs F16` 為相對 F16 基準的變化百分比。
> 峰值 VRAM 為 llama-bench 執行期間 `nvidia-smi` 輪詢到的整卡最高佔用(含桌面基礎佔用)。

# meta-llama/Llama-3.2-3B-Instruct GGUF 量化評測結果

- **GPU**:NVIDIA GeForce RTX 4090(`-ngl 99` 全部層放 GPU)
- **PPL 資料集**:wikitext-2-raw-v1 test split,context = 512
- **速度**:llama-bench 預設情境(pp512 = 一次吃 512 token 的 prompt;tg128 = 生成 128 token)
- **llama.cpp commit**:`4f37f51`

| 量化等級 | 檔案大小 | PPL ↓ | PPL vs F16 | pp512 tok/s | tg128 tok/s | 峰值 VRAM | 適用情境建議 |
|---|---|---|---|---|---|---|---|
| F16 | 5.99 GiB | 10.5876 ± 0.07597 | 基準 | 20715.86 | 127.37 | 8799 MiB | 無損基準;僅供對照或 VRAM 非常充裕時使用 |
| Q8_0 | 3.19 GiB | 10.5986 ± 0.07612 | +0.10% | 24897.05 | 206.29 | 5961 MiB | 幾乎無損(ΔPPL 通常 <0.1%);對品質敏感的正式服務 |
| Q5_K_M | 2.16 GiB | 10.6803 ± 0.07681 | +0.88% | 23300.73 | 280.13 | 4894 MiB | 品質/大小平衡佳;VRAM 稍緊但仍重視品質的部署 |
| Q4_K_M | 1.88 GiB | 10.8905 ± 0.07801 | +2.86% | 22565.35 | 308.00 | 4353 MiB | 檔案最小、速度最快、品質損失可接受;日常使用與邊緣裝置的推薦預設 |

> PPL(perplexity,困惑度)越低越好;`PPL vs F16` 為相對 F16 基準的變化百分比。
> 峰值 VRAM 為 llama-bench 執行期間 `nvidia-smi` 輪詢到的整卡最高佔用(含桌面基礎佔用)。

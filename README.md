# GGUF 量化工廠(GGUF Quantization Factory)

> 一條可重複執行的 pipeline:把任意 Hugging Face 模型轉成 GGUF、量化成多個等級、
> 用真實數據量測品質(perplexity)與速度(tokens/s),產出 Ollama 部署檔,
> 並上傳 Hugging Face。全程在本機 WSL2 + NVIDIA RTX 4090 上執行。

**English summary**: A reproducible pipeline that converts any Hugging Face model to GGUF,
quantizes it to multiple levels (Q4_K_M / Q5_K_M / Q8_0), benchmarks quality (wikitext-2
perplexity) and speed (llama-bench) with real measurements, deploys to Ollama, and uploads
to Hugging Face. Everything runs locally on WSL2 with CUDA. See
[Pipeline usage](#如何用這條-pipeline-量化任何模型) below — all scripts take any HF model id.

- 📦 量化成品:[betty0/Qwen2.5-3B-Instruct-GGUF](https://huggingface.co/betty0/Qwen2.5-3B-Instruct-GGUF)
- 📊 完整評測數據:[results/summary.md](results/summary.md)

---

## GGUF 是什麼?

[GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) 是 llama.cpp 生態系的
模型檔案格式:單一檔案裝下權重 + tokenizer + 中繼資料(含 chat template),
專為 **本機推論** 設計。搭配量化後:

- 不需要 Python 環境,一個執行檔就能跑
- CPU 也能推論,有 GPU 則可把任意層數 offload 上去(`-ngl`)
- Ollama、LM Studio、llama.cpp、Jan 等本機推論工具都吃這個格式

## 量化等級差異

量化(quantization)把原本 FP16(每個權重 16 bit)壓到 4~8 bit,
檔案變小、推論變快,代價是些微品質損失。K-quants(`_K_M` 系列)
會對重要權重(如 attention 的部分矩陣)保留較高精度,同樣位元數下品質更好。

| 等級 | 每權重約略位元 | 特性 |
|---|---|---|
| F16 | 16 | 轉檔基準,無量化損失 |
| Q8_0 | 8.5 | 幾乎無損,檔案約 F16 一半 |
| Q5_K_M | 5.7 | 品質/大小平衡,K-quant 混合精度 |
| Q4_K_M | 4.8 | 最常用的甜蜜點:小、快、品質可接受 |

### 本專案實測結果(Qwen2.5-3B-Instruct @ RTX 4090)

實測環境:RTX 4090、`-ngl 99` 全層 GPU、wikitext-2-raw-v1 test(ctx 512)、llama.cpp `4f37f51`。

| 量化等級 | 檔案大小 | PPL ↓ | PPL vs F16 | pp512 tok/s | tg128 tok/s | 峰值 VRAM |
|---|---|---|---|---|---|---|
| F16 | 5.75 GiB | 9.0631 ± 0.064 | 基準 | 19,999 | 127.8 | 8,698 MiB |
| Q8_0 | 3.06 GiB | 9.0806 ± 0.064 | +0.19% | 22,389 | 206.8 | 5,888 MiB |
| Q5_K_M | 2.07 GiB | 9.1883 ± 0.065 | +1.38% | 20,977 | 270.4 | 4,630 MiB |
| Q4_K_M | 1.80 GiB | 9.5741 ± 0.068 | +5.64% | 19,691 | 300.2 | 4,348 MiB |

幾個值得注意的觀察:
- **生成速度與檔案大小成反比**:tg128 是記憶體頻寬瓶頸,Q4_K_M(300 tok/s)≈ F16(128 tok/s)的 2.35 倍。
- **pp512 四者接近**(~2 萬 tok/s):prompt processing 是算力瓶頸,量化影響小。
- **Q8_0 幾乎無損**(+0.19%),Q5_K_M 只差 +1.38%;3B 這種小模型對 Q4 比較敏感(+5.64%),
  參數量越大的模型量化損失通常越小。

完整數據與原始 log:[results/](results/)

## 專案結構

```
gguf-quantization-factory/
├── scripts/
│   ├── setup_env.sh          # 環境建置:CUDA toolkit、編譯 llama.cpp、Python venv
│   ├── pipeline.sh           # 主 pipeline:下載→轉檔→量化→PPL→bench→summary
│   ├── prepare_wikitext.py   # 下載 wikitext-2-raw-v1 test split 當 PPL 語料
│   ├── measure_vram.sh       # 執行期間輪詢 nvidia-smi 記 VRAM 峰值
│   ├── make_summary.py       # 解析 logs → results/summary.md 比較表
│   └── upload_hf.py          # 上傳 GGUF 到 HF(可續傳)+ 自動產 model card
├── ollama/Modelfile          # Ollama 部署檔(Qwen ChatML template + stop tokens)
├── results/                  # 真實評測數據與原始 log
└── README.md
```

## 如何用這條 pipeline 量化任何模型

### 1. 環境建置(一次就好)

需求:WSL2(或任何 Linux)+ NVIDIA GPU + [uv](https://docs.astral.sh/uv/)。

```bash
sudo scripts/setup_env.sh system   # apt 裝 cmake + NVIDIA CUDA toolkit(WSL-Ubuntu repo)
scripts/setup_env.sh build         # clone llama.cpp、cmake -DGGML_CUDA=ON 編譯、venv 裝相依
```

### 2. 跑 pipeline

```bash
# 預設模型 Qwen/Qwen2.5-3B-Instruct,一路跑完:
scripts/pipeline.sh

# 量化任何 HF 模型(llama.cpp 支援的架構都行):
scripts/pipeline.sh meta-llama/Llama-3.2-3B-Instruct

# 也可以只跑單一步驟(download|convert|quantize|wikitext|ppl|bench|summary):
scripts/pipeline.sh Qwen/Qwen2.5-3B-Instruct ppl
```

流程:`hf download` → `convert_hf_to_gguf.py`(F16)→ `llama-quantize`(Q4_K_M/Q5_K_M/Q8_0)
→ `llama-perplexity`(wikitext-2,`-ngl 99`)→ `llama-bench` + VRAM 量測
→ 彙整成 `results/summary.md`。

模型與 GGUF 都放在 `~/gguf-factory/`(可用 `GGUF_WORK_DIR` 改),不進 git。

### 3. Ollama 部署

```bash
# 把 Q4_K_M GGUF 放到 ollama/Modelfile 同層後:
cd ollama
ollama create qwen2.5-3b-local -f Modelfile
ollama run qwen2.5-3b-local "用一句話介紹量化"
```

Modelfile 重點:Qwen2.5 用 ChatML 格式,`TEMPLATE` 要寫 `<|im_start|>` / `<|im_end|>`,
`stop` token 兩個都要設,不然模型會停不下來。實測輸出見
[results/ollama_test.md](results/ollama_test.md)。

### 4. 上傳 Hugging Face

```bash
~/gguf-factory/.venv/bin/python scripts/upload_hf.py \
    --model-id Qwen/Qwen2.5-3B-Instruct \
    --model-dir ~/gguf-factory/models/Qwen2.5-3B-Instruct \
    --summary-json results/summary.json
```

用 `upload_large_folder` 分 chunk 上傳、斷線可續傳;model card(含比較表)自動生成。
token 用 `hf auth login` 或 `HF_TOKEN` 環境變數。

## 來源模型與授權

- 來源模型:[Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)(Apache-2.0)
- llama.cpp:[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)(MIT)
- 本 repo 的腳本:MIT

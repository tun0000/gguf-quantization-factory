#!/usr/bin/env bash
# setup_env.sh — 建置 GGUF 量化工廠的 WSL2 環境
#
# 用法:
#   sudo ./setup_env.sh system   # 階段 1(root):apt 安裝 cmake/ccache + NVIDIA CUDA toolkit
#   ./setup_env.sh build         # 階段 2(user):clone llama.cpp、CUDA 編譯、uv venv 裝相依
#
# 工作目錄預設 ~/gguf-factory,可用環境變數 GGUF_WORK_DIR 覆寫。
set -euo pipefail

WORK_DIR="${GGUF_WORK_DIR:-$HOME/gguf-factory}"
LLAMA_DIR="$WORK_DIR/llama.cpp"
VENV_DIR="$WORK_DIR/.venv"
CUDA_ARCH="${CUDA_ARCH:-89}"   # RTX 4090 = compute capability 8.9

phase_system() {
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y -qq cmake ccache curl wget git

    # NVIDIA CUDA toolkit(WSL-Ubuntu 官方 repo,只裝 toolkit 不碰 driver)
    if ! command -v /usr/local/cuda/bin/nvcc >/dev/null 2>&1; then
        KEYRING=/tmp/cuda-keyring_1.1-1_all.deb
        wget -q -O "$KEYRING" \
            https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
        dpkg -i "$KEYRING"
        apt-get update -qq
        # 挑最新的 12-x toolkit(12.x 對 Ubuntu 24.04 的 gcc-13 相容性最穩)
        PKG=$(apt-cache search --names-only '^cuda-toolkit-12-[0-9]+$' \
              | awk '{print $1}' | sort -t- -k4 -n | tail -1)
        if [ -z "$PKG" ]; then
            echo "找不到 cuda-toolkit-12-x 套件" >&2; exit 1
        fi
        echo "安裝 $PKG ..."
        apt-get install -y -qq "$PKG"
    fi
    /usr/local/cuda/bin/nvcc --version
    echo "=== system phase 完成 ==="
}

phase_build() {
    # 非 login shell 下 uv 常不在 PATH(裝在 ~/.local/bin 或 ~/.cargo/bin)
    export PATH="/usr/local/cuda/bin:$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    mkdir -p "$WORK_DIR"

    # 1. clone llama.cpp
    if [ ! -d "$LLAMA_DIR/.git" ]; then
        git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
    fi
    cd "$LLAMA_DIR"
    git log -1 --format='llama.cpp commit: %h %s (%ci)'

    # 2. CUDA 編譯
    cmake -B build \
        -DGGML_CUDA=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
        -DLLAMA_CURL=OFF
    cmake --build build --config Release -j "$(nproc)" \
        --target llama-cli llama-quantize llama-perplexity llama-bench

    # 3. 驗證 binary
    for b in llama-cli llama-quantize llama-perplexity llama-bench; do
        test -x "$LLAMA_DIR/build/bin/$b" || { echo "缺 $b" >&2; exit 1; }
    done
    "$LLAMA_DIR/build/bin/llama-cli" --version 2>&1 | head -5

    # 4. Python venv(convert 腳本相依 + datasets + huggingface_hub)
    cd "$WORK_DIR"
    [ -d "$VENV_DIR" ] || uv venv "$VENV_DIR"
    # requirements 內含 pytorch extra index,uv 需放寬 index 策略才解得開
    uv pip install --python "$VENV_DIR/bin/python" -q \
        --index-strategy unsafe-best-match \
        -r "$LLAMA_DIR/requirements/requirements-convert_hf_to_gguf.txt" \
        datasets "huggingface_hub[cli]"
    "$VENV_DIR/bin/python" -c "import torch, gguf, datasets, huggingface_hub as h; print('torch', torch.__version__, '| gguf ok | datasets ok | hf_hub', h.__version__)"
    echo "=== build phase 完成 ==="
}

case "${1:-}" in
    system) phase_system ;;
    build)  phase_build ;;
    *) echo "用法: $0 {system|build}" >&2; exit 1 ;;
esac

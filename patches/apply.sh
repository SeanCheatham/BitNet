#!/usr/bin/env bash
# Apply patches to the llama.cpp submodule for BitNet-2B-4T support.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
LLAMA_CPP="$REPO_DIR/3rdparty/llama.cpp/src/llama.cpp"

if [ ! -f "$LLAMA_CPP" ]; then
    echo "ERROR: llama.cpp not found at $LLAMA_CPP"
    echo "Run 'git submodule update --init --recursive' first."
    exit 1
fi

# Cross-platform sed in-place (macOS requires '' argument, GNU does not)
sedi() {
    if [[ "$OSTYPE" == darwin* ]]; then
        sed -i '' "$@"
    else
        sed -i "$@"
    fi
}

# Patch build_bitnet() FFN activation: SiLU -> squared ReLU (relu2)
# BitNet-2B-4T config has hidden_act="relu2" but build_bitnet() defaults to SiLU.
# build_bitnet_158() already uses LLM_FFN_RELU_SQR correctly.
sedi '/struct ggml_cgraph \* build_bitnet() {/,/return gf;/{
  s/LLM_FFN_SILU, LLM_FFN_PAR/LLM_FFN_RELU_SQR, LLM_FFN_PAR/
}' "$LLAMA_CPP"

grep -q 'LLM_FFN_RELU_SQR, LLM_FFN_PAR' "$LLAMA_CPP" || {
    echo "ERROR: RELU_SQR patch was not applied"
    exit 1
}

echo "Patches applied successfully."

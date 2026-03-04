#!/usr/bin/env bash
# Apply patches to the llama.cpp submodule for BitNet-2B-4T support.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
LLAMA_DIR="$REPO_DIR/3rdparty/llama.cpp"
LLAMA_CPP="$LLAMA_DIR/src/llama.cpp"

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

# Fix the BitNet chat template heuristic detection.
# The existing handler checks for tmpl_contains("BITNET") but the HF model's
# template doesn't contain "BITNET" — it uses "| capitalize" and "<|eot_id|>".
sedi 's/tmpl_contains("BITNET")/tmpl_contains("| capitalize") \&\& tmpl_contains("<|eot_id|>")/' "$LLAMA_CPP"

grep -q 'tmpl_contains("| capitalize")' "$LLAMA_CPP" || {
    echo "ERROR: BitNet chat template heuristic patch was not applied"
    exit 1
}

# Fix TL1/TL2 weight scale: pass scale from GGUF _scale tensors to
# ggml_bitnet_transform_tensor instead of reading garbage from buffer offset.
git -C "$LLAMA_DIR" apply "$SCRIPT_DIR/tl-scale-fix.patch"

grep -q 'tl_scale' "$LLAMA_CPP" || {
    echo "ERROR: TL scale fix patch was not applied"
    exit 1
}

# Fix GGUF writer: include _scale tensors as named tensor entries so
# llama.cpp can look them up at model load time (pairs with tl-scale-fix).
git -C "$LLAMA_DIR" apply "$SCRIPT_DIR/gguf-write-scale-tensors.patch"

grep -q 'endswith("_scale")' "$LLAMA_DIR/gguf-py/gguf/gguf_writer.py" && {
    echo "ERROR: gguf-write-scale-tensors patch was not applied"
    exit 1
}

echo "Patches applied successfully."

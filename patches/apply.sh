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

# Fix the BitNet chat template heuristic detection.
# The existing handler checks for tmpl_contains("BITNET") but the HF model's
# template doesn't contain "BITNET" — it uses "| capitalize" and "<|eot_id|>".
sedi 's/tmpl_contains("BITNET")/tmpl_contains("| capitalize") \&\& tmpl_contains("<|eot_id|>")/' "$LLAMA_CPP"

grep -q 'tmpl_contains("| capitalize")' "$LLAMA_CPP" || {
    echo "ERROR: BitNet chat template heuristic patch was not applied"
    exit 1
}

echo "Patches applied successfully."

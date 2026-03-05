#!/usr/bin/env bash
# Apply patches to the llama.cpp submodule for BitNet-2B-4T support.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
LLAMA_DIR="$REPO_DIR/3rdparty/llama.cpp"
LLAMA_CPP="$LLAMA_DIR/src/llama.cpp"
GGML_C="$LLAMA_DIR/ggml/src/ggml.c"

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

# Fix tensor count check: _scale tensors are in the GGUF but the model
# graph never creates them, so done_getting_tensors() must exclude them.
git -C "$LLAMA_DIR" apply "$SCRIPT_DIR/tl-scale-tensor-count.patch"

grep -q 'n_aux' "$LLAMA_CPP" || {
    echo "ERROR: TL scale tensor count patch was not applied"
    exit 1
}

# Fix ggml_nbytes for TL2: use correct three/two split based on BK=96
# instead of hardcoded two_k=256 which over-reports tensor size.
python3 -c "
import sys
with open('$GGML_C', 'r') as f:
    content = f.read()
old = '            nbytes = (tensor->ne[0] - 256) * tensor->ne[1] / 3 * 5 / 8 + 256 * tensor->ne[1] / 2 * 4 / 8;'
new = '''            // TL2 splits K into base-3 (three_k) and base-2 (two_k) parts.
            // BK=96 is the fixed block size for TL2 base-3 encoding.
            const size_t K = tensor->ne[0];
            const size_t M = tensor->ne[1];
            const size_t three_k = (K / 96) * 96;
            const size_t two_k = K - three_k;
            nbytes = three_k * M / 3 * 5 / 8 + two_k * M / 2 * 4 / 8;'''
if old not in content:
    print('ERROR: ggml_nbytes TL2 formula not found in ggml.c', file=sys.stderr)
    sys.exit(1)
content = content.replace(old, new)
with open('$GGML_C', 'w') as f:
    f.write(content)
"

grep -q 'three_k = (K / 96)' "$GGML_C" || {
    echo "ERROR: TL2 nbytes fix was not applied"
    exit 1
}

# Fix two_qlut offset in bs32/bs8/bs1 paths: use two_k/2 (base-2 encoding)
# instead of two_k/3 (base-3, which is wrong for the two-part LUT).
sedi 's|two_k / 3 \* 32|two_k / 2 * 32|g' "$GGML_C"

grep -q 'two_k / 3' "$GGML_C" && {
    echo "ERROR: TL2 two_qlut offset fix was not applied"
    exit 1
}

echo "Patches applied successfully."

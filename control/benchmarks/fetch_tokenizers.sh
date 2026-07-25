#!/usr/bin/env bash
# Download tokenizer files (NOT weights) for each model into ./tokenizers/<alias>/.
# BFCL needs the tokenizer locally to apply the chat template client-side.
#
# This downloads only the small tokenizer-related files (~few MB each), not the
# multi-GB weight shards.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=spark-serve/models.env
source "${SCRIPT_DIR}/spark-serve/models.env"

cd "${SCRIPT_DIR}/aider"
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q huggingface_hub >/dev/null 2>&1 || true

mkdir -p "${SCRIPT_DIR}/tokenizers"

for alias in "${ALL_ALIASES[@]}"; do
  select_model "$alias"
  OUT="${SCRIPT_DIR}/tokenizers/${alias}"
  if [ -f "${OUT}/tokenizer_config.json" ]; then
    echo "[tok] $alias already present, skipping"
    continue
  fi
  echo "[tok] fetching $alias ← $SERVE_REPO"
  python3 - <<PY
from huggingface_hub import snapshot_download
snapshot_download(
    "$SERVE_REPO",
    local_dir="$OUT",
    allow_patterns=[
        "tokenizer*",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "added_tokens.json",
        "chat_template*",
        "config.json",
        "generation_config.json",
    ],
)
PY
done

echo "[tok] all tokenizers ready under ${SCRIPT_DIR}/tokenizers/"

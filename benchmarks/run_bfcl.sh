#!/usr/bin/env bash
# Run BFCL v3 Live AST against the currently-served model on spark.
#
# Caveat: BFCL does prompt-mode tool-call eval — it formats the prompt on the
# client side using a tokenizer-bundled chat template, then sends a raw text
# completion to the /v1/completions endpoint and parses tool calls from the
# response text. This bypasses vLLM's `--tool-call-parser` entirely.
# Result: BFCL needs the model's tokenizer cached LOCALLY, and the closest-fit
# BFCL registry entry must use a chat template compatible with what the served
# model expects. For Qwen3.6 weights served by vLLM, BFCL's existing
# `Qwen/Qwen3-32B-FC` template is the nearest match (Qwen3 → Qwen3.5/3.6
# templates are very similar). For MiniMax-M2 there is no upstream entry; we
# fall back to QuickTestingOSSHandler which uses the local tokenizer's own
# chat_template.
#
# This script assumes you've already (1) installed BFCL via uv in
# benchmarks/gorilla/.../.venv, (2) downloaded the model tokenizer files
# locally to ${TOKENIZERS_DIR}/${alias}, and (3) added the matching registry
# entry in bfcl_eval/constants/model_config.py.
#
# Usage: ./run_bfcl.sh <alias> [--category live_ast]
set -euo pipefail

ALIAS="${1:?alias required}"; shift || true
CATEGORY="${CATEGORY:-live}"  # 'live' covers live_simple/live_multiple/live_parallel/etc.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=spark-serve/models.env
source "${SCRIPT_DIR}/spark-serve/models.env"
select_model "$ALIAS"

cd "${SCRIPT_DIR}/gorilla/berkeley-function-call-leaderboard"
# shellcheck disable=SC1091
source .venv/bin/activate

# Map our alias to nearest BFCL registry entry. The served-model-name on vLLM
# must match the .model_name field in that registry entry (i.e. what BFCL
# passes to the API). For Qwen we strip the -FC suffix. For MiniMax the model
# id must match too.
case "$ALIAS" in
  qwen36-27b-*)        BFCL_MODEL="Qwen/Qwen3-32B-FC" ;;
  qwen36-35b-a3b-*)    BFCL_MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507-FC" ;;
  qwen35-122b-a10b-*)  BFCL_MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507-FC" ;;  # nearest Qwen3.x MoE-instruct FC template
  minimax)             BFCL_MODEL="spark/minimax-m2.7-reap-FC" ;;  # needs custom entry
  *)                   echo "no BFCL mapping for $ALIAS" >&2; exit 1 ;;
esac

TOK_DIR="${TOKENIZERS_DIR:-${SCRIPT_DIR}/tokenizers}/${ALIAS}"
if [ ! -d "$TOK_DIR" ]; then
  echo "[bfcl] tokenizer dir missing: $TOK_DIR" >&2
  echo "       run ./fetch_tokenizers.sh first" >&2
  exit 2
fi

echo "[bfcl] alias=$ALIAS bfcl_model=$BFCL_MODEL served=$SERVE_NAME category=$CATEGORY"

# Clear stale result files so bfcl generate never skips categories it already sees on disk.
BFCL_DIR="${BFCL_MODEL//\//_}"
rm -rf "result/$BFCL_DIR"
echo "[bfcl] cleared stale result dir: result/$BFCL_DIR"

# REMOTE_OPENAI_BASE_URL: default to spark; caller (sweep_bfcl_local.sh) overrides for local.
# REMOTE_OPENAI_MODEL_NAME: actual served-model-name on vLLM, which differs from the BFCL
# registry model_name (HuggingFace id). Without this the API call sends the wrong model id.
REMOTE_OPENAI_BASE_URL="${REMOTE_OPENAI_BASE_URL:-http://gx10-a6c7.lan:8000/v1}" \
REMOTE_OPENAI_API_KEY="dummy" \
REMOTE_OPENAI_TOKENIZER_PATH="$TOK_DIR" \
REMOTE_OPENAI_MODEL_NAME="$SERVE_NAME" \
  bfcl generate \
    --model "$BFCL_MODEL" \
    --test-category "$CATEGORY" \
    --skip-server-setup \
    --num-threads 8 \
    --backend vllm

# Score
bfcl evaluate \
  --model "$BFCL_MODEL" \
  --test-category "$CATEGORY"

# Copy results to a per-alias dir for aggregation
# BFCL stores under <model_name_with_slashes_replaced_by_underscores>
RESULTS_OUT="${SCRIPT_DIR}/results/bfcl/${ALIAS}-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RESULTS_OUT"
cp -r result/"$BFCL_DIR"/* "$RESULTS_OUT/" 2>/dev/null || true
cp -r score/"$BFCL_DIR"/* "$RESULTS_OUT/" 2>/dev/null || true
echo "[bfcl] results: $RESULTS_OUT"

#!/usr/bin/env bash
# SGLang launch for google/gemma-4-31B-it + z-lab/gemma-4-31B-it-DFlash on RTX PRO 6000.
# Run via:
#   ./serve.sh sglang gemma4_31b_dflash
# Tune via:
#   ./benchmarks/tune-dflash-extended.sh <model-dir>
set -euo pipefail

MODEL="${MODEL_PATH:?MODEL_PATH is required}"
DRAFTER="${DRAFTER_PATH:?DRAFTER_PATH is required}"
SERVED="${SERVED:?SERVED is required}"
MAXLEN="${MAXLEN:-65536}"
NUM_SPEC="${NUM_SPEC:-16}"
DRAFT_WINDOW="${DRAFT_WINDOW:-2048}"
MEM_FRACTION="${MEM_FRACTION:-0.78}"
ATTN_BACKEND="${ATTN_BACKEND:-triton}"
CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-8}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-2}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-16384}"
MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-32768}"
PORT=8000

# Do not use PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True here: on this WSL/CUDA13/SM120
# stack it causes `CUDA driver error: unknown error` even for a one-element CUDA tensor.
export SGLANG_ENABLE_SPEC_V2="${SGLANG_ENABLE_SPEC_V2:-1}"

DWIN_ARG=()
case "${DRAFT_WINDOW}" in off|0|none) ;; *) DWIN_ARG=(--speculative-draft-window-size "${DRAFT_WINDOW}") ;; esac

exec python3 -m sglang.launch_server \
  --model-path "$MODEL" \
  --served-model-name "$SERVED" \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path "$DRAFTER" \
  --speculative-num-draft-tokens "$NUM_SPEC" \
  "${DWIN_ARG[@]}" \
  --quantization modelopt \
  --tp-size 1 \
  --attention-backend "$ATTN_BACKEND" \
  --speculative-draft-attention-backend fa4 \
  --speculative-draft-model-quantization unquant \
  --mem-fraction-static "$MEM_FRACTION" \
  --cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS" \
  --max-running-requests "$MAX_RUNNING_REQUESTS" \
  --chunked-prefill-size "$CHUNKED_PREFILL_SIZE" \
  --max-prefill-tokens "$MAX_PREFILL_TOKENS" \
  --context-length "$MAXLEN" \
  --trust-remote-code \
  --enable-metrics \
  --host 0.0.0.0 --port "$PORT" \
  "$@"

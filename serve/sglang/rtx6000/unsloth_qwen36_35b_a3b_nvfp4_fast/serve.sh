#!/usr/bin/env bash
# SGLang launch for pinned Unsloth 35B-A3B NVFP4 Fast target with matching DFlash drafter.
set -euo pipefail

MODEL="${MODEL_PATH:?MODEL_PATH is required}"
DRAFTER="${DRAFTER_PATH:?DRAFTER_PATH is required}"
SERVED="${SERVED:?SERVED is required}"
MAXLEN="${MAXLEN:-262144}"
NUM_SPEC="${NUM_SPEC:-8}"
DRAFT_WINDOW="${DRAFT_WINDOW:-6144}"
MEM_FRACTION="${MEM_FRACTION:-0.82}"
ATTN_BACKEND="${ATTN_BACKEND:-flashinfer}"
CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-8}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-2}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-16384}"
MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-32768}"
PORT=8000

DWIN_ARG=()
case "${DRAFT_WINDOW}" in off|0|none) ;; *) DWIN_ARG=(--speculative-draft-window-size "${DRAFT_WINDOW}") ;; esac
PARSER_ARG=(--reasoning-parser qwen3 --tool-call-parser qwen3_coder)
[ "${PARSERS:-1}" = "1" ] || PARSER_ARG=()
if [ "${RADIX:-1}" = "1" ]; then
  CACHE_ARG=()
  MAMBA_ARG=(--mamba-scheduler-strategy "${MAMBA_STRATEGY:-extra_buffer}")
  export SGLANG_ENABLE_SPEC_V2="${SGLANG_ENABLE_SPEC_V2:-1}"
else
  CACHE_ARG=(--disable-radix-cache)
  MAMBA_ARG=(--mamba-scheduler-strategy "${MAMBA_STRATEGY:-auto}")
fi

exec python3 -m sglang.launch_server \
  --model-path "$MODEL" \
  --served-model-name "$SERVED" \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path "$DRAFTER" \
  --speculative-num-draft-tokens "$NUM_SPEC" \
  "${DWIN_ARG[@]}" \
  --tp-size 1 \
  --attention-backend "$ATTN_BACKEND" \
  --mem-fraction-static "$MEM_FRACTION" \
  --cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS" \
  --max-running-requests "$MAX_RUNNING_REQUESTS" \
  --chunked-prefill-size "$CHUNKED_PREFILL_SIZE" \
  --max-prefill-tokens "$MAX_PREFILL_TOKENS" \
  --context-length "$MAXLEN" \
  "${MAMBA_ARG[@]}" \
  "${CACHE_ARG[@]}" \
  --trust-remote-code \
  --enable-metrics \
  --host 0.0.0.0 --port "$PORT" \
  "${PARSER_ARG[@]}" \
  "$@"

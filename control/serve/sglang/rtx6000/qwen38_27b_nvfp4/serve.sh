#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL_PATH:?MODEL_PATH is required}"
SERVED="${SERVED:?SERVED is required}"
MAXLEN="${MAXLEN:-262144}"
MEM_FRACTION="${MEM_FRACTION:-0.85}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-4}"
MAX_MAMBA_CACHE_SIZE="${MAX_MAMBA_CACHE_SIZE:-32}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-2048}"
PORT=8000

export DGX_MODEL_CAPABILITIES_PATH=/run/inference/package/capabilities.json
export MAX_MODEL_LEN="$MAXLEN"

exec python3 -m sglang_with_capabilities \
  --model-path "$MODEL" \
  --served-model-name "$SERVED" \
  --trust-remote-code \
  --mem-fraction-static "$MEM_FRACTION" \
  --attention-backend flashinfer \
  --chunked-prefill-size "$CHUNKED_PREFILL_SIZE" \
  --kv-cache-dtype fp8_e4m3 \
  --mm-feature-transport cpu \
  --context-length "$MAXLEN" \
  --mamba-radix-cache-strategy extra_buffer_lazy \
  --max-mamba-cache-size "$MAX_MAMBA_CACHE_SIZE" \
  --max-running-requests "$MAX_RUNNING_REQUESTS" \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --sampling-defaults model \
  --enable-metrics \
  --host 0.0.0.0 --port "$PORT" \
  "$@"

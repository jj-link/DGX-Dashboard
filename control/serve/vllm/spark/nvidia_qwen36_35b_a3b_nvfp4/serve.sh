#!/usr/bin/env bash
set -euo pipefail
export VLLM_USE_V2_MODEL_RUNNER=1

args=(
  vllm
  serve
  "${MODEL_PATH}"
  --served-model-name
  "${SERVED}"
  --host
  0.0.0.0
  --port
  8000
  --trust-remote-code
  --max-model-len
  262144
  --gpu-memory-utilization
  0.85
  --max-num-batched-tokens
  262144
  --max-num-seqs
  4
  --attention-backend
  TRITON_ATTN
)
exec "${args[@]}" "$@"

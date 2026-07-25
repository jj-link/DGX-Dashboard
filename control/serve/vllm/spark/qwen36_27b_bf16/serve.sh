#!/usr/bin/env bash
set -euo pipefail

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
  --max-model-len
  262144
  --gpu-memory-utilization
  0.85
  --max-num-batched-tokens
  262144
)
exec "${args[@]}" "$@"

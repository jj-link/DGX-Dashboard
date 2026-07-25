#!/usr/bin/env bash
set -euo pipefail
export VLLM_USE_DEEP_GEMM=0

args=(
  vllm
  serve
  "${MODEL_PATH}"
  --served-model-name
  "${SERVED}"
  --trust-remote-code
  --spec-method
  dspark
  --spec-tokens
  5
  --kv-cache-dtype
  fp8
  --host
  0.0.0.0
  --port
  8000
)
exec "${args[@]}" "$@"

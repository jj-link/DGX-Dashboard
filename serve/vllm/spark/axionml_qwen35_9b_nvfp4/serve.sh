#!/usr/bin/env bash
set -euo pipefail
export VLLM_USE_V2_MODEL_RUNNER=1

args=(
  vllm
  serve
  "${MODEL_PATH}"
  --host
  0.0.0.0
  --port
  8000
  --trust-remote-code
  --max-num-seqs
  8
  --max-cudagraph-capture-size
  8
  --attention-backend
  TRITON_ATTN
  --served-model-name
  "${SERVED}"
)
exec "${args[@]}" "$@"

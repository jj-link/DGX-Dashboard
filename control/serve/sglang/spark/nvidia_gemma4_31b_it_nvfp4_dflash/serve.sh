#!/usr/bin/env bash
set -euo pipefail
export FLASHINFER_DISABLE_VERSION_CHECK=1
export SGLANG_ENABLE_JIT_DEEPGEMM=0
export SGLANG_ENABLE_SPEC_V2=1

args=(
  python3
  -m
  sglang.launch_server
  --model-path
  "${MODEL_PATH}"
  --host
  0.0.0.0
  --port
  8000
  --served-model-name
  "${SERVED}"
  --trust-remote-code
  --tp-size
  1
  --attention-backend
  triton
  --context-length
  262144
  --speculative-algorithm
  DFLASH
  --speculative-draft-model-path
  "${DRAFTER_PATH}"
  --speculative-draft-attention-backend
  fa4
  --speculative-num-draft-tokens
  8
  --speculative-draft-window-size
  8192
  --mamba-scheduler-strategy
  extra_buffer
  --mem-fraction-static
  0.85
  --enable-metrics
  --max-running-requests
  4
  --chunked-prefill-size
  2048
)
exec "${args[@]}" "$@"

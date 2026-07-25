#!/usr/bin/env bash
set -euo pipefail
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
  flashinfer
  --context-length
  262144
  --speculative-algorithm
  DFLASH
  --speculative-draft-model-path
  "${DRAFTER_PATH}"
  --speculative-num-draft-tokens
  8
  --speculative-draft-window-size
  8192
  --mamba-scheduler-strategy
  extra_buffer
  --tool-call-parser
  qwen3_coder
  --reasoning-parser
  qwen3
  --mem-fraction-static
  0.85
  --enable-metrics
  --max-running-requests
  4
  --chunked-prefill-size
  2048
)
exec "${args[@]}" "$@"

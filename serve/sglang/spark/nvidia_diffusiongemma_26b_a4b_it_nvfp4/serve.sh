#!/usr/bin/env bash
set -euo pipefail

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
  131072
  --mem-fraction-static
  0.85
  --enable-metrics
  --max-running-requests
  4
  --chunked-prefill-size
  2048
)
exec "${args[@]}" "$@"

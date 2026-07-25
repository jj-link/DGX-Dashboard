#!/usr/bin/env bash
set -euo pipefail

args=(
  vllm
  serve
  "${MODEL_PATH}"
  --served-model-name
  "${SERVED}"
  --max-model-len
  131072
  --reasoning-parser
  minimax_m2_append_think
  --tool-call-parser
  minimax_m2
  --host
  0.0.0.0
  --port
  8000
  --trust-remote-code
)
exec "${args[@]}" "$@"

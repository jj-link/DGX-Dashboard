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
  --quantization
  modelopt_fp4
  --kv-cache-dtype
  fp8_e4m3
  --reasoning-parser
  gemma4
  --tool-call-parser
  gemma4
  --mem-fraction-static
  0.80
  --host
  0.0.0.0
  --port
  8000
  --enable-metrics
  --max-running-requests
  4
  --cuda-graph-max-bs
  8
  --served-model-name
  "${SERVED}"
)
exec "${args[@]}" "$@"

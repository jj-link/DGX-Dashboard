#!/usr/bin/env bash
set -euo pipefail

args=(
  vllm
  serve
  "${MODEL_PATH}"
  --tokenizer
  "${TOKENIZER_PATH}"
  --served-model-name
  "${SERVED}"
  --max-model-len
  131072
  --reasoning-parser
  qwen3
  --tool-call-parser
  qwen3_xml
  --host
  0.0.0.0
  --port
  8000
  --trust-remote-code
)
exec "${args[@]}" "$@"

#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi
API_BASE="${API_BASE:-http://${BIND_HOST:-100.86.3.45}:${HOST_PORT:-8000}}"

curl -fsS "${API_BASE}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "thinkingcap-qwen36-27b-nvfp4-dflash8",
    "messages": [
      {
        "role": "user",
        "content": "Reply with exactly: OK"
      }
    ],
    "max_tokens": 64,
    "chat_template_kwargs": {"enable_thinking": false},
    "temperature": 0
  }'

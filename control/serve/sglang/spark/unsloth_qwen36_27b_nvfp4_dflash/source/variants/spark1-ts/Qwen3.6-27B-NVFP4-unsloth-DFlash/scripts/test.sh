#!/usr/bin/env bash
set -euo pipefail

curl -fsS http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3.6-27B-NVFP4-UNSLOTH-DFLASH",
    "messages": [
      {
        "role": "user",
        "content": "Reply with exactly: ok"
      }
    ],
    "max_tokens": 8,
    "chat_template_kwargs": {"enable_thinking": false},
    "temperature": 0
  }'

#!/usr/bin/env bash
set -euo pipefail

curl -fsS http://localhost:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "unsloth/Qwen3.6-35B-A3B-FP8",
    "messages": [
      {
        "role": "user",
        "content": "Reply with exactly: ok"
      }
    ],
    "max_tokens": 8,
    "temperature": 0
  }'

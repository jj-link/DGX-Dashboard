#!/usr/bin/env bash
set -euo pipefail

curl -fsS http://localhost:30000/v1/chat/completions   -H 'Content-Type: application/json'   -d '{
    model: r0b0tlab/nex-n2-mini-nvfp4,
    messages: [
      {
        role: user,
        content: Reply with exactly: ok
      }
    ],
    max_tokens: 8,
    temperature: 0
  }'

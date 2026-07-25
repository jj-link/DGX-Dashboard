#!/usr/bin/env bash
set -euo pipefail

# SGLang launch for Qwen2.5-7B-Instruct-AWQ on RTX 4090 (GPU 1)
# Auxiliary model for vision, compression, web extraction tasks.

# Upgrade Transformers for Gemma-4 support (idempotent)
pip install --upgrade --no-deps --break-system-packages transformers 2>&1 | tail -3

# Launch SGLang server
exec python3 -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-7B-Instruct-AWQ \
  --quantization awq \
  --port 8001 \
  --host 0.0.0.0 \
  --context-length 32768 \
  --mem-fraction-static 0.85 \
  --kv-cache-dtype fp8_e5m2 \
  --trust-remote-code \
  --skip-server-warmup

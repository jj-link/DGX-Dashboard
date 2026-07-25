#!/usr/bin/env bash
# SGLang launch for Gemma-4-12B-IT AWQ-INT4 on RTX 4090 (GPU 1)
# Auxiliary model for vision, compression, web extraction tasks.
# Run via the hardened profile runtime:
#   INFERENCE_PROFILE=rtx6000 CUDA_VISIBLE_DEVICES=1 HOST_PORT=8001 ./serve.sh sglang gemma4_12b_awq
#
# The pinned package image installs Transformers 5.12.0 for gemma4_unified
# support; the launcher performs no runtime package mutation.
#
# VRAM math (AWQ INT4 weights + FP8 KV cache):
#   - 12B AWQ INT4 weights: ~6-7 GB
#   - FP8 KV cache at 32k context: ~4-5 GB (vs ~8-10 GB with BF16)
#   - CUDA graphs (bs=8): ~1 GB
#   - Total at rest: ~12 GB / 24 GB -> plenty of headroom
set -euo pipefail

MODEL="${MODEL_PATH:?MODEL_PATH is required}"
SERVED="${SERVED:?SERVED is required}"
MAXLEN="${MAXLEN:-32768}"
MEM_FRACTION="${MEM_FRACTION:-0.85}"
CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-8}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-8}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-16384}"
MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-65536}"
PORT=8000

# Upgrade Transformers to 5.12.0 (supports gemma4_unified)

exec python3 -m sglang.launch_server \
  --model-path "$MODEL" \
  --served-model-name "$SERVED" \
  --quantization compressed-tensors \
  --kv-cache-dtype fp8_e4m3 \
  --tp-size 1 \
  --attention-backend flashinfer \
  --mem-fraction-static "$MEM_FRACTION" \
  --cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS" \
  --max-running-requests "$MAX_RUNNING_REQUESTS" \
  --chunked-prefill-size "$CHUNKED_PREFILL_SIZE" \
  --max-prefill-tokens "$MAX_PREFILL_TOKENS" \
  --context-length "$MAXLEN" \
  --trust-remote-code \
  --enable-metrics \
  --host 0.0.0.0 --port "$PORT" \
  "$@"

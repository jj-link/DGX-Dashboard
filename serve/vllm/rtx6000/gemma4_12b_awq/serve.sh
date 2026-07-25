#!/usr/bin/env bash
# vLLM launch for Gemma-4-12B-IT AWQ-INT4 on RTX 4090 (GPU 1)
# Auxiliary model for vision, compression, web extraction tasks.
# Run via the hardened profile runtime:
#   INFERENCE_PROFILE=rtx6000 CUDA_VISIBLE_DEVICES=1 HOST_PORT=8001 ./serve.sh vllm gemma4_12b_awq
#
# VRAM math (AWQ INT4 weights + FP8 KV cache):
#   - 12B AWQ INT4 weights: ~6-7 GB
#   - FP8 KV cache at 32k context: ~4-5 GB (vs ~8-10 GB with BF16)
#   - CUDA graphs (bs=8): ~1 GB
#   - Total at rest: ~12 GB / 24 GB -> plenty of headroom
set -euo pipefail

MODEL="${MODEL_PATH:?MODEL_PATH is required}"
SERVED="${SERVED:?SERVED is required}"
PORT=8000
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
MAXLEN="${MAXLEN:-32768}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"

# Gemma-4 is multimodal (any-to-any) - this is perfect for vision tasks.
# vLLM supports it natively with the gemma image.

exec python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name "$SERVED" \
  --quantization awq \
  --dtype auto \
  --max-model-len "$MAXLEN" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --kv-cache-dtype "$KV_CACHE_DTYPE" \
  --host 0.0.0.0 --port "$PORT" \
  --trust-remote-code \
  "$@"

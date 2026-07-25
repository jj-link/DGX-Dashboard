#!/usr/bin/env bash
# vLLM launch for Gemma-4-12B-IT AWQ-INT4 on RTX 4090 (GPU 1)
# Auxiliary model for vision, compression, web extraction tasks.
# Run via the hardened wrapper:
#   CUDA_VISIBLE_DEVICES=1 IMAGE=vllm/vllm-openai:gemma PORT=8001 ./run_vllm_docker.sh serve/vllm/serve_gemma4_12b_awq.sh
#
# VRAM math (AWQ INT4 weights + FP8 KV cache):
#   - 12B AWQ INT4 weights: ~6-7 GB
#   - FP8 KV cache at 32k context: ~4-5 GB (vs ~8-10 GB with BF16)
#   - CUDA graphs (bs=8): ~1 GB
#   - Total at rest: ~12 GB / 24 GB -> plenty of headroom
set -u

MODEL="${MODEL:-cyankiwi/gemma-4-12B-it-AWQ-INT4}"
SERVED="${SERVED:-gemma-4-12b-it-awq}"
PORT="${PORT:-8001}"
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
  --trust-remote-code

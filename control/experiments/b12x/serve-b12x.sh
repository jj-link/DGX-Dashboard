#!/usr/bin/env bash
# Isolated candidate launch; does not call or modify production launchers.
set -euo pipefail

IMAGE="${IMAGE:-qwen-vllm:b12x-fi0614}"
CONTAINER_NAME="${CONTAINER_NAME:-vllm-b12x-fi0614-candidate}"
PORT="${PORT:-8000}"
HF_CACHE="${HF_CACHE:-${HOME}/.cache/huggingface}"
CACHE_VOL="${CACHE_VOL:-vllm-b12x-fi0614-cache}"
MODEL="${MODEL:-nvidia/Qwen3.6-27B-NVFP4}"
DRAFTER="${DRAFTER:-z-lab/Qwen3.6-27B-DFlash}"
SERVED="${SERVED:-nvidia-qwen36-27b-nvfp4-b12x}"
MAXLEN="${MAXLEN:-262144}"
NUM_SPEC="${NUM_SPEC:-12}"
MAX_NUM_BATCHED="${MAX_NUM_BATCHED:-32768}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.88}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-bfloat16}"

RUNMODE=(-it)
[ -n "${DETACH:-}" ] && RUNMODE=(-d)

# Both the image default and this explicit runtime value select the PR-#40082
# dense backend. The verifier proves this maps to mm_fp4(backend="b12x").
exec docker run --rm "${RUNMODE[@]}" \
  --device nvidia.com/gpu=all \
  --name "${CONTAINER_NAME}" \
  --security-opt no-new-privileges \
  --cap-drop ALL \
  --pids-limit 4096 \
  -p "127.0.0.1:${PORT}:${PORT}" \
  --tmpfs /tmp:rw,nosuid,nodev,size=4g \
  -e HOME=/cache \
  -v "${CACHE_VOL}:/cache" \
  -e HF_HOME=/models \
  -v "${HF_CACHE}:/models:ro" \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_USE_FLASHINFER_SAMPLER=0 \
  -e VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=0 \
  -e VLLM_ALLREDUCE_USE_FLASHINFER=0 \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  -e VLLM_NVFP4_GEMM_BACKEND=flashinfer-b12x \
  --entrypoint vllm \
  "${IMAGE}" serve "${MODEL}" \
  --served-model-name "${SERVED}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --trust-remote-code \
  --max-model-len "${MAXLEN}" \
  --kv-cache-dtype "${KV_CACHE_DTYPE}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED}" \
  --attention-config '{"backend":"FLASH_ATTN"}' \
  --speculative-config "{\"method\":\"dflash\",\"model\":\"${DRAFTER}\",\"num_speculative_tokens\":${NUM_SPEC},\"attention_backend\":\"FLASH_ATTN\"}" \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --enable-log-requests

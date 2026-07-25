#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-vllm/vllm-openai@sha256:251eba5cc7c12fed0b75da22a9240e582b1c9e39f6fbc064f86781b963bd814f}"
MODEL_HOST="${MODEL_HOST:-/mnt/c/Users/josep/Models/unsloth-qwen36/Qwen3.6-27B-NVFP4-9c295353}"
HF_CACHE="${HF_CACHE:-/home/workbench/.cache/huggingface}"
CONFIG="${CONFIG:-baseline}"
PORT="${PORT:-8000}"
SERVED="${SERVED:-unsloth-qwen36-27b-nvfp4}"
CONTAINER_NAME="${CONTAINER_NAME:-unsloth-qwen36-vllm-27b-${CONFIG}}"
CACHE_VOLUME="${CACHE_VOLUME:-unsloth-qwen36-vllm-024-cache}"
NUM_SPEC="${NUM_SPEC:-2}"
DRAFTER="${DRAFTER:-/hf/hub/models--z-lab--Qwen3.6-27B-DFlash/snapshots/0919688658996800f86b895034249700e9481106}"
DRAFTER_HOST="${DRAFTER_HOST:-${HF_CACHE}/hub/models--z-lab--Qwen3.6-27B-DFlash/snapshots/0919688658996800f86b895034249700e9481106}"

[ -d "${MODEL_HOST}" ] || { echo "missing pinned model: ${MODEL_HOST}" >&2; exit 1; }
[ -d "${HF_CACHE}" ] || { echo "missing pinned Hugging Face cache: ${HF_CACHE}" >&2; exit 1; }
SPEC_ARGS=()
case "${CONFIG}" in
  baseline) ;;
  mtp) SPEC_ARGS=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${NUM_SPEC}}") ;;
  dflash)
    [ -d "${DRAFTER_HOST}" ] || { echo "missing pinned draft model: ${DRAFTER_HOST}" >&2; exit 1; }
    SPEC_ARGS=(--speculative-config "{\"method\":\"dflash\",\"model\":\"${DRAFTER}\",\"num_speculative_tokens\":${NUM_SPEC}}")
    ;;
  *) echo "CONFIG must be baseline, mtp, or dflash" >&2; exit 2 ;;
esac
RUN_MODE=(-it)
[ -n "${DETACH:-}" ] && RUN_MODE=(-d)

exec docker run "${RUN_MODE[@]}" \
  --device nvidia.com/gpu=all \
  --name "${CONTAINER_NAME}" \
  --security-opt no-new-privileges \
  --cap-drop ALL \
  --pids-limit 4096 \
  --shm-size 16g \
  -p "127.0.0.1:${PORT}:${PORT}" \
  --tmpfs /tmp:rw,nosuid,nodev,exec,size=8g \
  -e HOME=/cache \
  -e HF_HOME=/hf \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor \
  -e TRITON_CACHE_DIR=/cache/triton \
  -v "${CACHE_VOLUME}:/cache" \
  -v "${MODEL_HOST}:/model:ro" \
  -v "${HF_CACHE}:/hf:ro" \
  --entrypoint vllm \
  "${IMAGE}" serve /model \
  --served-model-name "${SERVED}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --max-model-len 8192 \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-8192}" \
  --max-num-seqs "${MAX_NUM_SEQS:-4}" \
  --kv-cache-dtype "${KV_CACHE_DTYPE:-bfloat16}" \
  --mamba-cache-dtype "${MAMBA_CACHE_DTYPE:-float32}" \
  --mamba-ssm-cache-dtype "${MAMBA_SSM_CACHE_DTYPE:-float32}" \
  --mamba-cache-mode align \
  --attention-backend "${ATTENTION_BACKEND:-auto}" \
  --enable-chunked-prefill \
  --no-enable-prefix-caching \
  --generation-config vllm \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --enable-log-requests \
  "${SPEC_ARGS[@]}"

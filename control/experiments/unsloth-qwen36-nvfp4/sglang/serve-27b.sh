#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-sha256:8c50d8465642f41fa43ac96dec1449c3ac4f377888d6f3dd6f0635d8dfcb9924}"
MODEL_HOST="${MODEL_HOST:-/mnt/c/Users/josep/Models/unsloth-qwen36/Qwen3.6-27B-NVFP4-9c295353}"
HF_CACHE="${HF_CACHE:-/home/workbench/.cache/huggingface}"
CONFIG="${CONFIG:-baseline}"
PORT="${PORT:-8000}"
SERVED="${SERVED:-unsloth-qwen36-27b-nvfp4}"
CONTAINER_NAME="${CONTAINER_NAME:-unsloth-qwen36-sglang-27b-${CONFIG}}"
CACHE_VOLUME="${CACHE_VOLUME:-unsloth-qwen36-sglang-0514-cache}"
NUM_SPEC="${NUM_SPEC:-4}"
SPEC_STEPS="${SPEC_STEPS:-3}"
DRAFTER="${DRAFTER:-z-lab/Qwen3.6-27B-DFlash}"
DRAFTER_REVISION="${DRAFTER_REVISION:-0919688658996800f86b895034249700e9481106}"
DRAFTER_HOST="${DRAFTER_HOST:-${HF_CACHE}/hub/models--z-lab--Qwen3.6-27B-DFlash/snapshots/${DRAFTER_REVISION}}"

[ -d "${MODEL_HOST}" ] || { echo "missing pinned model: ${MODEL_HOST}" >&2; exit 1; }
[ -d "${HF_CACHE}" ] || { echo "missing pinned Hugging Face cache: ${HF_CACHE}" >&2; exit 1; }
SPEC_ARGS=()
SPEC_ENV=()
QUANT_ARGS=(--quantization compressed-tensors)
case "${CONFIG}" in
  baseline) ;;
  mtp)
    SPEC_ENV=(-e SGLANG_ENABLE_SPEC_V2=1)
    SPEC_ARGS=(--speculative-algorithm NEXTN --speculative-num-steps "${SPEC_STEPS}" --speculative-eagle-topk 1 --speculative-num-draft-tokens "${NUM_SPEC}")
    ;;
  dflash)
    [ -d "${DRAFTER_HOST}" ] || { echo "missing pinned draft model: ${DRAFTER_HOST}" >&2; exit 1; }
    QUANT_ARGS=()
    SPEC_ENV=(-e SGLANG_ENABLE_SPEC_V2=1 -e SGLANG_ENABLE_DFLASH_SPEC_V2=1 -e SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1)
    SPEC_ARGS=(--speculative-algorithm DFLASH --speculative-draft-model-path "${DRAFTER}" --speculative-draft-model-revision "${DRAFTER_REVISION}" --speculative-num-draft-tokens "${NUM_SPEC}" --speculative-draft-window-size "${DRAFT_WINDOW:-2048}" --speculative-draft-attention-backend flashinfer)
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
  --shm-size 32g \
  -p "127.0.0.1:${PORT}:${PORT}" \
  --tmpfs /tmp:rw,nosuid,nodev,exec,size=16g \
  -e HOME=/cache \
  -e HF_HOME=/hf \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor \
  -e TRITON_CACHE_DIR=/cache/triton \
  "${SPEC_ENV[@]}" \
  -v "${CACHE_VOLUME}:/cache" \
  -v "${MODEL_HOST}:/model:ro" \
  -v "${HF_CACHE}:/hf:ro" \
  --entrypoint python3 \
  "${IMAGE}" -m sglang.launch_server \
  --model-path /model \
  --served-model-name "${SERVED}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --tp-size 1 \
  --context-length 8192 \
  --mem-fraction-static "${MEM_FRACTION_STATIC:-0.88}" \
  --max-running-requests "${MAX_RUNNING_REQUESTS:-4}" \
  --chunked-prefill-size "${CHUNKED_PREFILL_SIZE:-8192}" \
  --max-prefill-tokens "${MAX_PREFILL_TOKENS:-8192}" \
  "${QUANT_ARGS[@]}" \
  --kv-cache-dtype "${KV_CACHE_DTYPE:-bfloat16}" \
  --attention-backend "${ATTENTION_BACKEND:-flashinfer}" \
  --linear-attn-prefill-backend "${LINEAR_ATTN_PREFILL_BACKEND:-triton}" \
  --linear-attn-decode-backend "${LINEAR_ATTN_DECODE_BACKEND:-flashinfer}" \
  --fp4-gemm-backend "${FP4_GEMM_BACKEND:-auto}" \
  --sampling-backend "${SAMPLING_BACKEND:-flashinfer}" \
  --mamba-ssm-dtype "${MAMBA_SSM_DTYPE:-bfloat16}" \
  --disable-radix-cache \
  --sampling-defaults openai \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --enable-metrics \
  --log-requests \
  --log-level info \
  "${SPEC_ARGS[@]}"

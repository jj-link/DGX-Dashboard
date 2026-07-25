#!/usr/bin/env bash
# Run an SGLang serve/sglang/*.sh launch script inside a hardened container.
# Mirror of run_vllm_docker.sh -- same hardening rationale: cap-drop ALL +
# no-new-privileges (a compromised dep can't escalate), HF cache mounted
# READ-ONLY (can't poison weights), only the API port published, no $HOME or
# secrets mounted. Network egress is still allowed (see run_vllm_docker.sh note).
#
#   ./run_sglang_docker.sh serve/sglang/serve_qwen36_27b_fp8_dflash.sh
#   ./serve.sh sglang 27b_fp8_dflash            # easier
set -euo pipefail

SERVE_SCRIPT="${1:?usage: run_sglang_docker.sh <serve/sglang/script.sh>}"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
[ -f "${PROJECT_DIR}/${SERVE_SCRIPT}" ] || { echo "no such script: ${SERVE_SCRIPT}" >&2; exit 1; }

# SGLang >= 0.5.10 needed for Qwen3.6 (qwen3_5.py). No SM120/RTX-specific image
# exists; the generic cu130 (CUDA 13) image is the right base for Blackwell.
# Override with IMAGE=. If FP8/attention break on SM120, the script header lists
# an unofficial community SM120 image to try.
case "${SERVE_SCRIPT}" in
  *unsloth_qwen36*)             DEFAULT_IMAGE="sha256:8c50d8465642f41fa43ac96dec1449c3ac4f377888d6f3dd6f0635d8dfcb9924" ;;
  *nvidia_qwen36_27b_nvfp4*) DEFAULT_IMAGE="sglang:modelopt-pr30078" ;;
  *)                          DEFAULT_IMAGE="lmsysorg/sglang:v0.5.12-cu130" ;;
esac
IMAGE="${IMAGE:-$DEFAULT_IMAGE}"
PORT="${PORT:-8000}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
HF_CACHE_MODE="${HF_CACHE_MODE:-ro}"            # set 'rw' for a first-time model download
CACHE_VOL="${CACHE_VOL:-sglang-compile-cache}"  # flashinfer / torch.compile JIT cache
SHM_SIZE="${SHM_SIZE:-32g}"                      # SGLang wants generous /dev/shm

mkdir -p "${HF_CACHE}"

RUNMODE="-it"; [ -n "${DETACH:-}" ] && RUNMODE="-d"
RM_FLAG=(--rm); [ -n "${KEEP:-}" ] && RM_FLAG=()   # KEEP=1 keeps a crashed container so `docker logs` survives

CONTAINER_NAME="sglang-$(basename "${SERVE_SCRIPT}" .sh)"
existing_status="$(docker inspect -f '{{.State.Status}}' "${CONTAINER_NAME}" 2>/dev/null || true)"
case "${existing_status}" in
  exited|dead|created)
    echo "Removing stale container ${CONTAINER_NAME} (${existing_status})" >&2
    docker rm "${CONTAINER_NAME}" >/dev/null
    ;;
  running|restarting|paused)
    echo "Container ${CONTAINER_NAME} is already ${existing_status}; refusing to replace it." >&2
    exit 1
    ;;
esac

exec docker run "${RM_FLAG[@]}" ${RUNMODE} \
  --device nvidia.com/gpu=all \
  --name "${CONTAINER_NAME}" \
  --security-opt no-new-privileges \
  --cap-drop ALL \
  --pids-limit 4096 \
  --shm-size "${SHM_SIZE}" \
  -p "127.0.0.1:${PORT}:${PORT}" \
  --tmpfs /tmp:rw,nosuid,nodev,exec,size=16g \
  -e HOME=/cache -v "${CACHE_VOL}:/cache" \
  -e HF_HOME=/models -v "${HF_CACHE}:/models:${HF_CACHE_MODE}" \
  -e HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
  -e CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}" \
  -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  -e TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor -e TRITON_CACHE_DIR=/cache/triton \
  ${HF_TOKEN:+-e HF_TOKEN="${HF_TOKEN}"} \
  ${MODEL:+-e MODEL="${MODEL}"} \
  ${DRAFTER:+-e DRAFTER="${DRAFTER}"} \
  ${SERVED:+-e SERVED="${SERVED}"} \
  ${MAXLEN:+-e MAXLEN="${MAXLEN}"} \
  ${NUM_SPEC:+-e NUM_SPEC="${NUM_SPEC}"} \
  ${DRAFT_WINDOW:+-e DRAFT_WINDOW="${DRAFT_WINDOW}"} \
  ${MEM_FRACTION:+-e MEM_FRACTION="${MEM_FRACTION}"} \
  ${CUDA_GRAPH_MAX_BS:+-e CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS}"} \
  ${MAX_RUNNING_REQUESTS:+-e MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS}"} \
  ${CHUNKED_PREFILL_SIZE:+-e CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE}"} \
  ${MAX_PREFILL_TOKENS:+-e MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS}"} \
  ${ATTN_BACKEND:+-e ATTN_BACKEND="${ATTN_BACKEND}"} \
  ${MAMBA_STRATEGY:+-e MAMBA_STRATEGY="${MAMBA_STRATEGY}"} \
  ${RADIX:+-e RADIX="${RADIX}"} \
  ${PARSERS:+-e PARSERS="${PARSERS}"} \
  ${PORT:+-e PORT="${PORT}"} \
  ${SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN:+-e SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN="${SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN}"} \
  ${SGLANG_ENABLE_SPEC_V2:+-e SGLANG_ENABLE_SPEC_V2="${SGLANG_ENABLE_SPEC_V2}"} \
  ${SGLANG_ENABLE_DFLASH_SPEC_V2:+-e SGLANG_ENABLE_DFLASH_SPEC_V2="${SGLANG_ENABLE_DFLASH_SPEC_V2}"} \
  ${SGLANG_ENABLE_OVERLAP_PLAN_STREAM:+-e SGLANG_ENABLE_OVERLAP_PLAN_STREAM="${SGLANG_ENABLE_OVERLAP_PLAN_STREAM}"} \
  ${SGLANG_ENABLE_DEEP_GEMM:+-e SGLANG_ENABLE_DEEP_GEMM="${SGLANG_ENABLE_DEEP_GEMM}"} \
  -v "${PROJECT_DIR}:/workspace:ro" \
  -w /workspace \
  --entrypoint /bin/bash \
  "${IMAGE}" "/workspace/${SERVE_SCRIPT}"

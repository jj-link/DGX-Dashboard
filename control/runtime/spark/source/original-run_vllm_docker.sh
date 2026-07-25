#!/usr/bin/env bash
# Run any serve_*.sh launch script inside a hardened container.
#
#   ./run_vllm_docker.sh serve/vllm/serve_qwen36_a3b_fp8.sh
#   ./run_vllm_docker.sh serve/vllm/serve_qwen36_a3b_nvfp4.sh
#
# Hardening rationale (the point of doing this at all):
#   - cap-drop ALL + no-new-privileges: a compromised PyPI dep can't
#     escalate or use Linux capabilities it never needs for inference.
#   - No $HOME / secrets mounted: the container cannot read ~/.ssh,
#     ~/.hermes/config.yaml, your project source, etc. Blast radius is
#     just the model cache + a scratch volume.
#   - HF cache mounted READ-ONLY: malicious code can't poison your
#     cached weights for the next run. (First-ever download needs RW —
#     see HF_CACHE_MODE below.)
#   - Only port 8000 published; nothing else reachable.
#
# NOT covered here (deliberately, so the server still works):
#   - Network egress is still allowed. If exfiltration is a concern,
#     put this behind an egress allowlist / locked-down docker network.
set -euo pipefail

SERVE_SCRIPT="${1:?usage: run_vllm_docker.sh <serve_script.sh>}"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

[ -f "${PROJECT_DIR}/${SERVE_SCRIPT}" ] || { echo "no such script: ${SERVE_SCRIPT}" >&2; exit 1; }

# DFlash scripts need the PR-#40898 image; everything else uses the pinned
# one. Auto-selected from the script name so `./run_vllm_docker.sh
# serve_qwen36_a3b_fp8_dflash.sh` just works, like the fp8 invocation.
case "${SERVE_SCRIPT}" in
  *qwen36_35b_a3b_nvfp4_mtp3*)
    DEFAULT_IMAGE="a3b-fast:sm121-mtp-v2"
    DEFAULT_HF_CACHE="/home/jjlink/hf-cache"
    DEFAULT_CACHE_VOL="vllm-compile-cache-mtp2"
    DEFAULT_BIND_HOST="100.86.3.45"
    DEFAULT_IPC_MODE="host"
    DEFAULT_RESTART_POLICY="unless-stopped"
    DEFAULT_CONTAINER_NAME="qwen36-a3b-mtp3"
    MOE_CONFIG_FILE="${MOE_CONFIG_FILE-}"
    ;;
  *gemma4_31b_nvfp4*) DEFAULT_IMAGE="vllm/vllm-openai:gemma" ;; # Gemma4 + NVFP4 + DFlash requires vLLM 0.22.1rc1/PR support
  *dflash*) DEFAULT_IMAGE="qwen-vllm:dflash" ;;   # built from ./docker/Dockerfile.dflash
  *)        DEFAULT_IMAGE="qwen-vllm:pinned" ;;   # built from ./docker/Dockerfile
esac
IMAGE="${IMAGE:-$DEFAULT_IMAGE}"
PORT="${PORT:-8000}"
HF_CACHE="${HF_CACHE:-${DEFAULT_HF_CACHE:-$HOME/.cache/huggingface}}"
HF_CACHE_MODE="${HF_CACHE_MODE:-ro}"        # set to 'rw' for first-time model download
CACHE_VOL="${CACHE_VOL:-${DEFAULT_CACHE_VOL:-vllm-compile-cache}}" # torch.compile / flashinfer JIT cache
BIND_HOST="${BIND_HOST:-${DEFAULT_BIND_HOST:-127.0.0.1}}"
IPC_MODE="${IPC_MODE:-${DEFAULT_IPC_MODE:-}}"
RESTART_POLICY="${RESTART_POLICY:-${DEFAULT_RESTART_POLICY:-}}"
CONTAINER_NAME="${CONTAINER_NAME:-${DEFAULT_CONTAINER_NAME:-vllm-$(basename "${SERVE_SCRIPT}" .sh)}}"

mkdir -p "${HF_CACHE}"

# Optional: overlay a tuned fused-MoE config the image doesn't ship for
# this exact device_name (e.g. the RTX PRO 6000 *Workstation* Edition,
# which can't auto-pick the *Server* Edition's tuned tile config).
# MOE_CONFIG_FILE = filename under ./moe_configs/ ; single-file bind so
# the rest of the shipped configs dir stays intact.
# Defaults to the FP8 Workstation-Edition tuned tile config (vLLM ships
# it only under the Server-Edition device name; same SM120 silicon).
# Harmless for non-matching models (vLLM keys configs by E/N/dtype, so an
# fp8_w8a8 file is ignored by an NVFP4 run). Set MOE_CONFIG_FILE= to skip.
MOE_CFG_DIR="/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/configs"
MOE_CONFIG_FILE="${MOE_CONFIG_FILE-E=256,N=512,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Workstation_Edition,dtype=fp8_w8a8,block_shape=[128,128].json}"
MOE_CFG_MOUNT=()
if [ -n "${MOE_CONFIG_FILE:-}" ]; then
  [ -f "${PROJECT_DIR}/moe_configs/${MOE_CONFIG_FILE}" ] || { echo "missing moe_configs/${MOE_CONFIG_FILE}" >&2; exit 1; }
  MOE_CFG_MOUNT=(-v "${PROJECT_DIR}/moe_configs/${MOE_CONFIG_FILE}:${MOE_CFG_DIR}/${MOE_CONFIG_FILE}:ro")
fi

# Option B: bind-mount the patched gpu_model_runner.py (DFLASH_MAX_CONTEXT -> skip
# drafter past that context = plain decode) over the image's copy, only when
# DFLASH_MAX_CONTEXT is set. No rebuild; reversible (unset the env to disable).
PATCH_MOUNT=()
if [ -n "${DFLASH_MAX_CONTEXT:-}" ] && [ -f "${PROJECT_DIR}/patches/gpu_model_runner.py" ]; then
  PATCH_MOUNT=(-v "${PROJECT_DIR}/patches/gpu_model_runner.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py:ro")
fi

RUNMODE=(-it)
RM_ARG=(--rm)
RESTART_ARG=()
IPC_ARG=()
[ -n "${IPC_MODE}" ] && IPC_ARG=(--ipc "${IPC_MODE}")
[ -n "${DETACH:-}" ] && RUNMODE=(-d)
[ -n "${KEEP:-}" ] && RM_ARG=()
if [ -n "${RESTART_POLICY}" ]; then
  RUNMODE=(-d)
  RM_ARG=()
  RESTART_ARG=(--restart "${RESTART_POLICY}")
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
fi
exec docker run "${RM_ARG[@]}" "${RUNMODE[@]}" \
  "${MOE_CFG_MOUNT[@]}" \
  "${PATCH_MOUNT[@]}" \
  --gpus all \
  "${IPC_ARG[@]}" \
  --name "${CONTAINER_NAME}" \
  "${RESTART_ARG[@]}" \
  --security-opt no-new-privileges \
  --cap-drop ALL \
  --pids-limit 4096 \
  -p "${BIND_HOST}:${PORT}:${PORT}" \
  --tmpfs /tmp:rw,nosuid,nodev,size=4g \
  -e HOME=/cache -v "${CACHE_VOL}:/cache" \
  -e HF_HOME=/models -v "${HF_CACHE}:/models:${HF_CACHE_MODE}" \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e PORT="${PORT}" \
  -e CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}" \
  -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  ${MODEL:+-e MODEL="${MODEL}"} \
  ${SERVED:+-e SERVED="${SERVED}"} \
  ${TOKENIZER:+-e TOKENIZER="${TOKENIZER}"} \
  ${MOE_BACKEND:+-e MOE_BACKEND="${MOE_BACKEND}"} \
  ${EAGER:+-e EAGER="${EAGER}"} \
  ${VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE:+-e VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE="${VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE}"} \
  ${GPU_MEM_UTIL:+-e GPU_MEM_UTIL="${GPU_MEM_UTIL}"} \
  ${VLLM_MARLIN_USE_ATOMIC_ADD:+-e VLLM_MARLIN_USE_ATOMIC_ADD="${VLLM_MARLIN_USE_ATOMIC_ADD}"} \
  ${VLLM_MARLIN_INPUT_DTYPE:+-e VLLM_MARLIN_INPUT_DTYPE="${VLLM_MARLIN_INPUT_DTYPE}"} \
  ${VLLM_NVFP4_GEMM_BACKEND:+-e VLLM_NVFP4_GEMM_BACKEND="${VLLM_NVFP4_GEMM_BACKEND}"} \
  ${KV_CACHE_DTYPE:+-e KV_CACHE_DTYPE="${KV_CACHE_DTYPE}"} \
  ${PREFIX_CACHING:+-e PREFIX_CACHING="${PREFIX_CACHING}"} \
  ${CHUNKED_PREFILL:+-e CHUNKED_PREFILL="${CHUNKED_PREFILL}"} \
  ${VLLM_ATTENTION_BACKEND:+-e VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND}"} \
  ${FI_AUTOTUNE_ARG:+-e FI_AUTOTUNE_ARG="${FI_AUTOTUNE_ARG}"} \
  ${MAMBA_DTYPE:+-e MAMBA_DTYPE="${MAMBA_DTYPE}"} \
  ${INDUCTOR_MAX_AUTOTUNE:+-e INDUCTOR_MAX_AUTOTUNE="${INDUCTOR_MAX_AUTOTUNE}"} \
  ${PARSERS:+-e PARSERS="${PARSERS}"} \
  ${DRAFTER:+-e DRAFTER="${DRAFTER}"} \
  ${NUM_SPEC:+-e NUM_SPEC="${NUM_SPEC}"} \
  ${SPEC_MAXLEN:+-e SPEC_MAXLEN="${SPEC_MAXLEN}"} \
  ${DFLASH_MAX_CONTEXT:+-e DFLASH_MAX_CONTEXT="${DFLASH_MAX_CONTEXT}"} \
  ${MTP_METHOD:+-e MTP_METHOD="${MTP_METHOD}"} \
  ${MAX_NUM_BATCHED:+-e MAX_NUM_BATCHED="${MAX_NUM_BATCHED}"} \
  ${MAX_NUM_SEQS:+-e MAX_NUM_SEQS="${MAX_NUM_SEQS}"} \
  ${ATTENTION_BACKEND:+-e ATTENTION_BACKEND="${ATTENTION_BACKEND}"} \
  ${SPECULATIVE_CONFIG:+-e SPECULATIVE_CONFIG="${SPECULATIVE_CONFIG}"} \
  ${DFLASH_DRAFT_WINDOW:+-e DFLASH_DRAFT_WINDOW="${DFLASH_DRAFT_WINDOW}"} \
  ${MAXLEN:+-e MAXLEN="${MAXLEN}"} \
  ${MODELS_HOST:+-v "${MODELS_HOST}:${MODELS_HOST}:ro"} \
  -v "${PROJECT_DIR}:/workspace:ro" \
  -w /workspace \
  --entrypoint /bin/bash \
  "${IMAGE}" "/workspace/${SERVE_SCRIPT}"

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

# ThinkingCap defaults to isolated, checkpoint-specific runtime images. Keep
# these cases before the generic 27B and DFlash matches.
IS_THINKINGCAP_FP8=0
case "${SERVE_SCRIPT}" in
  *bottlecapai_thinkingcap_qwen36_27b_fp8*)
    CACHE_VOL="${CACHE_VOL:-vllm-thinkingcap-fp8-compile-cache}"
    DEFAULT_IMAGE="sha256:82083ab4f7b3815af617aa451c2dfe55cc25b8c990a1839afec5c85c2d07448d"
    VLLM_BATCH_INVARIANT="${VLLM_BATCH_INVARIANT:-1}"
    IS_THINKINGCAP_FP8=1
    ;;
  *morosystems_thinkingcap_qwen36_27b_nvfp4*)
    CACHE_VOL="${CACHE_VOL:-vllm-mrv2-compile-cache-standard}"
    if [ "${RECOVERY_EAGER:-0}" = "1" ]; then
      DEFAULT_IMAGE="sha256:dcb48c55f342e50d29def6125ffeb78e2bba2eba57d77262d4d88b7a922d8245"
      EAGER="${EAGER:-1}"
      SPEC="${SPEC:-dflash}"
    else
      DEFAULT_IMAGE="sha256:36985ad979cb947dab7c1fc3f0ff29ffbcbabab15102191c082d4f0bebf2085e"
      VLLM_BATCH_INVARIANT="${VLLM_BATCH_INVARIANT:-1}"
    fi
    ;;
  *nvidia_qwen36_27b_nvfp4_mtp3_bf16kv*|*nvidia_qwen36_27b_nvfp4_dflash11_bf16kv*|*nvidia_qwen36_27b_nvfp4_dflsh11*)
    DEFAULT_IMAGE="sha256:61d7a2ae1f7e087f7212ad3931a55456607da9ccbc86e16e304151eae44a3191"
    ;;
  *nvidia_qwen36_27b_nvfp4*)         DEFAULT_IMAGE="sha256:251eba5cc7c12fed0b75da22a9240e582b1c9e39f6fbc064f86781b963bd814f" ;;
  *unsloth_qwen36*)                  DEFAULT_IMAGE="sha256:251eba5cc7c12fed0b75da22a9240e582b1c9e39f6fbc064f86781b963bd814f" ;;
  *gemma4_31b_nvfp4*)                DEFAULT_IMAGE="vllm/vllm-openai:gemma" ;;
  *aeon*)                            DEFAULT_IMAGE="aeon-vllm-local:x86" ;;
  *dflash*)                          DEFAULT_IMAGE="qwen-vllm:dflash" ;;
  *)                                 DEFAULT_IMAGE="qwen-vllm:pinned" ;;
esac
# DFlash11 wins below the measured long-context crossover but loses to plain
# target decoding above it. An explicitly empty value disables the cutoff.
case "${SERVE_SCRIPT}" in
  *nvidia_qwen36_27b_nvfp4_dflash11_bf16kv*)
    DFLASH_MAX_CONTEXT="${DFLASH_MAX_CONTEXT-131072}"
    ;;
esac
IMAGE="${IMAGE:-$DEFAULT_IMAGE}"
PORT="${PORT:-8000}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
HF_CACHE_MODE="${HF_CACHE_MODE:-ro}"        # set to 'rw' for first-time model download
CACHE_VOL="${CACHE_VOL:-vllm-compile-cache}" # torch.compile / flashinfer JIT cache
CONTAINER_NAME="${CONTAINER_NAME:-vllm-$(basename "${SERVE_SCRIPT}" .sh)}"

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

# Bind-mount scheduler overlays when speculative decoding is delayed or DFlash
# uses a context cutoff. The DFlash cutoff additionally needs V2 runner and
# graph-manager overlays.
PATCH_MOUNT=()
if [ -n "${DFLASH_MAX_CONTEXT:-}" ] || [ -n "${VLLM_SPECULATIVE_MIN_OUTPUT_TOKENS:-}" ]; then
  [ -f "${PROJECT_DIR}/patches/scheduler.py" ] || { echo "missing patches/scheduler.py" >&2; exit 1; }
  [ -f "${PROJECT_DIR}/patches/async_scheduler.py" ] || { echo "missing patches/async_scheduler.py" >&2; exit 1; }
  PATCH_MOUNT=(
    -v "${PROJECT_DIR}/patches/scheduler.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py:ro"
    -v "${PROJECT_DIR}/patches/async_scheduler.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/async_scheduler.py:ro"
  )
fi
if [ -n "${DFLASH_MAX_CONTEXT:-}" ]; then
  [ -f "${PROJECT_DIR}/patches/gpu_model_runner.py" ] || { echo "missing patches/gpu_model_runner.py" >&2; exit 1; }
  [ -f "${PROJECT_DIR}/patches/cudagraph_utils.py" ] || { echo "missing patches/cudagraph_utils.py" >&2; exit 1; }
  PATCH_MOUNT+=(
    -v "${PROJECT_DIR}/patches/gpu_model_runner.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/model_runner.py:ro"
    -v "${PROJECT_DIR}/patches/cudagraph_utils.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/cudagraph_utils.py:ro"
  )
fi
if [ "${VLLM_USE_V2_MODEL_RUNNER:-0}" = "1" ]; then
  [ -f "${PROJECT_DIR}/patches/vllm-config-v2-align.py" ] || { echo "missing patches/vllm-config-v2-align.py" >&2; exit 1; }
  PATCH_MOUNT+=(
    -v "${PROJECT_DIR}/patches/vllm-config-v2-align.py:/usr/local/lib/python3.12/dist-packages/vllm/config/vllm.py:ro"
  )
fi
if [ "${VLLM_BATCH_INVARIANT:-0}" = "1" ]; then
  if [ -z "${DFLASH_MAX_CONTEXT:-}" ] && [ -z "${VLLM_SPECULATIVE_MIN_OUTPUT_TOKENS:-}" ]; then
    SCHEDULER_PATCH="${PROJECT_DIR}/patches/vllm-0.24-align-upstream/vllm/v1/core/sched/scheduler.py"
    [ -f "$SCHEDULER_PATCH" ] || { echo "missing batch-invariant scheduler patch" >&2; exit 1; }
    PATCH_MOUNT+=(
      -v "${SCHEDULER_PATCH}:/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py:ro"
    )
  fi
  GDN_BACKEND_PATCH="${PROJECT_DIR}/patches/vllm-0.24-align-upstream/vllm/v1/attention/backends/gdn_attn.py"
  FORWARD_CONTEXT_PATCH="${PROJECT_DIR}/patches/vllm-0.24-align-upstream/vllm/forward_context.py"
  NVFP4_CUTLASS_PATCH="${PROJECT_DIR}/patches/vllm-0.24-align-upstream/vllm/model_executor/kernels/linear/nvfp4/cutlass.py"
  PLATFORM_INTERFACE_PATCH="${PROJECT_DIR}/patches/vllm-0.24-align-upstream/vllm/platforms/interface.py"
  KV_CACHE_UTILS_PATCH="${PROJECT_DIR}/patches/vllm-0.24-align-upstream/vllm/v1/core/kv_cache_utils.py"
  if [ "$IS_THINKINGCAP_FP8" != "1" ]; then
    [ -f "$NVFP4_CUTLASS_PATCH" ] || { echo "missing batch-invariant NVFP4 CUTLASS patch" >&2; exit 1; }
  fi
  [ -f "$FORWARD_CONTEXT_PATCH" ] || { echo "missing batch-invariant forward context patch" >&2; exit 1; }
  [ -f "$GDN_BACKEND_PATCH" ] || { echo "missing batch-invariant GDN backend patch" >&2; exit 1; }
  [ -f "$PLATFORM_INTERFACE_PATCH" ] || { echo "missing batch-invariant platform patch" >&2; exit 1; }
  [ -f "$KV_CACHE_UTILS_PATCH" ] || { echo "missing batch-invariant KV cache patch" >&2; exit 1; }
  PATCH_MOUNT+=(
    -v "${FORWARD_CONTEXT_PATCH}:/usr/local/lib/python3.12/dist-packages/vllm/forward_context.py:ro"
    -v "${GDN_BACKEND_PATCH}:/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/gdn_attn.py:ro"
    -v "${PLATFORM_INTERFACE_PATCH}:/usr/local/lib/python3.12/dist-packages/vllm/platforms/interface.py:ro"
    -v "${KV_CACHE_UTILS_PATCH}:/usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_utils.py:ro"
  )
  if [ "$IS_THINKINGCAP_FP8" != "1" ]; then
    PATCH_MOUNT+=(
      -v "${NVFP4_CUTLASS_PATCH}:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/kernels/linear/nvfp4/cutlass.py:ro"
    )
  fi
fi
if [ "${DFLASH_V1_MIXED:-0}" = "1" ]; then
  [ -f "${PROJECT_DIR}/patches/qwen3-dflash-v1-mixed.py" ] || { echo "missing patches/qwen3-dflash-v1-mixed.py" >&2; exit 1; }
  [ -f "${PROJECT_DIR}/patches/vllm-0.24-align-upstream/vllm/v1/spec_decode/dflash.py" ] || { echo "missing patched DFlash proposer" >&2; exit 1; }
  PATCH_MOUNT+=(
    -v "${PROJECT_DIR}/patches/qwen3-dflash-v1-mixed.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_dflash.py:ro"
    -v "${PROJECT_DIR}/patches/vllm-0.24-align-upstream/vllm/v1/spec_decode/dflash.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/spec_decode/dflash.py:ro"
  )
fi


RUNMODE="-it"; [ -n "${DETACH:-}" ] && RUNMODE="-d"
RM_ARG="--rm"
[ -n "${KEEP:-}" ] && RM_ARG=""
exec docker run ${RM_ARG} ${RUNMODE} \
  "${MOE_CFG_MOUNT[@]}" \
  "${PATCH_MOUNT[@]}" \
  --device nvidia.com/gpu=all \
  --name "${CONTAINER_NAME}" \
  --security-opt no-new-privileges \
  --cap-drop ALL \
  --pids-limit 4096 \
  -p "127.0.0.1:${PORT}:${PORT}" \
  --tmpfs /tmp:rw,nosuid,nodev,exec,size=4g \
  -e HOME=/cache -v "${CACHE_VOL}:/cache" \
  -e HF_HOME=/models -v "${HF_CACHE}:/models:${HF_CACHE_MODE}" \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e PORT="${PORT}" \
  -e CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}" \
  -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  ${MODEL:+-e MODEL="${MODEL}"} \
  ${SERVED:+-e SERVED="${SERVED}"} \
  ${SERVED_ALIASES:+-e SERVED_ALIASES="${SERVED_ALIASES}"} \
  ${TOKENIZER:+-e TOKENIZER="${TOKENIZER}"} \
  ${MOE_BACKEND:+-e MOE_BACKEND="${MOE_BACKEND}"} \
  ${EAGER:+-e EAGER="${EAGER}"} \
  ${VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE:+-e VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE="${VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE}"} \
  ${GPU_MEM_UTIL:+-e GPU_MEM_UTIL="${GPU_MEM_UTIL}"} \
  ${VLLM_MARLIN_USE_ATOMIC_ADD:+-e VLLM_MARLIN_USE_ATOMIC_ADD="${VLLM_MARLIN_USE_ATOMIC_ADD}"} \
  ${VLLM_MARLIN_INPUT_DTYPE:+-e VLLM_MARLIN_INPUT_DTYPE="${VLLM_MARLIN_INPUT_DTYPE}"} \
  ${VLLM_NVFP4_GEMM_BACKEND:+-e VLLM_NVFP4_GEMM_BACKEND="${VLLM_NVFP4_GEMM_BACKEND}"} \
  ${VLLM_USE_V2_MODEL_RUNNER:+-e VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER}"} \
  ${VLLM_BATCH_INVARIANT:+-e VLLM_BATCH_INVARIANT="${VLLM_BATCH_INVARIANT}"} \
  ${VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE:+-e VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE="${VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE}"} \
  ${VLLM_TRACE_SPEC_ACCEPTANCE:+-e VLLM_TRACE_SPEC_ACCEPTANCE="${VLLM_TRACE_SPEC_ACCEPTANCE}"} \
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
  ${SPEC:+-e SPEC="${SPEC}"} \
  ${SPEC_MAXLEN:+-e SPEC_MAXLEN="${SPEC_MAXLEN}"} \
  ${DFLASH_MAX_CONTEXT:+-e DFLASH_MAX_CONTEXT="${DFLASH_MAX_CONTEXT}"} \
  ${VLLM_SPECULATIVE_MIN_OUTPUT_TOKENS:+-e VLLM_SPECULATIVE_MIN_OUTPUT_TOKENS="${VLLM_SPECULATIVE_MIN_OUTPUT_TOKENS}"} \
  ${VLLM_SPECULATIVE_LONG_CONTEXT_THRESHOLD:+-e VLLM_SPECULATIVE_LONG_CONTEXT_THRESHOLD="${VLLM_SPECULATIVE_LONG_CONTEXT_THRESHOLD}"} \
  ${VLLM_SPECULATIVE_LONG_CONTEXT_TOKENS:+-e VLLM_SPECULATIVE_LONG_CONTEXT_TOKENS="${VLLM_SPECULATIVE_LONG_CONTEXT_TOKENS}"} \
  ${DISABLE_PADDED_DRAFTER_BATCH:+-e DISABLE_PADDED_DRAFTER_BATCH="${DISABLE_PADDED_DRAFTER_BATCH}"} \
  ${TORCHINDUCTOR_CACHE_DIR:+-e TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR}"} \
  ${MTP_METHOD:+-e MTP_METHOD="${MTP_METHOD}"} \
  ${MAX_NUM_BATCHED:+-e MAX_NUM_BATCHED="${MAX_NUM_BATCHED}"} \
  ${MAX_NUM_SEQS:+-e MAX_NUM_SEQS="${MAX_NUM_SEQS}"} \
  ${BLOCK_SIZE:+-e BLOCK_SIZE="${BLOCK_SIZE}"} \
  ${ATTENTION_BACKEND:+-e ATTENTION_BACKEND="${ATTENTION_BACKEND}"} \
  ${LINEAR_BACKEND:+-e LINEAR_BACKEND="${LINEAR_BACKEND}"} \
  ${COMPILATION_CONFIG:+-e COMPILATION_CONFIG="${COMPILATION_CONFIG}"} \
  ${SPECULATIVE_CONFIG:+-e SPECULATIVE_CONFIG="${SPECULATIVE_CONFIG}"} \
  ${DFLASH_DRAFT_WINDOW:+-e DFLASH_DRAFT_WINDOW="${DFLASH_DRAFT_WINDOW}"} \
  ${DISABLE_HYBRID_KV_CACHE_MANAGER:+-e DISABLE_HYBRID_KV_CACHE_MANAGER="${DISABLE_HYBRID_KV_CACHE_MANAGER}"} \
  ${MAXLEN:+-e MAXLEN="${MAXLEN}"} \
  ${MODELS_HOST:+-v "${MODELS_HOST}:${MODELS_HOST}:ro"} \
  -v "${PROJECT_DIR}:/workspace:ro" \
  -w /workspace \
  --entrypoint /bin/bash \
  "${IMAGE}" "-e" "/workspace/${SERVE_SCRIPT}"

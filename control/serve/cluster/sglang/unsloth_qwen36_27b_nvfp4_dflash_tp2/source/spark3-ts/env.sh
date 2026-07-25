#!/usr/bin/env bash
# Qwen3.6-27B Unsloth NVFP4 + z-lab DFlash on SGLang, TP=2 across Spark2/Spark3.
# Spark2 is rank0/head/API, Spark3 is rank1/worker.

export RECIPE_DIR="/home/jjlink/inference/qwen36-27b-unsloth-nvfp4-sglang-dflash"
export IMAGE="lmsysorg/sglang:v0.5.12-cu130"
export IMAGE_EXPECTED_SGLANG="0.5.12"
export ENABLE_DFLASH="${ENABLE_DFLASH:-1}"
if [[ "$ENABLE_DFLASH" == "1" ]]; then
  export CONTAINER="sglang_qwen36_unsloth_nvfp4_dflash_tp2"
else
  export CONTAINER="sglang_qwen36_unsloth_nvfp4_tp2"
fi

export HEAD_HOST="spark2-ts"
export WORKER_HOST="spark3-ts"
export HEAD_IP="10.0.0.1"
export WORKER_IP="10.0.0.2"
export DIST_INIT_ADDR="10.0.0.1:25001"
export API_PORT="8888"
export SGLANG_PORT="8888"

# Complete HF cache paths on Spark2/Spark3. Do not use incomplete blob paths.
export HOST_HF_ROOT="/home/jjlink/models/hub"
export TARGET_MODEL_PATH="/models/hub/models--unsloth--Qwen3.6-27B-NVFP4/snapshots/890bdef7a42feba6d83b6e17a03315c694112f2a"
export DRAFT_MODEL_PATH="/models/hub/models--z-lab--Qwen3.6-27B-DFlash/snapshots/0919688658996800f86b895034249700e9481106"
if [[ "$ENABLE_DFLASH" == "1" ]]; then
  export SERVED_MODEL_NAME="qwen36-27b-unsloth-nvfp4-dflash-tp2"
else
  export SERVED_MODEL_NAME="qwen36-27b-unsloth-nvfp4-tp2"
fi

export NNODES="2"
export TP_SIZE="2"
export CONTEXT_LENGTH="262144"
export MEM_FRACTION_STATIC="0.85"
export MAX_RUNNING_REQUESTS="8"
export CHUNKED_PREFILL_SIZE="2048"
export DISABLE_CUDA_GRAPH="${DISABLE_CUDA_GRAPH:-1}"
export CUDA_GRAPH_MAX_BS="16"

# Preserved from Spark1 known-good DFlash compose.
export SPECULATIVE_ALGORITHM="DFLASH"
export SPECULATIVE_NUM_DRAFT_TOKENS="20"
export SPECULATIVE_DRAFT_WINDOW_SIZE="4096"
export MAMBA_SCHEDULER_STRATEGY="extra_buffer"

# GB10 RoCE/RDMA/NCCL discipline.
export NCCL_NET="IB"
export NCCL_IB_DISABLE="0"
export NCCL_IB_HCA="rocep1s0f1"
export NCCL_SOCKET_IFNAME="enp1s0f1np1"
export GLOO_SOCKET_IFNAME="enp1s0f1np1"
export TP_SOCKET_IFNAME="enp1s0f1np1"
export NCCL_IB_GID_INDEX="3"
export NCCL_CROSS_NIC="1"
export NCCL_DEBUG="INFO"
export NCCL_IGNORE_CPU_AFFINITY="1"
export HF_HOME="/models"
export HF_HUB_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"
export HF_HUB_DISABLE_XET="1"
export SGLANG_ENABLE_SPEC_V2="1"
export SGLANG_ENABLE_JIT_DEEPGEMM="0"

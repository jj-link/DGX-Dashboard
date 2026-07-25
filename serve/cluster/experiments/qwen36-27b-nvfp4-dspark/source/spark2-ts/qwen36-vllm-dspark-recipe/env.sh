#!/usr/bin/env bash
# Qwen3.6 NVFP4 vLLM TP=2 on Spark2/Spark3.
# DSpark-style GB10/RDMA/RoCE/NCCL environment, no MiMo image/patches.

export RECIPE_DIR="/home/jjlink/inference/qwen36-vllm-dspark-recipe"
export IMAGE="qwen36-vllm-dspark:0.24.0-ray"
export CONTAINER="qwen36_vllm_dspark"
export HEAD_HOST="spark2-ts"
export WORKER_HOST="spark3-ts"
export HEAD_IP="10.0.0.1"
export WORKER_IP="10.0.0.2"
export API_PORT="8888"

# Host model cache: inside container this becomes /models/hub/models--nvidia--...
export HOST_MODELS_ROOT="/home/jjlink/models/hub"
export MODEL_PATH="/models/hub/models--nvidia--Qwen3.6-27B-NVFP4"
export SERVED_MODEL_NAME="qwen36-27b-nvfp4"

# DSpark-style RoCE/NCCL settings for jjlink Spark fleet.
export NCCL_NET="IB"
export NCCL_IB_DISABLE="0"
export NCCL_IB_HCA="rocep1s0f1"
export NCCL_SOCKET_IFNAME="enp1s0f1np1"
export GLOO_SOCKET_IFNAME="enp1s0f1np1"
export TP_SOCKET_IFNAME="enp1s0f1np1"
export NCCL_IB_GID_INDEX="3"
export NCCL_CROSS_NIC="1"
export NCCL_CUMEM_ENABLE="0"
export NCCL_NVLS_ENABLE="0"
export NCCL_NET_GDR_LEVEL="LOC"
export NCCL_SOCKET_FAMILY="AF_INET"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export RAY_memory_monitor_refresh_ms="0"
export RAY_TMPDIR="/dev/shm/ray"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN="1"

# Targeted Ray stability knobs from the DSpark/MiMo stable pattern. These are not
# MiMo model patches; they affect vLLM/Ray distributed execution only.
export VLLM_USE_RAY_V2_EXECUTOR_BACKEND="0"
export VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM="0"

# Offline/local-cache discipline.
export HF_HOME="/models"
export HF_HUB_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"

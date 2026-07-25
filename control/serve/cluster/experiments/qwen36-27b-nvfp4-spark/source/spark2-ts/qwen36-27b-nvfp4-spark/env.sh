#!/usr/bin/env bash
# Qwen3.6-27B-NVFP4 vLLM rank-based TP=2 on Spark2/Spark3.
# DSpark-style Docker/RDMA/RoCE/NCCL discipline; no Ray, no MiMo, no SGLang.

export RECIPE_DIR="/home/jjlink/inference/qwen36-27b-nvfp4-spark"
export IMAGE="vllm/vllm-openai:v0.23.0"
export IMAGE_EXPECTED_VLLM="0.23.0"
export IMAGE_EXPECTED_COMMIT="91df0fad4dc98a67c7659d9dbd915245d5c43d96"
export CONTAINER="qwen36_nvfp4_spark"

export HEAD_HOST="spark2-ts"
export WORKER_HOST="spark3-ts"
export HEAD_IP="10.0.0.1"
export WORKER_IP="10.0.0.2"
export MASTER_ADDR="10.0.0.1"
export MASTER_PORT="25000"
export API_PORT="8888"

# Complete cache is the double-hub path on both Spark nodes.
# Host:      /home/jjlink/models/hub/hub/models--nvidia--Qwen3.6-27B-NVFP4
# Container: /models/hub/models--nvidia--Qwen3.6-27B-NVFP4
export HOST_MODEL_ROOT="/home/jjlink/models/hub/hub"
export MODEL_PATH="/models/hub/models--nvidia--Qwen3.6-27B-NVFP4"
export SERVED_MODEL_NAME="qwen36-27b-nvfp4"

export NNODES="2"
export TENSOR_PARALLEL_SIZE="2"
export MAX_MODEL_LEN="262144"
export GPU_MEMORY_UTILIZATION="0.90"
export MAX_NUM_BATCHED_TOKENS="2048"
export MAX_NUM_SEQS="8"
export BLOCK_SIZE="64"

# DSpark/RoCE/NCCL settings for the Spark2/Spark3 GB10 fabric.
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
export NCCL_IGNORE_CPU_AFFINITY="1"
export NCCL_DEBUG="INFO"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN="1"
# v0.23.0 is still V1-only for this model, but disabling V1 multiprocessing
# keeps the EngineCore in-process. This is the next diagnostic against the
# follower-node `collective_rpc should not be called on follower node` crash.
export VLLM_ENABLE_V1_MULTIPROCESSING="0"
export HF_HOME="/models"
export HF_HUB_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"
export HF_HUB_DISABLE_XET="1"

#!/usr/bin/env bash
set -euo pipefail

export NCCL_NET=IB
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=enp1s0f1np1
export GLOO_SOCKET_IFNAME=enp1s0f1np1
export NCCL_IB_HCA=rocep1s0f1
export NCCL_IB_GID_INDEX=3
export NCCL_CROSS_NIC=1
export NCCL_CUMEM_ENABLE=0
export NCCL_NVLS_ENABLE=0
export NCCL_NET_GDR_LEVEL=LOC
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RAY_memory_monitor_refresh_ms=0
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_USE_RAY_V2_EXECUTOR_BACKEND=0
export RAY_TMPDIR=/dev/shm/ray
export VLLM_HOST_IP=10.0.0.1

vllm serve "/root/.cache/huggingface/hub/hub/models--nvidia--Qwen3.6-27B-NVFP4" \
  --served-model-name qwen36-27b-nvfp4 \
  --trust-remote-code \
  --dtype auto \
  --tensor-parallel-size 2 \
  --pipeline-parallel-size 1 \
  --distributed-executor-backend ray \
  --load-format safetensors \
  --kv-cache-dtype auto \
  --gpu-memory-utilization 0.90 \
  --max-model-len 262144 \
  --max-num-batched-tokens 2048 \
  --max-num-seqs 8 \
  --block-size 64 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --no-async-scheduling \
  --host 0.0.0.0 \
  --port 8888
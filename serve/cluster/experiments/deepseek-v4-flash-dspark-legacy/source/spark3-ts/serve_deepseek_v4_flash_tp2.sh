#!/usr/bin/env bash
# Multi-node TP=2 serve for deepseek-ai/DeepSeek-V4-Flash-DSpark
# spark2 (10.0.0.1) = rank 0, spark3 (10.0.0.2) = rank 1
# Uses 200G QSFP link (enp1s0f1np1) for NCCL
set -u

MODEL="/home/jjlink/models/hub/models--deepseek-ai--DeepSeek-V4-Flash-DSpark"
TP=2

# NCCL over 200G Ethernet
export NCCL_SOCKET_IFNAME=enp1s0f1np1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=info

# Multi-node config
export MASTER_ADDR=10.0.0.1
export MASTER_PORT=29500
export WORLD_SIZE=2
export RANK="${RANK:-0}"

PORT="${PORT:-8000}"

exec vllm serve "$MODEL" \
  --tensor-parallel-size $TP \
  --distributed-executor-backend mp \
  --host 0.0.0.0 \
  --port "$PORT" \
  --trust-remote-code \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.92

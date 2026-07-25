#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh

NODE_RANK="${1:?usage: start-node.sh NODE_RANK VLLM_HOST_IP}"
VLLM_HOST_IP_VALUE="${2:?usage: start-node.sh NODE_RANK VLLM_HOST_IP}"

if [[ "$NODE_RANK" != "0" && "$NODE_RANK" != "1" ]]; then
  echo "NODE_RANK must be 0 or 1" >&2
  exit 2
fi

HEADLESS_FLAG=""
if [[ "$NODE_RANK" == "1" ]]; then
  # DSpark-style rank launch starts the non-head node as a headless worker.
  # Without --headless, vLLM v0.23/v0.24 starts an API/EngineCore on the follower
  # and crashes with: collective_rpc should not be called on follower node.
  HEADLESS_FLAG="--headless"
fi

docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

exec docker run -d \
  --name "$CONTAINER" \
  --entrypoint /bin/bash \
  --privileged \
  --gpus all \
  --network host \
  --ipc host \
  --shm-size 32g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --device=/dev/infiniband:/dev/infiniband \
  -e VLLM_HOST_IP="$VLLM_HOST_IP_VALUE" \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN="$VLLM_ALLOW_LONG_MAX_MODEL_LEN" \
  -e VLLM_ENABLE_V1_MULTIPROCESSING="$VLLM_ENABLE_V1_MULTIPROCESSING" \
  -e HF_HOME="$HF_HOME" \
  -e HF_HUB_OFFLINE="$HF_HUB_OFFLINE" \
  -e TRANSFORMERS_OFFLINE="$TRANSFORMERS_OFFLINE" \
  -e HF_HUB_DISABLE_XET="$HF_HUB_DISABLE_XET" \
  -e NCCL_NET="$NCCL_NET" \
  -e NCCL_IB_DISABLE="$NCCL_IB_DISABLE" \
  -e NCCL_IB_HCA="$NCCL_IB_HCA" \
  -e NCCL_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME" \
  -e GLOO_SOCKET_IFNAME="$GLOO_SOCKET_IFNAME" \
  -e TP_SOCKET_IFNAME="$TP_SOCKET_IFNAME" \
  -e NCCL_IB_GID_INDEX="$NCCL_IB_GID_INDEX" \
  -e NCCL_CROSS_NIC="$NCCL_CROSS_NIC" \
  -e NCCL_CUMEM_ENABLE="$NCCL_CUMEM_ENABLE" \
  -e NCCL_NVLS_ENABLE="$NCCL_NVLS_ENABLE" \
  -e NCCL_NET_GDR_LEVEL="$NCCL_NET_GDR_LEVEL" \
  -e NCCL_SOCKET_FAMILY="$NCCL_SOCKET_FAMILY" \
  -e NCCL_IGNORE_CPU_AFFINITY="$NCCL_IGNORE_CPU_AFFINITY" \
  -e NCCL_DEBUG="$NCCL_DEBUG" \
  -e PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" \
  -v "$HOST_MODEL_ROOT:/models/hub:ro" \
  -v "$RECIPE_DIR:/workspace/recipe:ro" \
  "$IMAGE" -lc "exec vllm serve '$MODEL_PATH' \
    --served-model-name '$SERVED_MODEL_NAME' \
    --host 0.0.0.0 \
    --port '$API_PORT' \
    --trust-remote-code \
    --dtype auto \
    --tensor-parallel-size '$TENSOR_PARALLEL_SIZE' \
    --pipeline-parallel-size 1 \
    --distributed-executor-backend mp \
    --nnodes '$NNODES' \
    --node-rank '$NODE_RANK' \
    --master-addr '$MASTER_ADDR' \
    --master-port '$MASTER_PORT' \
    --load-format safetensors \
    --kv-cache-dtype auto \
    --gpu-memory-utilization '$GPU_MEMORY_UTILIZATION' \
    --max-model-len '$MAX_MODEL_LEN' \
    --max-num-batched-tokens '$MAX_NUM_BATCHED_TOKENS' \
    --max-num-seqs '$MAX_NUM_SEQS' \
    --block-size '$BLOCK_SIZE' \
    --enforce-eager \
    --no-enable-flashinfer-autotune \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --no-async-scheduling \
    $HEADLESS_FLAG"

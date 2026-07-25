#!/usr/bin/env bash
# Start MiMo-V2.5-NVFP4 on the 2x GB10 Spark cluster
# Runs on BOTH nodes: container startup, mod patches, Ray cluster, vLLM launch
set -eu

IMAGE="ghcr.io/tonyd2wild/mimo-v2.5-tp2-1m-nvfp4kv:20260620"
CONTAINER="vllm_mimo_tp2"
HEAD_IP="10.0.0.1"
WORKER_IP="10.0.0.2"

echo "=== Starting MiMo-V2.5-NVFP4 cluster ==="

# Step 1: Start containers on both nodes (privileged for drop-caches)
echo "--- Starting containers ---"
for HOST in spark2-ts spark3-ts; do
  ssh "$HOST" "docker rm -f $CONTAINER 2>/dev/null || true"
  ssh "$HOST" "docker run -d \
    --name $CONTAINER \
    --privileged \
    --gpus all \
    --network host \
    --ipc host \
    --shm-size 16g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v \$HOME/models:/root/.cache/huggingface \
    -v \$HOME/inference/mimo-v25-recipe:/workspace/recipe \
    $IMAGE sleep infinity" >/dev/null 2>&1
  echo "  $HOST: container up"
done

# Step 2: Apply mods on both nodes
echo "--- Applying vLLM patches ---"
for HOST in spark2-ts spark3-ts; do
  ssh "$HOST" "cd \$HOME/inference/mimo-v25-recipe && bash apply-mods.sh $CONTAINER" >/dev/null 2>&1
  echo "  $HOST: mods applied"
done

# Step 3: Start Ray head on spark2
echo "--- Starting Ray head (spark2) ---"
ssh spark2-ts "docker exec $CONTAINER bash -c '
  export NCCL_NET=IB
  export NCCL_IB_DISABLE=0
  export NCCL_SOCKET_IFNAME=enp1s0f1np1
  export GLOO_SOCKET_IFNAME=enp1s0f1np1
  export NCCL_IB_HCA=rocep1s0f1
  export NCCL_IB_GID_INDEX=3
  export NCCL_CROSS_NIC=1
  export NCCL_CUMEM_ENABLE=0
  export NCCL_NVLS_ENABLE=0
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export RAY_memory_monitor_refresh_ms=0
  export VLLM_HOST_IP=$HEAD_IP
  export RAY_TMPDIR=/dev/shm/ray
  mkdir -p /dev/shm/ray
  ray stop --force 2>/dev/null || true
  ray start --head --port=6379 --node-ip-address=$HEAD_IP --dashboard-host=0.0.0.0 --num-gpus=1 --object-store-memory=1073741824
'" >/dev/null 2>&1
echo "  spark2: Ray head up"

# Step 4: Start Ray worker on spark3
echo "--- Starting Ray worker (spark3) ---"
ssh spark3-ts "docker exec $CONTAINER bash -c '
  export NCCL_NET=IB
  export NCCL_IB_DISABLE=0
  export NCCL_SOCKET_IFNAME=enp1s0f1np1
  export GLOO_SOCKET_IFNAME=enp1s0f1np1
  export NCCL_IB_HCA=rocep1s0f1
  export NCCL_IB_GID_INDEX=3
  export NCCL_CROSS_NIC=1
  export NCCL_CUMEM_ENABLE=0
  export NCCL_NVLS_ENABLE=0
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export RAY_memory_monitor_refresh_ms=0
  export VLLM_HOST_IP=$WORKER_IP
  export RAY_TMPDIR=/dev/shm/ray
  mkdir -p /dev/shm/ray
  ray stop --force 2>/dev/null || true
  ray start --address=$HEAD_IP:6379 --node-ip-address=$WORKER_IP --num-gpus=1 --object-store-memory=1073741824
'" >/dev/null 2>&1
echo "  spark3: Ray worker joined"

# Step 5: Wait for 2 GPUs
echo "--- Waiting for Ray cluster (2 GPUs) ---"
for i in $(seq 1 30); do
  if ssh spark2-ts "docker exec $CONTAINER bash -c 'RAY_ADDRESS=$HEAD_IP:6379 ray status 2>/dev/null'" 2>/dev/null | grep -q "2.0/2.0 GPU"; then
    echo "  2 GPUs visible"
    break
  fi
  sleep 2
done

# Step 6: Launch vLLM on spark2 (head) — output goes to container stdout/stderr
echo "--- Launching vLLM ---"
ssh spark2-ts "docker exec -d $CONTAINER bash -c 'exec bash /workspace/recipe/launch-spark.sh'"
echo "  vLLM launching (model load takes ~8 min)"

# Step 7: Wait for API
echo "--- Waiting for API ---"
for i in $(seq 1 120); do
  if ssh spark2-ts "curl -fsS --max-time 5 http://127.0.0.1:8888/v1/models" >/dev/null 2>&1; then
    echo "  API: UP"
    echo ""
    echo "=== MiMo-V2.5-NVFP4 is running ==="
    echo "=== API: http://100.92.139.82:8888/v1 ==="
    exit 0
  fi
  sleep 10
done

echo "ERROR: API did not come up within 20 minutes. Check logs with: ~/inference/serve/cluster/mimo-v25/logs.sh"
exit 1

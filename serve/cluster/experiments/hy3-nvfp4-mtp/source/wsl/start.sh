#!/usr/bin/env bash
set -euo pipefail

# Hy3-295B NVFP4-W4A16 · 2x DGX Spark TP=2 · Spark2 head + Spark3 worker
# Adapted from tonyd2wild/Hy3-295B-NVFP4-MTP-2x-DGX-Spark for the jjlink Spark2/Spark3 fabric.

HEAD_HOST=${HEAD_HOST:-spark2-ts}
WORKER_HOST=${WORKER_HOST:-spark3-ts}
WORKER_SSH_TARGET=${WORKER_SSH_TARGET:-jjlink@10.0.0.2}
SSH_KEY=${SSH_KEY:-/home/jjlink/.ssh/gb10_ed25519}

CONTAINER_HEAD=${CONTAINER_HEAD:-hy3-head}
CONTAINER_WORKER=${CONTAINER_WORKER:-hy3-worker}
IMAGE=${IMAGE:-hy3-vllm-upstream:ab666069}
MODEL_HOST_DIR=${MODEL_HOST_DIR:-/home/jjlink/models/hy3-nvfp4-w4a16}
MODEL_CONTAINER_DIR=${MODEL_CONTAINER_DIR:-/models}
SERVED_NAME=${SERVED_NAME:-hy3}
PORT=${PORT:-8888}
HEAD_IP=${HEAD_IP:-10.0.0.1}
WORKER_IP=${WORKER_IP:-10.0.0.2}
RAY_PORT=${RAY_PORT:-26480}
FABRIC_IFACE=${FABRIC_IFACE:-enp1s0f1np1}
NCCL_IB_HCA=${NCCL_IB_HCA:-rocep1s0f1}
MAX_LEN=${MAX_LEN:-131072}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-6}
KV_DTYPE=${KV_DTYPE:-auto}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.90}
SPEC_CONFIG=${SPEC_CONFIG:-'{"method":"mtp","num_speculative_tokens":1}'}

ssh_worker=(ssh -i "${SSH_KEY}" -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 "${WORKER_SSH_TARGET}")

echo "== Hy3 preflight on Spark2/Spark3 =="
test -d "${MODEL_HOST_DIR}" || { echo "missing model dir on head: ${MODEL_HOST_DIR}" >&2; exit 1; }
"${ssh_worker[@]}" "test -d '${MODEL_HOST_DIR}'" || { echo "missing model dir on worker: ${MODEL_HOST_DIR}" >&2; exit 1; }

echo "== remove prior Hy3 containers =="
docker rm -f "${CONTAINER_HEAD}" 2>/dev/null || true
"${ssh_worker[@]}" "docker rm -f '${CONTAINER_WORKER}' 2>/dev/null || true"

docker_common=(
  --network host --ipc host --privileged --security-opt label=disable --gpus all
  --entrypoint ray
  --ulimit memlock=-1 --ulimit stack=67108864
  -v "${MODEL_HOST_DIR}:${MODEL_CONTAINER_DIR}:ro"
  -e RAY_memory_usage_threshold=0.99 -e RAY_memory_monitor_refresh_ms=0
  -e RAY_health_check_initial_delay_ms=30000 -e RAY_health_check_period_ms=10000
  -e RAY_health_check_timeout_ms=30000 -e RAY_health_check_failure_threshold=30
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e CUDA_DEVICE_MAX_CONNECTIONS=32
  -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA="${NCCL_IB_HCA}"
  -e NCCL_SOCKET_IFNAME="${FABRIC_IFACE}" -e GLOO_SOCKET_IFNAME="${FABRIC_IFACE}" -e TP_SOCKET_IFNAME="${FABRIC_IFACE}"
  -e NCCL_IB_GID_INDEX=3 -e NCCL_CROSS_NIC=1
  -e NCCL_MAX_NCHANNELS=4 -e NCCL_MIN_NCHANNELS=4
  -e VLLM_USE_B12X_FP8_GEMM=0 -e VLLM_USE_B12X_MOE=0 -e VLLM_USE_B12X_SPARSE_INDEXER=0
)

echo "== start Ray head on Spark2 (${HEAD_IP}) =="
docker run -d --name "${CONTAINER_HEAD}" "${docker_common[@]}" \
  -e VLLM_HOST_IP="${HEAD_IP}" "${IMAGE}" \
  start --head --node-ip-address="${HEAD_IP}" --port="${RAY_PORT}" \
    --num-gpus=1 --object-store-memory=134217728 --disable-usage-stats --block >/dev/null

echo "== start Ray worker on Spark3 (${WORKER_IP}) =="
"${ssh_worker[@]}" "docker run -d --name '${CONTAINER_WORKER}' \
  --network host --ipc host --privileged --security-opt label=disable --gpus all \
  --entrypoint ray \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v '${MODEL_HOST_DIR}:${MODEL_CONTAINER_DIR}:ro' \
  -e RAY_memory_usage_threshold=0.99 -e RAY_memory_monitor_refresh_ms=0 \
  -e RAY_health_check_initial_delay_ms=30000 -e RAY_health_check_period_ms=10000 \
  -e RAY_health_check_timeout_ms=30000 -e RAY_health_check_failure_threshold=30 \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e CUDA_DEVICE_MAX_CONNECTIONS=32 \
  -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA='${NCCL_IB_HCA}' \
  -e NCCL_SOCKET_IFNAME='${FABRIC_IFACE}' -e GLOO_SOCKET_IFNAME='${FABRIC_IFACE}' -e TP_SOCKET_IFNAME='${FABRIC_IFACE}' \
  -e NCCL_IB_GID_INDEX=3 -e NCCL_CROSS_NIC=1 \
  -e NCCL_MAX_NCHANNELS=4 -e NCCL_MIN_NCHANNELS=4 \
  -e VLLM_USE_B12X_FP8_GEMM=0 -e VLLM_USE_B12X_MOE=0 -e VLLM_USE_B12X_SPARSE_INDEXER=0 \
  -e VLLM_HOST_IP='${WORKER_IP}' '${IMAGE}' \
  start --address='${HEAD_IP}:${RAY_PORT}' --node-ip-address='${WORKER_IP}' \
    --num-gpus=1 --object-store-memory=134217728 --disable-usage-stats --block >/dev/null"

echo "== wait for Ray to see 2 nodes =="
for i in $(seq 1 60); do
  node_count=$(docker exec "${CONTAINER_HEAD}" ray status 2>/dev/null | grep -c '^[[:space:]]*1 node_' || true)
  if [ "${node_count}" -ge 2 ]; then
    docker exec "${CONTAINER_HEAD}" ray status | sed -n '/Active:/,/Resources/p'
    break
  fi
  if [ "$i" = 60 ]; then
    echo "Ray did not report 2 active nodes" >&2
    docker exec "${CONTAINER_HEAD}" ray status || true
    exit 1
  fi
  sleep 5
done

serve_script=/tmp/serve-hy3-tools.sh
cat > "${serve_script}" <<SERVE
#!/usr/bin/env bash
set -euo pipefail
exec > /tmp/hy3-serve.log 2>&1

# This checkpoint emits :opensource-suffixed special tokens. Patch stock
# vLLM's bare-token hy_v3 parsers before enabling reasoning and tool parsing.
RP=/usr/local/lib/python3.12/dist-packages/vllm/reasoning/hy_v3_reasoning_parser.py
TP=/usr/local/lib/python3.12/dist-packages/vllm/tool_parsers/hy_v3_tool_parser.py
sed -i 's|"<think>"|"<think:opensource>"|g; s|"</think>"|"</think:opensource>"|g' "\${RP}"
sed -i 's|"<tool_calls>"|"<tool_calls:opensource>"|g; s|"</tool_calls>"|"</tool_calls:opensource>"|g; s|"<tool_call>"|"<tool_call:opensource>"|g; s|"</tool_call>"|"</tool_call:opensource>"|g; s|"<tool_sep>"|"<tool_sep:opensource>"|g; s|"<arg_key>"|"<arg_key:opensource>"|g; s|"</arg_key>"|"</arg_key:opensource>"|g; s|"<arg_value>"|"<arg_value:opensource>"|g; s|"</arg_value>"|"</arg_value:opensource>"|g' "\${TP}"

export VLLM_HOST_IP=${HEAD_IP}
exec vllm serve ${MODEL_CONTAINER_DIR} \
  --served-model-name ${SERVED_NAME} --host 0.0.0.0 --port ${PORT} \
  --tensor-parallel-size 2 \
  --distributed-executor-backend ray \
  --max-model-len ${MAX_LEN} \
  --max-num-seqs ${MAX_NUM_SEQS} \
  --kv-cache-dtype ${KV_DTYPE} \
  --gpu-memory-utilization ${GPU_MEM_UTIL} \
  --speculative-config '${SPEC_CONFIG}' \
  --tool-call-parser hy_v3 --reasoning-parser hy_v3 --enable-auto-tool-choice \
  --trust-remote-code --enforce-eager
SERVE
chmod +x "${serve_script}"
docker cp "${serve_script}" "${CONTAINER_HEAD}:/serve-hy3-tools.sh"
rm -f "${serve_script}"

echo "== launch patched vLLM tools/reasoning serve on Spark2:${PORT} =="
docker exec -d "${CONTAINER_HEAD}" bash /serve-hy3-tools.sh

echo "== wait for OpenAI API health =="
for i in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    curl -fsS "http://127.0.0.1:${PORT}/v1/models"
    echo
    echo "Hy3 is up on Spark2:${PORT}"
    exit 0
  fi
  if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_HEAD}"; then
    echo "head container exited" >&2
    exit 1
  fi
  sleep 10
done

echo "Timed out waiting for API. Tail logs with ./logs.sh head 200" >&2
exit 1

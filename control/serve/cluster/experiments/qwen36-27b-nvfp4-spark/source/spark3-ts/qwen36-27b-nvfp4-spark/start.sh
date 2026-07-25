#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh

remote_recipe_exists() {
  local host="$1"
  ssh "$host" "test -f '$RECIPE_DIR/env.sh' && test -f '$RECIPE_DIR/start-node.sh'"
}

check_prereqs() {
  echo "=== checking prerequisites ==="
  for host in "$HEAD_HOST" "$WORKER_HOST"; do
    echo "--- $host ---"
    ssh "$host" "set -euo pipefail
      test -d '$HOST_MODEL_ROOT/models--nvidia--Qwen3.6-27B-NVFP4'
      test -f '$HOST_MODEL_ROOT/models--nvidia--Qwen3.6-27B-NVFP4/config.json'
      test -f '$HOST_MODEL_ROOT/models--nvidia--Qwen3.6-27B-NVFP4/model.safetensors.index.json'
      docker image inspect '$IMAGE' >/dev/null
      docker run --rm --entrypoint python3 '$IMAGE' -c 'import vllm; print(vllm.__version__)'
    "
  done
}

sync_recipe() {
  echo "=== syncing recipe to Spark nodes ==="
  for host in "$HEAD_HOST" "$WORKER_HOST"; do
    ssh "$host" "mkdir -p '$RECIPE_DIR'"
    rsync -az --delete ./ "$host:$RECIPE_DIR/"
    ssh "$host" "chmod +x '$RECIPE_DIR'/*.sh"
    echo "  synced $host:$RECIPE_DIR"
  done
}

wait_api() {
  echo "=== waiting for API on $HEAD_HOST:$API_PORT ==="
  for i in $(seq 1 180); do
    if ssh "$HEAD_HOST" "curl -fsS --max-time 5 'http://127.0.0.1:$API_PORT/v1/models' >/dev/null" 2>/dev/null; then
      echo "API is up: http://$HEAD_HOST:$API_PORT/v1"
      return 0
    fi
    if (( i % 12 == 0 )); then
      echo "  still waiting ($((i*5))s elapsed)"
      ssh "$HEAD_HOST" "docker logs --tail 8 '$CONTAINER' 2>&1 || true" | sed 's/^/  head log: /'
    fi
    sleep 5
  done
  echo "ERROR: API did not become ready within 15 minutes" >&2
  exit 1
}

check_prereqs
sync_recipe

# Stop only this recipe's containers and the old sleeping qwen test container name.
echo "=== stopping old qwen rank containers ==="
ssh "$WORKER_HOST" "docker rm -f '$CONTAINER' vllm_qwen36_tp2 >/dev/null 2>&1 || true"
ssh "$HEAD_HOST" "docker rm -f '$CONTAINER' vllm_qwen36_tp2 >/dev/null 2>&1 || true"

# Rank1 first, then rank0, matching DSpark multi-node launch discipline.
echo "=== starting rank1 on $WORKER_HOST ==="
ssh "$WORKER_HOST" "cd '$RECIPE_DIR' && ./start-node.sh 1 '$WORKER_IP'"
sleep 5
echo "=== starting rank0 on $HEAD_HOST ==="
ssh "$HEAD_HOST" "cd '$RECIPE_DIR' && ./start-node.sh 0 '$HEAD_IP'"

wait_api
./status.sh || true

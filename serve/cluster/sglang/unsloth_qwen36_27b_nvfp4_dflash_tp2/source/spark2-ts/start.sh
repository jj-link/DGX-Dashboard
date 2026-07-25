#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh

if [[ "$ENABLE_DFLASH" == "1" ]]; then
  echo "[mode] SGLang + DFlash final (TP=2, CUDA graphs disabled for GB10 multi-node stability)"
else
  echo "[mode] SGLang TP=2 NON-DFLASH fallback; DFlash is not silently disabled, see DFLASH-INCOMPATIBILITY.md"
fi
echo "[preflight] checking ports and model files"
ssh "$HEAD_HOST" "test -s '$HOST_HF_ROOT/models--unsloth--Qwen3.6-27B-NVFP4/snapshots/890bdef7a42feba6d83b6e17a03315c694112f2a/model.safetensors' && test -s '$HOST_HF_ROOT/models--z-lab--Qwen3.6-27B-DFlash/snapshots/0919688658996800f86b895034249700e9481106/model.safetensors'"
ssh "$WORKER_HOST" "test -s '$HOST_HF_ROOT/models--unsloth--Qwen3.6-27B-NVFP4/snapshots/890bdef7a42feba6d83b6e17a03315c694112f2a/model.safetensors' && test -s '$HOST_HF_ROOT/models--z-lab--Qwen3.6-27B-DFlash/snapshots/0919688658996800f86b895034249700e9481106/model.safetensors'"
ssh "$HEAD_HOST" "if ss -tln | grep -q ':$API_PORT '; then echo 'Port $API_PORT is already listening on $HEAD_HOST' >&2; exit 3; fi"
ssh "$HEAD_HOST" "mkdir -p '$RECIPE_DIR'"
ssh "$WORKER_HOST" "mkdir -p '$RECIPE_DIR'"
rsync -az --delete ./ "$HEAD_HOST:$RECIPE_DIR/"
rsync -az --delete ./ "$WORKER_HOST:$RECIPE_DIR/"

echo "[start] rank1 worker on $WORKER_HOST"
ssh "$WORKER_HOST" "cd '$RECIPE_DIR' && ENABLE_DFLASH='$ENABLE_DFLASH' DISABLE_CUDA_GRAPH='$DISABLE_CUDA_GRAPH' bash ./start-node.sh 1 '$WORKER_IP'"
sleep 3
echo "[start] rank0 head/API on $HEAD_HOST"
ssh "$HEAD_HOST" "cd '$RECIPE_DIR' && ENABLE_DFLASH='$ENABLE_DFLASH' DISABLE_CUDA_GRAPH='$DISABLE_CUDA_GRAPH' bash ./start-node.sh 0 '$HEAD_IP'"

echo "[wait] /v1/models on http://$HEAD_HOST:$API_PORT/v1/models"
for i in $(seq 1 180); do
  if ssh "$HEAD_HOST" "curl -fsS --max-time 5 http://127.0.0.1:$API_PORT/v1/models >/dev/null"; then
    echo "ready"
    exit 0
  fi
  if ! ssh "$HEAD_HOST" "docker ps --format '{{.Names}}' | grep -qx '$CONTAINER'"; then
    echo "head container exited before readiness" >&2
    ssh "$HEAD_HOST" "docker logs '$CONTAINER' --tail 200 2>&1 || true" >&2
    exit 4
  fi
  sleep 5
done
echo "timed out waiting for API" >&2
ssh "$HEAD_HOST" "docker logs '$CONTAINER' --tail 200 2>&1 || true" >&2
exit 5

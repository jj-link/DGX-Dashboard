#!/usr/bin/env bash
set -euo pipefail
IMAGE="ghcr.io/tonyd2wild/mimo-v2.5-tp2-1m-nvfp4kv:20260620"
CONTAINER="vllm_mimo_tp2"
HF_CACHE="$HOME/models"
RECIPE_DIR="$HOME/inference/mimo-v25-recipe"

docker rm -f "$CONTAINER" 2>/dev/null || true
docker run -d \
  --name "$CONTAINER" \
  --privileged \
  --gpus all \
  --network host \
  --ipc host \
  --shm-size 16g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "$HF_CACHE:/root/.cache/huggingface" \
  -v "$RECIPE_DIR:/workspace/recipe" \
  "$IMAGE" sleep infinity

echo "Container $CONTAINER up on $(hostname)."

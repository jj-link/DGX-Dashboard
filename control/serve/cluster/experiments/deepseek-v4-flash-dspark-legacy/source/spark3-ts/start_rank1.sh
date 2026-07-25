#!/usr/bin/env bash
# Start DeepSeek-V4-Flash-DSpark rank 1 on spark3
# Run this on spark3 only
set -eu
DIR="$HOME/inference/vllm/DeepSeek-V4-Flash-DSpark"

echo "=== Starting rank 1 (spark3) ==="
docker compose -f "$DIR/rank1-docker-compose.yml" up -d

echo "=== Done ==="
echo "Rank 0 will connect from spark2"

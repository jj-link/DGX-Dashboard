#!/usr/bin/env bash
# Start DeepSeek-V4-Flash-DSpark rank 0 on spark2
# Then SSH to spark3 to start rank 1
set -eu
SPARK3_IP=10.0.0.2
DIR="$HOME/inference/vllm/DeepSeek-V4-Flash-DSpark"

echo "=== Starting rank 0 (spark2) ==="
docker compose -f "$DIR/rank0-docker-compose.yml" up -d

echo "=== Starting rank 1 on spark3 via SSH ==="
ssh jjlink@"$SPARK3_IP" "docker compose -f \$HOME/inference/vllm/DeepSeek-V4-Flash-DSpark/rank1-docker-compose.yml up -d"

echo "=== Done ==="
echo "API available on spark2:8000"

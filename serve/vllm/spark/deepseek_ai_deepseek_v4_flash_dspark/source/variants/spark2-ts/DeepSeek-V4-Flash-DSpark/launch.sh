#!/usr/bin/env bash
# Launch TP=2 DeepSeek-V4-Flash-DSpark across spark2 (rank 0) and spark3 (rank 1)
# Run this on spark2. It starts rank 0 locally and rank 1 on spark3 via SSH.
set -eu
SPARK3_IP=10.0.0.2
DIR="$HOME/inference/vllm/DeepSeek-V4-Flash-DSpark"

echo "=== Starting rank 0 (spark2) ==="
docker compose -f "$DIR/rank0-docker-compose.yml" up -d

echo "=== Starting rank 1 (spark3) ==="
ssh jjlink@"$SPARK3_IP" "docker compose -f $DIR/rank1-docker-compose.yml up -d"

echo "=== Done ==="
echo "API on spark2 port 8000"

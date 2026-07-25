#!/usr/bin/env bash
# Stop the TP=2 DeepSeek setup on both sparks
set -eu
SPARK3_IP=10.0.0.2

docker compose -f ~/inference/vllm/DeepSeek-V4-Flash-DSpark/rank0-docker-compose.yml down 2>/dev/null
ssh jjlink@$SPARK3_IP "docker compose -f \$HOME/inference/vllm/DeepSeek-V4-Flash-DSpark/rank1-docker-compose.yml down 2>/dev/null"
echo "Stopped"

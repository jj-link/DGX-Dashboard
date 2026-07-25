#!/usr/bin/env bash
# Stop DeepSeek-V4-Flash-DSpark on the 2x GB10 Spark cluster
# SSHs to both nodes and runs docker compose down.
set -eu

echo "=== Stopping DSpark cluster ==="
ssh spark2-ts "cd /home/jjlink/dspark-recipe && docker compose --env-file .env.dspark -f docker-compose.dspark.yml down"
ssh spark3-ts "cd /home/jjlink/dspark-recipe && docker compose --env-file .env.dspark -f docker-compose.dspark.yml down"
echo "=== Stopped ==="

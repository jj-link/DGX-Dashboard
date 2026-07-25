#!/usr/bin/env bash
# Start DeepSeek-V4-Flash-DSpark on the 2x GB10 Spark cluster
# SSHs to spark2-ts and runs the recipe's start script.
set -eu

echo "=== Starting DSpark cluster ==="
ssh spark2-ts "bash /home/jjlink/dspark-recipe/start-deepseek-v4-flash-dspark.sh"
echo "=== API: http://100.92.139.82:8888/v1 ==="

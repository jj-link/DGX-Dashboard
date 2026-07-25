#!/usr/bin/env bash
# Start Ray head on spark2
set -eu
docker compose -f ~/inference/sglang/qwen36_nvfp4_tp2/rank0-docker-compose.yml up -d
echo "Ray head started on spark2"

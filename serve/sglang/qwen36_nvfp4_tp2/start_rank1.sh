#!/usr/bin/env bash
# Start Ray worker on spark3
set -eu
docker compose -f ~/inference/sglang/qwen36_nvfp4_tp2/rank1-docker-compose.yml up -d
echo "Ray worker started on spark3"

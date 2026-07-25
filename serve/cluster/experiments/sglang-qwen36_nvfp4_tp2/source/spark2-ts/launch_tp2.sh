#!/usr/bin/env bash
# Launch SGLang TP=2 across spark2+spark3
# Run on spark2 AFTER both Ray containers are running
set -eu

echo "=== Launching SGLang TP=2 ==="
docker run --rm --network host --gpus all \
  -v ${HOME}/models:/models \
  -e NCCL_SOCKET_IFNAME=enp1s0f1np1 \
  -e NCCL_IB_DISABLE=1 \
  -e NCCL_DEBUG=info \
  lmsysorg/sglang:v0.5.12-cu130 \
  python3 -m sglang.launch_server \
  --model-path /models/hub/models--unsloth--Qwen3.6-27B-NVFP4/snapshots/890bdef7a42feba6d83b6e17a03315c694112f2a \
  --host 0.0.0.0 --port 30000 \
  --served-model-name Qwen3.6-27B-NVFP4-TP2 \
  --trust-remote-code \
  --tp-size 2 \
  --attention-backend flashinfer \
  --context-length 262144 \
  --mem-fraction-static 0.85 \
  --enable-metrics \
  --max-running-requests 8 \
  --chunked-prefill-size 2048 \
  --ray-address 10.0.0.1:6379

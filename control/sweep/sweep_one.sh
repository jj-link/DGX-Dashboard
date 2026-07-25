#!/usr/bin/env bash
# Restart the NVFP4 container with whatever env is exported, wait for
# readiness, run the short benchmark. Usage:
#   MOE_BACKEND=marlin VLLM_MARLIN_USE_ATOMIC_ADD=1 ./sweep_one.sh "label"
set -uo pipefail
LABEL="${1:-unlabeled}"
cd "$(dirname "$0")"

docker rm -f vllm-serve_qwen36_a3b_nvfp4 >/dev/null 2>&1 || true
DETACH=1 ../run_vllm_docker.sh serve/vllm/serve_qwen36_a3b_nvfp4.sh >/dev/null

echo "[$LABEL] waiting for server..."
for i in $(seq 1 180); do
  curl -s -m 2 http://localhost:8000/v1/models >/dev/null 2>&1 && { echo "[$LABEL] up after ${i}0s"; break; }
  sleep 10
done
curl -s -m 3 http://localhost:8000/v1/models >/dev/null 2>&1 || { echo "[$LABEL] SERVER DID NOT COME UP"; docker logs --tail 40 vllm-serve_qwen36_a3b_nvfp4 2>&1; exit 1; }

echo "[$LABEL] kernel selection:"
docker logs vllm-serve_qwen36_a3b_nvfp4 2>&1 | grep -iE 'NVFP4 GEMM|NvFp4 MoE backend|Forced NVFP4' | tail -3
echo "[$LABEL] benchmark:"
python ../bench/bench_tps.py --base http://localhost:8000 --model Qwen3.6-35B-A3B --max-tokens 512 --runs 4 --warmup 1

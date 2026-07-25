#!/usr/bin/env bash
# Generic: restart <serve_script> in the hardened container with exported
# env, wait for readiness, run the short benchmark.
#   MOE_BACKEND=deepgemm ./sweep_model.sh serve/vllm/serve_qwen36_a3b_fp8.sh "fp8+deepgemm"
set -uo pipefail
SERVE="${1:?usage: sweep_model.sh <serve_script.sh> <label>}"
LABEL="${2:-unlabeled}"
cd "$(dirname "$0")"
CNAME="vllm-$(basename "$SERVE" .sh)"

docker rm -f "$CNAME" >/dev/null 2>&1 || true
DETACH=1 ../run_vllm_docker.sh "$SERVE" >/dev/null

echo "[$LABEL] waiting for server..."
for i in $(seq 1 180); do
  curl -s -m 2 http://localhost:8000/v1/models >/dev/null 2>&1 && { echo "[$LABEL] up after ${i}0s"; break; }
  sleep 10
done
curl -s -m 3 http://localhost:8000/v1/models >/dev/null 2>&1 || { echo "[$LABEL] SERVER DID NOT COME UP"; docker logs --tail 40 "$CNAME" 2>&1; exit 1; }

echo "[$LABEL] kernel selection:"
docker logs "$CNAME" 2>&1 | grep -iE 'MoE backend|NVFP4 GEMM|Forced|fp8' | grep -iE 'backend|kernel' | tail -3
echo "[$LABEL] benchmark:"
python ../bench/bench_tps.py --base http://localhost:8000 --model Qwen3.6-35B-A3B --max-tokens 512 --runs 4 --warmup 1

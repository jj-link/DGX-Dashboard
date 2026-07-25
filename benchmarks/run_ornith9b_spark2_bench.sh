#!/usr/bin/env bash
# Run oneshot benchmark against spark2 Ornith-1.0-9B-NVFP4-AWQ (port 8000) with concurrency 16.
set -uo pipefail
cd /home/workbench/inference/benchmarks
BASE_URL="http://100.86.3.45:8000/v1"
ALIAS="spark2-ornith-9b-nvfp4"
CONCURRENCY=16
MAX_TOKENS=32768
TIMEOUT=1800

echo "[pre] waiting for $BASE_URL to respond..."
for i in $(seq 1 300); do
  if curl -s --max-time 5 "$BASE_URL/models" > /dev/null 2>&1; then
    echo "[pre] server ready"
    break
  fi
  sleep 2
done

curl -s --max-time 5 "$BASE_URL/models" > /dev/null 2>&1 || {
  echo "[fatal] server not ready after 10 min"
  exit 1
}

OPENAI_API_BASE="$BASE_URL" python3 oneshot_bench.py \
  "$ALIAS" \
  --num-tests -1 \
  --max-tokens "$MAX_TOKENS" \
  --timeout "$TIMEOUT" \
  --test-timeout 60 \
  --concurrency "$CONCURRENCY"

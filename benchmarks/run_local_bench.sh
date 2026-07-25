#!/usr/bin/env bash
# Run oneshot benchmark against local RTX PRO 6000 (port 8000) with concurrency 8.
# Waits for the vLLM server to be ready, then runs all 6 languages (all problems).
set -uo pipefail

cd /home/workbench/inference/benchmarks

BASE_URL="http://localhost:8000/v1"
ALIAS="local-aeon-ultimate"
CONCURRENCY=8
MAX_TOKENS=32768
TIMEOUT=1800

# Wait for server readiness
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

# Run benchmark — all problems, all languages
OPENAI_API_BASE="$BASE_URL" python3 oneshot_bench.py \
  "$ALIAS" \
  --num-tests -1 \
  --max-tokens "$MAX_TOKENS" \
  --timeout "$TIMEOUT" \
  --test-timeout 60 \
  --concurrency "$CONCURRENCY"

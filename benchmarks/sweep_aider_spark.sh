#!/usr/bin/env bash
# Multi-turn aider sweep across the 6 Qwen3.6 quant variants on the SPARK.
# For each model: swap it onto the spark's single vLLM, read vLLM's own
# reported max-concurrency, set --threads from it (capped at the server's
# --max-num-seqs 32), run run_aider_spark.sh (full python polyglot), collect
# stats. Sequential by necessity (spark serves one model at a time).
#
# Concurrency basis: vLLM prints "Maximum concurrency for N tokens per
# request: Xx" at load = KV_tokens / max_model_len. bf16 weights leave less
# KV -> lower X -> fewer threads; fp8/nvfp4 -> higher. This is the correct
# per-model size-scaling (GPU-VRAM bound), not a host-RAM heuristic.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/spark-serve/models.env"

LANGS="${LANGS:-python}"
NUM_TESTS="${NUM_TESTS:--1}"
SPARK_HOST="${SPARK_HOST:-spark}"
TS=$(date +%Y%m%d-%H%M%S)
LOG="${SCRIPT_DIR}/results/sweep-aider-spark-${TS}.log"
mkdir -p "${SCRIPT_DIR}/results"
exec > >(tee -a "$LOG") 2>&1

# ALIASES_OVERRIDE (space-separated) runs a subset; otherwise all SWEEP_ALIASES.
if [ -n "${ALIASES_OVERRIDE:-}" ]; then read -ra SWEEP_ALIASES <<< "$ALIASES_OVERRIDE"; fi
echo "[sweep] models: ${SWEEP_ALIASES[*]}"
echo "[sweep] langs=$LANGS num_tests=$NUM_TESTS log=$LOG"

for alias in "${SWEEP_ALIASES[@]}"; do
  echo "======== $alias :: swap $(date +%H:%M:%S) ========"
  if ! READY_TIMEOUT=1800 "${SCRIPT_DIR}/spark-serve/swap.sh" "$alias"; then
    echo "[sweep] SWAP FAILED for $alias — skipping"; continue
  fi
  # Concurrency = KV-cache tokens / realistic per-request tokens, capped at the
  # problem count. KV size (read from the spark's vLLM container) shrinks with
  # model size, so this auto-scales. Use a realistic per-request size, NOT the
  # full context window (that under-counts ~10x). Under-estimate is safe — vLLM
  # queues the excess.
  EST_TOKENS_PER_REQ="${EST_TOKENS_PER_REQ:-32768}"
  KV=$(ssh -o BatchMode=yes "$SPARK_HOST" \
        "docker logs vllm 2>&1 | grep -oE 'GPU KV cache size: [0-9,]+ tokens' | tail -1 | grep -oE '[0-9,]+'" \
        2>/dev/null | tr -d ',' || true)
  NPROB=0
  for L in $(echo "$LANGS" | tr ',' ' '); do
    NPROB=$((NPROB + $(ls -d "${SCRIPT_DIR}/aider/tmp.benchmarks/polyglot-benchmark/${L}/exercises/practice/"*/ 2>/dev/null | wc -l)))
  done
  [ "$NPROB" -lt 1 ] && NPROB=34
  if [ -n "$KV" ]; then
    THREADS=$(awk -v kv="$KV" -v e="$EST_TOKENS_PER_REQ" -v np="$NPROB" 'BEGIN{t=int(kv/e); if(t<1)t=1; if(t>np)t=np; print t}')
  else
    THREADS=$NPROB
    echo "[sweep] WARN: no KV size for $alias; THREADS=problem count ($NPROB)"
  fi
  echo "======== $alias :: KV=${KV:-?}tok /${EST_TOKENS_PER_REQ}est NPROB=$NPROB -> THREADS=$THREADS :: bench $(date +%H:%M:%S) ========"
  THREADS="$THREADS" NUM_TESTS="$NUM_TESTS" LANGS="$LANGS" \
    "${SCRIPT_DIR}/run_aider_spark.sh" --languages "$LANGS" --num-tests "$NUM_TESTS" \
    || echo "[sweep] BENCH returned nonzero for $alias (continuing)"
  echo "======== $alias :: done $(date +%H:%M:%S) ========"
done

echo "[sweep] === per-model results ==="
for d in "${SCRIPT_DIR}"/aider/tmp.benchmarks/*-aider-spark-*; do
  [ -f "$d/_stats.yml" ] || continue
  echo "--- $(basename "$d") ---"
  grep -E 'pass_rate_1|pass_rate_2|test_cases|seconds_per_case' "$d/_stats.yml" 2>/dev/null
done
echo "SWEEP_COMPLETE $(date +%H:%M:%S)"

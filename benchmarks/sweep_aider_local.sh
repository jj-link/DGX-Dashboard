#!/usr/bin/env bash
# Multi-turn aider sweep across the LOCAL_ALIASES Qwen3.6 quant variants on
# the RTX 6000 (local). For each model: swap it via local-serve/swap.sh,
# read vLLM's reported max-concurrency, set --threads from it (capped at
# the server's --max-num-seqs cap), run run_aider_local.sh (full python
# polyglot, streaming:true via benchmark.py:827 stream=True). Sequential
# by necessity (one GPU, one model at a time).
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/local-serve/models.env"

LANGS="${LANGS:-python}"
NUM_TESTS="${NUM_TESTS:--1}"
TS=$(date +%Y%m%d-%H%M%S)
LOG="${SCRIPT_DIR}/results/sweep-aider-local-${TS}.log"
mkdir -p "${SCRIPT_DIR}/results"
exec > >(tee -a "$LOG") 2>&1

# ALIASES (or legacy ALIASES_OVERRIDE) runs a subset; otherwise all LOCAL_ALIASES.
if [ -n "${ALIASES:-}" ]; then
  read -ra ALIASES <<< "$ALIASES"
elif [ -n "${ALIASES_OVERRIDE:-}" ]; then
  read -ra ALIASES <<< "$ALIASES_OVERRIDE"
else
  ALIASES=("${LOCAL_ALIASES[@]}")
fi
echo "[sweep] models: ${ALIASES[*]}"
echo "[sweep] langs=$LANGS num_tests=$NUM_TESTS log=$LOG"

for alias in "${ALIASES[@]}"; do
  echo "======== $alias :: swap $(date +%H:%M:%S) ========"
  if ! READY_TIMEOUT=1800 "${SCRIPT_DIR}/local-serve/swap.sh" "$alias"; then
    echo "[sweep] SWAP FAILED for $alias — skipping"; continue
  fi
  # Concurrency = KV-cache tokens / realistic per-request tokens, capped at the
  # problem count (can't parallelize more exercises than exist). vLLM reports
  # the KV size per model, so this auto-scales with model size: a big model
  # leaves less VRAM for KV -> fewer threads; a small/quantized one -> more.
  # Using a realistic per-request estimate (NOT the full context window) is the
  # whole point — the full-window number under-counts ~10x. Under-estimating is
  # safe: vLLM queues anything that doesn't fit.
  EST_TOKENS_PER_REQ="${EST_TOKENS_PER_REQ:-32768}"
  VLOG=$(ls -t /tmp/local-vllm-logs/${alias}-*.log 2>/dev/null | head -1)
  KV=""
  [ -n "$VLOG" ] && KV=$(grep -oE 'GPU KV cache size: [0-9,]+ tokens' "$VLOG" | tail -1 | grep -oE '[0-9,]+' | tr -d ',')
  NPROB=0
  for L in $(echo "$LANGS" | tr ',' ' '); do
    NPROB=$((NPROB + $(ls -d "${SCRIPT_DIR}/aider/tmp.benchmarks/polyglot-benchmark/${L}/exercises/practice/"*/ 2>/dev/null | wc -l)))
  done
  [ "$NPROB" -lt 1 ] && NPROB=34
  if [ -n "$KV" ]; then
    THREADS=$(awk -v kv="$KV" -v e="$EST_TOKENS_PER_REQ" -v np="$NPROB" 'BEGIN{t=int(kv/e); if(t<1)t=1; if(t>np)t=np; print t}')
  else
    THREADS=$NPROB  # no KV reading -> submit all; vLLM queues whatever doesn't fit
    echo "[sweep] WARN: no KV size for $alias from $VLOG; THREADS=problem count ($NPROB)"
  fi
  echo "======== $alias :: KV=${KV:-?}tok /${EST_TOKENS_PER_REQ}est NPROB=$NPROB -> THREADS=$THREADS :: bench $(date +%H:%M:%S) ========"
  # local-serve/swap.sh is now docker-based with `-p 127.0.0.1:8000:8000`,
  # so localhost reaches the container fine — no need to override OPENAI_API_BASE.
  THREADS="$THREADS" NUM_TESTS="$NUM_TESTS" LANGS="$LANGS" \
    "${SCRIPT_DIR}/run_aider_local.sh" --languages "$LANGS" --num-tests "$NUM_TESTS" \
    || echo "[sweep] BENCH returned nonzero for $alias (continuing)"
  echo "======== $alias :: done $(date +%H:%M:%S) ========"
done

echo "[sweep] === per-model results ==="
for d in "${SCRIPT_DIR}"/aider/tmp.benchmarks/*-aider-local-*; do
  [ -f "$d/_stats.yml" ] || continue
  echo "--- $(basename "$d") ---"
  grep -E 'pass_rate_1|pass_rate_2|test_cases|seconds_per_case' "$d/_stats.yml" 2>/dev/null
done
echo "SWEEP_COMPLETE $(date +%H:%M:%S)"

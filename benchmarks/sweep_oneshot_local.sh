#!/usr/bin/env bash
# Local sweep: runs oneshot_bench against the LOCAL vLLM (RTX PRO 6000).
# Used in parallel with the spark sweep so the 4 Qwens don't share the
# GB10's slow decode path with minimax.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=local-serve/models.env
source "${SCRIPT_DIR}/local-serve/models.env"

ALIASES="${ALIASES:-${LOCAL_ALIASES[*]}}"
NUM_TESTS="${NUM_TESTS:-34}"
KEYWORDS="${KEYWORDS:-}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
TIMEOUT="${TIMEOUT:-3600}"
TEST_TIMEOUT="${TEST_TIMEOUT:-60}"
CONCURRENCY="${CONCURRENCY:-auto}"
EST_TOKENS_PER_REQ="${EST_TOKENS_PER_REQ:-$MAX_TOKENS}"
OPENAI_API_BASE="${OPENAI_API_BASE:-${LOCAL_URL:-http://localhost:8000/v1}}"
export READY_TIMEOUT="${READY_TIMEOUT:-1200}"

mkdir -p "${SCRIPT_DIR}/results"
SWEEP_LOG="${SCRIPT_DIR}/results/sweep-local-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$SWEEP_LOG") 2>&1
echo "[lsweep] log: $SWEEP_LOG"
echo "[lsweep] aliases: $ALIASES"
echo "[lsweep] N=$NUM_TESTS, max_tokens=$MAX_TOKENS, timeout=${TIMEOUT}s, concurrency=$CONCURRENCY"
echo "[lsweep] est_tokens_per_req=$EST_TOKENS_PER_REQ"
echo "[lsweep] api: $OPENAI_API_BASE"

for alias in $ALIASES; do
  echo ""
  echo "============================================================"
  echo "[$(date '+%H:%M:%S')] LOCAL model: $alias"
  echo "============================================================"
  if ! "${SCRIPT_DIR}/local-serve/swap.sh" "$alias"; then
    echo "[lsweep] swap to $alias FAILED — skipping"
    continue
  fi

  RUN_CONCURRENCY="$CONCURRENCY"
  if [ "$RUN_CONCURRENCY" = "auto" ]; then
    VLOG=$(ls -t /tmp/local-vllm-logs/${alias}-*.log 2>/dev/null | head -1)
    KV=""
    [ -n "$VLOG" ] && KV=$(grep -oE 'GPU KV cache size: [0-9,]+ tokens' "$VLOG" | tail -1 | grep -oE '[0-9,]+' | tr -d ',')
    if [ -n "$KV" ]; then
      RUN_CONCURRENCY=$(awk -v kv="$KV" -v e="$EST_TOKENS_PER_REQ" -v n="$NUM_TESTS" 'BEGIN{t=int(kv/e); if(t<1)t=1; if(t>n)t=n; print t}')
      echo "[lsweep] auto concurrency: KV=${KV}tok / ${EST_TOKENS_PER_REQ}est -> $RUN_CONCURRENCY"
    else
      RUN_CONCURRENCY="$NUM_TESTS"
      echo "[lsweep] WARN: no KV size found in ${VLOG:-/tmp/local-vllm-logs/${alias}-*.log}; using concurrency=$RUN_CONCURRENCY"
    fi
  fi

  OPENAI_API_BASE="$OPENAI_API_BASE" \
  OPENAI_API_KEY="dummy" \
    python3 "${SCRIPT_DIR}/oneshot_bench.py" "$alias" \
      --num-tests "$NUM_TESTS" \
      --keywords "$KEYWORDS" \
      --max-tokens "$MAX_TOKENS" \
      --timeout "$TIMEOUT" \
      --test-timeout "$TEST_TIMEOUT" \
      --concurrency "$RUN_CONCURRENCY" \
    || echo "[lsweep] bench for $alias failed — continuing"
done

echo ""
echo "[$(date '+%H:%M:%S')] local sweep done"
python3 "${SCRIPT_DIR}/aggregate_minimal.py"

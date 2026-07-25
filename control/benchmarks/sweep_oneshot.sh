#!/usr/bin/env bash
# Run oneshot_bench.py across all 5 models using the SAME problem set, so
# pass rates are directly comparable. Swaps the spark vllm container between
# models.
#
# Usage:
#   ./sweep_oneshot.sh                       # 5 problems × 5 models, all alphabetical-first
#   ALIASES="qwen36-27b-fp8" ./sweep_oneshot.sh
#   NUM_TESTS=10 ./sweep_oneshot.sh          # 10 problems per model
#   KEYWORDS="leap,pig-latin" ./sweep_oneshot.sh
set -uo pipefail   # NOT -e: per-model failures shouldn't abort the whole sweep

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=spark-serve/models.env
source "${SCRIPT_DIR}/spark-serve/models.env"

ALIASES="${ALIASES:-${ALL_ALIASES[*]}}"
NUM_TESTS="${NUM_TESTS:-5}"
KEYWORDS="${KEYWORDS:-}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
TIMEOUT="${TIMEOUT:-900}"
TEST_TIMEOUT="${TEST_TIMEOUT:-60}"
# Minimax + large MoE models can take 15+ min to load weights cold —
# bump the ready timeout for swap.sh.
export READY_TIMEOUT="${READY_TIMEOUT:-1800}"

mkdir -p "${SCRIPT_DIR}/results"
SWEEP_LOG="${SCRIPT_DIR}/results/sweep-oneshot-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$SWEEP_LOG") 2>&1
echo "[sweep] log: $SWEEP_LOG"
echo "[sweep] aliases: $ALIASES"
echo "[sweep] N=$NUM_TESTS, keywords='${KEYWORDS}', max_tokens=$MAX_TOKENS, timeout=${TIMEOUT}s"

for alias in $ALIASES; do
  echo ""
  echo "============================================================"
  echo "[$(date '+%H:%M:%S')] model: $alias"
  echo "============================================================"
  if ! "${SCRIPT_DIR}/spark-serve/swap.sh" "$alias"; then
    echo "[sweep] swap to $alias FAILED — skipping this model"
    continue
  fi

  if ! python3 "${SCRIPT_DIR}/oneshot_bench.py" "$alias" \
      --num-tests "$NUM_TESTS" \
      --keywords "$KEYWORDS" \
      --max-tokens "$MAX_TOKENS" \
      --timeout "$TIMEOUT" \
      --test-timeout "$TEST_TIMEOUT"; then
    echo "[sweep] bench for $alias FAILED — continuing to next model"
  fi
done

echo ""
echo "[$(date '+%H:%M:%S')] sweep done. Aggregating..."
python3 "${SCRIPT_DIR}/aggregate_minimal.py"

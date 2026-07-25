#!/usr/bin/env bash
# Run the full 5-model head-to-head. For each model:
#   1. SSH to spark and swap the vllm container to that model.
#   2. Wait for /v1/models to confirm ready.
#   3. Run aider polyglot against it.
#   4. (When wired) run BFCL Live AST.
#   5. Move on.
#
# Resumable: if a model's aider run dir already has a _stats.yml, skip it.
#
# Usage:
#   ./sweep.sh              # all 5 models, Python only, default tries
#   ALIASES="minimax qwen36-27b-fp8" ./sweep.sh      # subset
#   LANGS="python,javascript" ./sweep.sh             # more languages
#   NUM_TESTS=10 ./sweep.sh                          # quick pass

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=spark-serve/models.env
source "${SCRIPT_DIR}/spark-serve/models.env"

ALIASES="${ALIASES:-${ALL_ALIASES[*]}}"
LANGS="${LANGS:-python}"
NUM_TESTS="${NUM_TESTS:--1}"
TRIES="${TRIES:-2}"
SKIP_AIDER="${SKIP_AIDER:-0}"
SKIP_BFCL="${SKIP_BFCL:-1}"  # default off until BFCL handlers are registered

mkdir -p "${SCRIPT_DIR}/results"
SWEEP_LOG="${SCRIPT_DIR}/results/sweep-$(date +%Y%m%d-%H%M%S).log"
echo "[sweep] log: $SWEEP_LOG"

for alias in $ALIASES; do
  echo "" | tee -a "$SWEEP_LOG"
  echo "============================================================" | tee -a "$SWEEP_LOG"
  echo "[sweep] $(date '+%H:%M:%S') model: $alias" | tee -a "$SWEEP_LOG"
  echo "============================================================" | tee -a "$SWEEP_LOG"

  # Skip if we already have an aider run for this alias
  if [ "$SKIP_AIDER" != "1" ]; then
    EXISTING=$(ls -d "${SCRIPT_DIR}/aider/tmp.benchmarks/"*--"${alias}"-* 2>/dev/null | head -1 || true)
    if [ -n "$EXISTING" ] && [ -f "$EXISTING/_stats.yml" ]; then
      echo "[sweep] aider results already exist for $alias: $EXISTING — skipping" | tee -a "$SWEEP_LOG"
      continue
    fi
  fi

  # Swap on spark
  "${SCRIPT_DIR}/spark-serve/swap.sh" "$alias" 2>&1 | tee -a "$SWEEP_LOG"

  # Aider polyglot
  if [ "$SKIP_AIDER" != "1" ]; then
    NUM_TESTS="$NUM_TESTS" TRIES="$TRIES" LANGS="$LANGS" \
      "${SCRIPT_DIR}/run_aider_spark.sh" "$alias" 2>&1 | tee -a "$SWEEP_LOG"
  fi

  # BFCL — placeholder until registry entries land
  if [ "$SKIP_BFCL" != "1" ]; then
    "${SCRIPT_DIR}/run_bfcl.sh" "$alias" 2>&1 | tee -a "$SWEEP_LOG"
  fi
done

echo "" | tee -a "$SWEEP_LOG"
echo "[sweep] all done. Aggregating..." | tee -a "$SWEEP_LOG"
"${SCRIPT_DIR}/aggregate.sh" 2>&1 | tee -a "$SWEEP_LOG"

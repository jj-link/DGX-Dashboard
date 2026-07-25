#!/usr/bin/env bash
# BFCL v3 Live AST on the spark vllm. Single alias (minimax). Run AFTER the
# spark polyglot sweep completes so the vllm endpoint isn't contended.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ALIAS="${ALIAS:-minimax}"
CATEGORY="${CATEGORY:-live}"
export READY_TIMEOUT="${READY_TIMEOUT:-1800}"

mkdir -p "${SCRIPT_DIR}/results/bfcl"
SWEEP_LOG="${SCRIPT_DIR}/results/sweep-bfcl-spark-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$SWEEP_LOG") 2>&1
echo "[bfcl-spark] log: $SWEEP_LOG"
echo "[bfcl-spark] alias: $ALIAS  category: $CATEGORY"

echo ""
echo "============================================================"
echo "[$(date '+%H:%M:%S')] SPARK BFCL: $ALIAS"
echo "============================================================"
if ! "${SCRIPT_DIR}/spark-serve/swap.sh" "$ALIAS"; then
  echo "[bfcl-spark] swap to $ALIAS FAILED"
  exit 1
fi
if ! CATEGORY="$CATEGORY" "${SCRIPT_DIR}/run_bfcl.sh" "$ALIAS"; then
  echo "[bfcl-spark] bfcl for $ALIAS FAILED"
  exit 1
fi

echo ""
echo "[$(date '+%H:%M:%S')] spark BFCL sweep done"

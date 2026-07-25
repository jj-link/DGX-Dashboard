#!/usr/bin/env bash
# BFCL v3 Live AST sweep across the 4 local Qwens. Runs AFTER local
# polyglot sweep completes so the vllm endpoint isn't contended.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=local-serve/models.env
source "${SCRIPT_DIR}/local-serve/models.env"

ALIASES="${ALIASES:-${LOCAL_ALIASES[*]}}"
CATEGORY="${CATEGORY:-live}"
export READY_TIMEOUT="${READY_TIMEOUT:-1200}"

mkdir -p "${SCRIPT_DIR}/results/bfcl"
SWEEP_LOG="${SCRIPT_DIR}/results/sweep-bfcl-local-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$SWEEP_LOG") 2>&1
echo "[bfcl-local] log: $SWEEP_LOG"
echo "[bfcl-local] aliases: $ALIASES"
echo "[bfcl-local] category: $CATEGORY"

for alias in $ALIASES; do
  echo ""
  echo "============================================================"
  echo "[$(date '+%H:%M:%S')] LOCAL BFCL: $alias"
  echo "============================================================"
  if ! "${SCRIPT_DIR}/local-serve/swap.sh" "$alias"; then
    echo "[bfcl-local] swap to $alias FAILED — skipping"
    continue
  fi
  if ! REMOTE_OPENAI_BASE_URL="http://192.168.86.84:8000/v1" \
       CATEGORY="$CATEGORY" \
       "${SCRIPT_DIR}/run_bfcl.sh" "$alias"; then
    echo "[bfcl-local] bfcl for $alias FAILED — continuing"
  fi
done

echo ""
echo "[$(date '+%H:%M:%S')] local BFCL sweep done"

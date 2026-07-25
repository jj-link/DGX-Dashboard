#!/usr/bin/env bash
set -euo pipefail

TUNE_SCRIPT="$HOME/inference/sglang/scripts/tune-dflash-extended.sh"
BASE="$HOME/inference/sglang"
LOG="$BASE/nvfp4-sweep.log"

echo "=== NVFP4 Extended Sweep at $(date) ===" | tee -a "$LOG"

DFLASH_TUNE_SESSION="tune-nvfp4" \
DFLASH_TUNE_LOG="tune-nvfp4.log" \
"$TUNE_SCRIPT" "$BASE/Qwen3.6-27B-NVFP4-DFlash" 2>&1 | tee -a "$LOG"

echo "=== NVFP4 SWEEP COMPLETE at $(date) ===" | tee -a "$LOG"

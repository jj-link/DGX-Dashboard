#!/usr/bin/env bash
set -euo pipefail

TUNE_SCRIPT="$HOME/inference/sglang/scripts/tune-dflash-extended.sh"
BASE="$HOME/inference/sglang"

MODELS=(
  Qwen3.6-27B-FP8-DFlash
  Qwen3.6-27B-BF16-DFlash
  Qwen3.6-27B-NVFP4-DFlash
  Qwen3.6-35B-A3B-FP8-DFlash
  Qwen3.6-35B-A3B-BF16-DFlash
  Qwen3.6-35B-A3B-NVFP4-DFlash
)

LOG="$BASE/all-dflash-tune.log"

echo "=== Starting all-model DFlash tuning at $(date) ===" | tee -a "$LOG"

for model in "${MODELS[@]}"; do
  echo "" | tee -a "$LOG"
  echo "========================================" | tee -a "$LOG"
  echo "TUNING: $model at $(date)" | tee -a "$LOG"
  echo "========================================" | tee -a "$LOG"

  DFLASH_TUNE_SESSION="tune-${model}" DFLASH_TUNE_LOG="tune-all-${model}.log" "$TUNE_SCRIPT" "$BASE/$model" 2>&1 | tee -a "$LOG" || {
    echo "FAILED: $model" | tee -a "$LOG"
  }

  echo "DONE: $model at $(date)" | tee -a "$LOG"
  sleep 30
done

echo "" | tee -a "$LOG"
echo "=== ALL TUNING COMPLETE at $(date) ===" | tee -a "$LOG"

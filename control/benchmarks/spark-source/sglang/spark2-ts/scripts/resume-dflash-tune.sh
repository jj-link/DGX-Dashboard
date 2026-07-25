#!/usr/bin/env bash
set -euo pipefail

TUNE_SCRIPT="$HOME/inference/sglang/scripts/tune-dflash-extended.sh"
BASE="$HOME/inference/sglang"
LOG="$BASE/all-dflash-tune-resume.log"

echo "=== Resuming DFlash tuning at $(date) ===" | tee -a "$LOG"

# Resume Qwen3.6-27B-FP8-DFlash: remaining candidates (draft_tokens=20, windows 2048 4096 6144 8192)
echo "" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
echo "RESUMING: Qwen3.6-27B-FP8-DFlash (remaining candidates) at $(date)" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"

DFLASH_TUNE_SESSION="tune-resume-fp8" \
DFLASH_TUNE_LOG="tune-resume-fp8.log" \
USER_TOKENS_LIST="20" \
TOKENS_LIST="20" \
WINDOW_LIST="2048 4096 6144 8192" \
"$TUNE_SCRIPT" "$BASE/Qwen3.6-27B-FP8-DFlash" 2>&1 | tee -a "$LOG" || {
  echo "FAILED: Qwen3.6-27B-FP8-DFlash resume" | tee -a "$LOG"
}

echo "DONE: Qwen3.6-27B-FP8-DFlash resume at $(date)" | tee -a "$LOG"
sleep 30

# Now run the remaining 5 models
MODELS=(
  "Qwen3.6-27B-BF16-DFlash"
  "Qwen3.6-27B-NVFP4-DFlash"
  "Qwen3.6-35B-A3B-FP8-DFlash"
  "Qwen3.6-35B-A3B-BF16-DFlash"
  "Qwen3.6-35B-A3B-NVFP4-DFlash"
)

for model in "${MODELS[@]}"; do
  echo "" | tee -a "$LOG"
  echo "========================================" | tee -a "$LOG"
  echo "TUNING: $model at $(date)" | tee -a "$LOG"
  echo "========================================" | tee -a "$LOG"

  DFLASH_TUNE_SESSION="tune-${model}" \
  DFLASH_TUNE_LOG="tune-all-${model}.log" \
  "$TUNE_SCRIPT" "$BASE/$model" 2>&1 | tee -a "$LOG" || {
    echo "FAILED: $model" | tee -a "$LOG"
  }

  echo "DONE: $model at $(date)" | tee -a "$LOG"
  sleep 30
done

echo "" | tee -a "$LOG"
echo "=== ALL TUNING COMPLETE at $(date) ===" | tee -a "$LOG"

#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
IMAGE="${1:-qwen-vllm:fp8-tiles-cutlass46-bootstrap}"
CONTAINER_NAME="${CONTAINER_NAME:-fp8-tiles-cutlass46-tune}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
MODEL="${MODEL:-Qwen/Qwen3.6-35B-A3B-FP8}"
REVISION="${REVISION:-95a723d08a9490559dae23d0cff1d9466213d989}"
NUM_SPEC="${NUM_SPEC:-15}"
BATCH_SIZES="${BATCH_SIZES:-1,2,3,4}"

if [[ "$IMAGE" == "qwen-vllm:dflash" ]]; then
  echo "tuning requires the compiled fp8-tiles candidate image" >&2
  exit 2
fi

REVISION_ARGS=()
[[ -n "$REVISION" ]] && REVISION_ARGS=(--revision "$REVISION")

exec docker run --rm \
  --device nvidia.com/gpu=all \
  --name "$CONTAINER_NAME" \
  --security-opt no-new-privileges \
  --cap-drop ALL \
  -e HOME=/tmp \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e HF_HOME=/models \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -v "$HF_CACHE:/models:ro" \
  -v "$ROOT:/experiment" \
  -w /experiment \
  --entrypoint python3 \
  "$IMAGE" \
  /experiment/tune_fp8_tiles.py \
  --model "$MODEL" \
  "${REVISION_ARGS[@]}" \
  --num-speculative-tokens "$NUM_SPEC" \
  --batch-sizes "$BATCH_SIZES" \
  --output-dir /experiment/tuning \
  --header-output /experiment/generated_tactics.hpp

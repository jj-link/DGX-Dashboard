#!/usr/bin/env bash
set -euo pipefail

experiment_dir="$(cd "$(dirname "$0")" && pwd)"
inference_dir="$(cd "$experiment_dir/../.." && pwd)"
image="${IMAGE:-qwen-vllm:mrv2-a0f6d767-fi0614}"
base_image="qwen-vllm:dflash"
vllm_commit="a0f6d767e42acf09f94e7af2bacca7f4c268c264"
flashinfer_version="0.6.14"
fix_root="$inference_dir/patches/vllm"

command -v docker >/dev/null 2>&1 || { echo 'fatal: docker is required' >&2; exit 20; }
docker buildx version >/dev/null 2>&1 || { echo 'fatal: Docker Buildx with named-context support is required' >&2; exit 21; }
docker image inspect "$base_image" >/dev/null 2>&1 || { echo "fatal: required production base image is missing: $base_image" >&2; exit 22; }

required_fixes=(
  model_executor/layers/quantization/modelopt.py
  model_executor/models/qwen3_5.py
  model_executor/parameter.py
  model_executor/layers/vocab_parallel_embedding.py
)
for rel in "${required_fixes[@]}"; do
  [ -f "$fix_root/$rel" ] || { echo "fatal: required host fix is missing: $fix_root/$rel" >&2; exit 23; }
done

exec docker buildx build \
  --load \
  --build-context "modelopt-fixes=$fix_root" \
  --build-arg "BASE_IMAGE=$base_image" \
  --build-arg "VLLM_COMMIT=$vllm_commit" \
  --build-arg "FLASHINFER_VERSION=$flashinfer_version" \
  --tag "$image" \
  --file "$experiment_dir/Dockerfile" \
  "$experiment_dir"

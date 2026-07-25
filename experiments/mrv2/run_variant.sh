#!/usr/bin/env bash
set -euo pipefail

variant="${1:?usage: run_variant.sh standard|block}"
case "$variant" in
  standard) default_port=8001 ;;
  block) default_port=8002 ;;
  *) echo "fatal: variant must be standard or block" >&2; exit 64 ;;
esac

experiment_dir="$(cd "$(dirname "$0")" && pwd)"
image="${IMAGE:-qwen-vllm:mrv2-a0f6d767-fi0614}"
container_name="${CONTAINER_NAME:-vllm-mrv2-dflash-${variant}-a0f6d767}"
port="${PORT:-$default_port}"
hf_cache="${HF_CACHE:-$HOME/.cache/huggingface}"
cache_volume="${CACHE_VOLUME:-vllm-mrv2-compile-cache-${variant}}"
model_revision="${MODEL_REVISION:-0893e1606ff3d5f97a441f405d5fc541a6bdf404}"
drafter_revision="${DRAFTER_REVISION:-0919688658996800f86b895034249700e9481106}"

command -v docker >/dev/null 2>&1 || { echo 'fatal: docker is required' >&2; exit 20; }
docker image inspect "$image" >/dev/null 2>&1 || { echo "fatal: candidate image is missing: $image (run ./build.sh)" >&2; exit 21; }
[ -x "$experiment_dir/serve_${variant}.sh" ] || { echo "fatal: missing serve variant: $experiment_dir/serve_${variant}.sh" >&2; exit 22; }
mkdir -p "$hf_cache"

network_env=(-e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1)
hf_cache_mode=ro
if [ "${ALLOW_ONLINE:-0}" = 1 ]; then
  network_env=(-e HF_HUB_OFFLINE=0 -e TRANSFORMERS_OFFLINE=0)
  hf_cache_mode=rw
else
  target_snapshot="$hf_cache/hub/models--nvidia--Qwen3.6-27B-NVFP4/snapshots/$model_revision"
  draft_snapshot="$hf_cache/hub/models--z-lab--Qwen3.6-27B-DFlash/snapshots/$drafter_revision"
  [ -d "$target_snapshot" ] || { echo "fatal: pinned target snapshot is not cached: $target_snapshot (set ALLOW_ONLINE=1 for an explicit download run)" >&2; exit 23; }
  [ -d "$draft_snapshot" ] || { echo "fatal: pinned DFlash snapshot is not cached: $draft_snapshot (set ALLOW_ONLINE=1 for an explicit download run)" >&2; exit 24; }
fi

run_mode=(-it)
rm_mode=(--rm)
if [ -n "${DETACH:-}" ]; then run_mode=(-d); fi
if [ -n "${KEEP:-}" ]; then rm_mode=(); fi

optional_env=()
for name in GPU_MEM_UTIL MAXLEN NUM_SPEC MAX_NUM_BATCHED MAX_NUM_SEQS; do
  if [ -n "${!name:-}" ]; then
    optional_env+=(-e "$name=${!name}")
  fi
done

exec docker run \
  "${rm_mode[@]}" \
  "${run_mode[@]}" \
  --device nvidia.com/gpu=all \
  --name "$container_name" \
  --security-opt no-new-privileges \
  --cap-drop ALL \
  --pids-limit 4096 \
  -p "127.0.0.1:${port}:${port}" \
  --tmpfs /tmp:rw,nosuid,nodev,size=4g \
  -e HOME=/cache \
  -e HF_HOME=/models \
  -e PORT="$port" \
  -e MODEL_REVISION="$model_revision" \
  -e DRAFTER_REVISION="$drafter_revision" \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  "${network_env[@]}" \
  -v "$cache_volume:/cache" \
  -v "$hf_cache:/models:$hf_cache_mode" \
  -v "$experiment_dir:/experiment:ro" \
  "${optional_env[@]}" \
  --entrypoint /bin/bash \
  "$image" "/experiment/serve_${variant}.sh"

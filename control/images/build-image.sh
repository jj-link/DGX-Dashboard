#!/usr/bin/env bash
set -euo pipefail

fail() { printf 'error: %s
' "$*" >&2; exit 1; }
[[ $# == 1 ]] || fail "usage: images/build-image.sh <target>"
TARGET="$1"
CONTROL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
REPO_ROOT="$(cd "$CONTROL_ROOT/.." && pwd -P)"
command -v git >/dev/null 2>&1 || fail "git is required"
command -v docker >/dev/null 2>&1 || fail "docker is required"
git -C "$REPO_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1 || fail "workspace is not a Git checkout"
[[ -z "$(git -C "$REPO_ROOT" status --porcelain=v1 --untracked-files=all)" ]] || fail "workspace is dirty; refusing image build"

dependencies=()
case "$TARGET" in
  sglang-modelopt-rtx) dockerfile=sglang-modelopt; tag=inference-workspace/sglang-modelopt-rtx:pr30078-20260724 ;;
  sglang-modelopt-spark) dockerfile=sglang-modelopt; tag=inference-workspace/sglang-modelopt-spark:pr30078-20260724 ;;
  sglang-gemma-spark) dockerfile=sglang-gemma; tag=inference-workspace/sglang-gemma-spark:pr23000-20260724 ;;
  sglang-transformers-rtx) dockerfile=sglang-transformers; tag=inference-workspace/sglang-transformers-rtx:v0.5.12-20260724 ;;
  sglang-transformers-spark) dockerfile=sglang-transformers; tag=inference-workspace/sglang-transformers-spark:v0.5.12-20260724 ;;
  sglang-eagle3-rtx) dockerfile=sglang-eagle3; tag=inference-workspace/sglang-eagle3-rtx:v0.5.12-20260724 ;;
  sglang-eagle3-spark) dockerfile=sglang-eagle3; tag=inference-workspace/sglang-eagle3-spark:v0.5.12-20260724 ;;
  vllm-thinkingcap-fp8-rtx) dockerfile=vllm-thinkingcap; tag=inference-workspace/vllm-thinkingcap-fp8-rtx:v0.24.0-20260724; dependencies=(vllm-0.24-align-upstream vllm-0.24-dflash-depth-matrix) ;;
  vllm-thinkingcap-nvfp4-rtx) dockerfile=vllm-thinkingcap; tag=inference-workspace/vllm-thinkingcap-nvfp4-rtx:v0.24.0-20260724; dependencies=(vllm-0.24-align-upstream vllm-0.24-dflash-depth-matrix) ;;
  vllm-hybrid-rtx) dockerfile=vllm-thinkingcap; tag=inference-workspace/vllm-hybrid-rtx:v0.24.0-20260724; dependencies=(vllm-0.24-align-upstream vllm-0.24-dflash-depth-matrix) ;;
  vllm-thinkingcap-spark) dockerfile=vllm-thinkingcap; tag=inference-workspace/vllm-thinkingcap-spark:v0.24.0-20260724; dependencies=(vllm-0.24-align-upstream vllm-0.24-dflash-depth-matrix) ;;
  vllm-dflash-rtx) dockerfile=vllm-dflash; tag=inference-workspace/vllm-dflash-rtx:v0.21.0-20260724 ;;
  vllm-spark) dockerfile=vllm-spark; tag=inference-workspace/vllm-spark:v0.21.0-20260724 ;;
  vllm-poolside-spark) dockerfile=vllm-poolside-spark; tag=inference-workspace/vllm-poolside-spark:v0.25.2-flashinfer-20260712 ;;
  *) fail "unknown image target '$TARGET'" ;;
esac
if ((${#dependencies[@]})); then
  "$CONTROL_ROOT/bootstrap-dependencies.sh" "${dependencies[@]}"
fi
exec docker build --pull=false --tag "$tag" --file "$CONTROL_ROOT/images/$dockerfile/Dockerfile" "$CONTROL_ROOT"

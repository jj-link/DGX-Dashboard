#!/usr/bin/env bash
set -euo pipefail

readonly ROOT="/home/workbench/inference/experiments/unsloth-qwen36-nvfp4"
readonly VLLM_IMAGE="vllm/vllm-openai@sha256:251eba5cc7c12fed0b75da22a9240e582b1c9e39f6fbc064f86781b963bd814f"
readonly SGLANG_IMAGE="sha256:8c50d8465642f41fa43ac96dec1449c3ac4f377888d6f3dd6f0635d8dfcb9924"
readonly MODEL_27B="/mnt/c/Users/josep/Models/unsloth-qwen36/Qwen3.6-27B-NVFP4-9c295353"
readonly MODEL_35B="/mnt/c/Users/josep/Models/unsloth-qwen36/Qwen3.6-35B-A3B-NVFP4-Fast-11fb1aff"
readonly HF_CACHE="/home/workbench/.cache/huggingface"

usage() {
  cat <<'EOF'
Print pinned launch commands; this script never executes them.

Usage:
  ./reproduce.sh list
  ./reproduce.sh show ENGINE MODEL CONFIGURATION
  ./reproduce.sh winner ENGINE MODEL

The catalog exposes baseline, MTP, and DFlash launch modes. Winner commands are
the selected concurrency-1 configurations on the pinned final images.
EOF
}

command_for() {
  local engine="$1"
  local model="$2"
  local configuration="$3"

  case "${engine}:${model}:${configuration}" in
    vllm:27b:baseline)
      printf 'IMAGE=%q MODEL_HOST=%q HF_CACHE=%q DETACH=1 PORT=8000 CONFIG=baseline CONTAINER_NAME=unsloth-qwen36-vllm-27b-baseline %q\n' "${VLLM_IMAGE}" "${MODEL_27B}" "${HF_CACHE}" "${ROOT}/vllm/serve-27b.sh"
      ;;
    vllm:27b:mtp)
      printf 'IMAGE=%q MODEL_HOST=%q HF_CACHE=%q DETACH=1 PORT=8000 CONFIG=mtp NUM_SPEC=4 MAMBA_CACHE_DTYPE=bfloat16 MAMBA_SSM_CACHE_DTYPE=bfloat16 CONTAINER_NAME=unsloth-qwen36-vllm-27b-mtp4-statebf16 %q\n' "${VLLM_IMAGE}" "${MODEL_27B}" "${HF_CACHE}" "${ROOT}/vllm/serve-27b.sh"
      ;;
    vllm:27b:dflash)
      printf 'IMAGE=%q MODEL_HOST=%q HF_CACHE=%q DETACH=1 PORT=8001 CONFIG=dflash NUM_SPEC=12 CONTAINER_NAME=unsloth-qwen36-vllm-27b-dflash12 %q\n' "${VLLM_IMAGE}" "${MODEL_27B}" "${HF_CACHE}" "${ROOT}/vllm/serve-27b.sh"
      ;;
    vllm:35b:baseline)
      printf 'IMAGE=%q MODEL_HOST=%q HF_CACHE=%q DETACH=1 PORT=8000 CONFIG=baseline CONTAINER_NAME=unsloth-qwen36-vllm-35b-a3b-baseline %q\n' "${VLLM_IMAGE}" "${MODEL_35B}" "${HF_CACHE}" "${ROOT}/vllm/serve-35b-a3b.sh"
      ;;
    vllm:35b:mtp)
      printf 'IMAGE=%q MODEL_HOST=%q HF_CACHE=%q DETACH=1 PORT=8000 CONFIG=mtp NUM_SPEC=4 MAMBA_CACHE_DTYPE=bfloat16 MAMBA_SSM_CACHE_DTYPE=bfloat16 CONTAINER_NAME=unsloth-qwen36-vllm-35b-a3b-mtp4-statebf16 %q\n' "${VLLM_IMAGE}" "${MODEL_35B}" "${HF_CACHE}" "${ROOT}/vllm/serve-35b-a3b.sh"
      ;;
    vllm:35b:dflash)
      printf 'IMAGE=%q MODEL_HOST=%q HF_CACHE=%q DETACH=1 PORT=8000 CONFIG=dflash NUM_SPEC=12 CONTAINER_NAME=unsloth-qwen36-vllm-35b-a3b-dflash12 %q\n' "${VLLM_IMAGE}" "${MODEL_35B}" "${HF_CACHE}" "${ROOT}/vllm/serve-35b-a3b.sh"
      ;;
    sglang:27b:baseline)
      printf 'IMAGE=%q MODEL_HOST=%q HF_CACHE=%q DETACH=1 PORT=8000 CONFIG=baseline LINEAR_ATTN_PREFILL_BACKEND=triton LINEAR_ATTN_DECODE_BACKEND=flashinfer MAMBA_SSM_DTYPE=bfloat16 CONTAINER_NAME=unsloth-qwen36-sglang-27b-baseline %q\n' "${SGLANG_IMAGE}" "${MODEL_27B}" "${HF_CACHE}" "${ROOT}/sglang/serve-27b.sh"
      ;;
    sglang:27b:mtp)
      printf 'IMAGE=%q MODEL_HOST=%q HF_CACHE=%q DETACH=1 PORT=8000 CONFIG=mtp SPEC_STEPS=1 NUM_SPEC=2 LINEAR_ATTN_PREFILL_BACKEND=triton LINEAR_ATTN_DECODE_BACKEND=flashinfer MAMBA_SSM_DTYPE=bfloat16 CONTAINER_NAME=unsloth-qwen36-sglang-27b-mtp-s1-d2 %q\n' "${SGLANG_IMAGE}" "${MODEL_27B}" "${HF_CACHE}" "${ROOT}/sglang/serve-27b.sh"
      ;;
    sglang:27b:dflash)
      printf 'IMAGE=%q MODEL_HOST=%q HF_CACHE=%q DETACH=1 PORT=8000 CONFIG=dflash NUM_SPEC=8 DRAFT_WINDOW=2048 LINEAR_ATTN_PREFILL_BACKEND=triton LINEAR_ATTN_DECODE_BACKEND=flashinfer MAMBA_SSM_DTYPE=bfloat16 CONTAINER_NAME=unsloth-qwen36-sglang-27b-dflash8-final %q\n' "${SGLANG_IMAGE}" "${MODEL_27B}" "${HF_CACHE}" "${ROOT}/sglang/serve-27b.sh"
      ;;
    sglang:35b:baseline)
      printf 'IMAGE=%q MODEL_HOST=%q HF_CACHE=%q DETACH=1 PORT=8000 CONFIG=baseline LINEAR_ATTN_PREFILL_BACKEND=triton LINEAR_ATTN_DECODE_BACKEND=flashinfer MAMBA_SSM_DTYPE=bfloat16 CONTAINER_NAME=unsloth-qwen36-sglang-35b-a3b-baseline %q\n' "${SGLANG_IMAGE}" "${MODEL_35B}" "${HF_CACHE}" "${ROOT}/sglang/serve-35b-a3b.sh"
      ;;
    sglang:35b:mtp)
      printf 'IMAGE=%q MODEL_HOST=%q HF_CACHE=%q DETACH=1 PORT=8000 CONFIG=mtp SPEC_STEPS=4 NUM_SPEC=5 LINEAR_ATTN_PREFILL_BACKEND=triton LINEAR_ATTN_DECODE_BACKEND=flashinfer MAMBA_SSM_DTYPE=bfloat16 CONTAINER_NAME=unsloth-qwen36-sglang-35b-a3b-mtp-s4-d5 %q\n' "${SGLANG_IMAGE}" "${MODEL_35B}" "${HF_CACHE}" "${ROOT}/sglang/serve-35b-a3b.sh"
      ;;
    sglang:35b:dflash)
      printf 'IMAGE=%q MODEL_HOST=%q HF_CACHE=%q DETACH=1 PORT=8000 CONFIG=dflash NUM_SPEC=8 DRAFT_WINDOW=6144 LINEAR_ATTN_PREFILL_BACKEND=triton LINEAR_ATTN_DECODE_BACKEND=flashinfer MAMBA_SSM_DTYPE=bfloat16 CONTAINER_NAME=unsloth-qwen36-sglang-35b-a3b-dflash8-final %q\n' "${SGLANG_IMAGE}" "${MODEL_35B}" "${HF_CACHE}" "${ROOT}/sglang/serve-35b-a3b.sh"
      ;;
    *)
      echo "unsupported selection: ${engine} ${model} ${configuration}" >&2
      return 2
      ;;
  esac
}

winner_for() {
  case "$1:$2" in
    vllm:27b) command_for vllm 27b mtp ;;
    vllm:35b) command_for vllm 35b mtp ;;
    sglang:27b) command_for sglang 27b dflash ;;
    sglang:35b) command_for sglang 35b dflash ;;
    *) echo "unsupported winner selection: $1 $2" >&2; return 2 ;;
  esac
}

list_all() {
  local engine model configuration
  for engine in vllm sglang; do
    for model in 27b 35b; do
      for configuration in baseline mtp dflash; do
        printf '%s %s %s:\n' "${engine}" "${model}" "${configuration}"
        command_for "${engine}" "${model}" "${configuration}"
      done
    done
  done
  printf '%s\n' 'selected concurrency-1 winners:'
  for engine in vllm sglang; do
    for model in 27b 35b; do
      printf '%s %s winner:\n' "${engine}" "${model}"
      winner_for "${engine}" "${model}"
    done
  done
}

case "${1:-list}" in
  list)
    [[ "$#" -le 1 ]] || { usage >&2; exit 2; }
    list_all
    ;;
  show)
    [[ "$#" -eq 4 ]] || { usage >&2; exit 2; }
    command_for "$2" "$3" "$4"
    ;;
  winner)
    [[ "$#" -eq 3 ]] || { usage >&2; exit 2; }
    winner_for "$2" "$3"
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

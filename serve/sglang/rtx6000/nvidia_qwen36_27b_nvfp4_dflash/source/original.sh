#!/usr/bin/env bash
# SGLang launch for nvidia/Qwen3.6-27B-NVFP4 target + z-lab/Qwen3.6-27B-DFlash.
# Applies the local DFlash ModelOpt LM-head patch before server start.
set -u

MODEL="${MODEL:-nvidia/Qwen3.6-27B-NVFP4}"
DRAFTER="${DRAFTER:-z-lab/Qwen3.6-27B-DFlash}"
SERVED="${SERVED:-Qwen3.6-27B-NVFP4-DFlash}"
MAXLEN="${MAXLEN:-8192}"
NUM_SPEC="${NUM_SPEC:-16}"
DRAFT_WINDOW="${DRAFT_WINDOW:-2048}"
MEM_FRACTION="${MEM_FRACTION:-0.45}"
ATTN_BACKEND="${ATTN_BACKEND:-flashinfer}"
CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-8}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-1}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-2048}"
MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-4096}"
PORT="${PORT:-8000}"

DWIN_ARG=()
case "${DRAFT_WINDOW}" in off|0|none) ;; *) DWIN_ARG=(--speculative-draft-window-size "${DRAFT_WINDOW}") ;; esac

CACHE_ARG=()
[ "${RADIX:-0}" = "1" ] || CACHE_ARG=(--disable-radix-cache)

CUDA_GRAPH_ARG=()
[ "${DISABLE_CUDA_GRAPH:-1}" = "1" ] && CUDA_GRAPH_ARG=(--disable-cuda-graph)

PARSER_ARG=(--reasoning-parser qwen3 --tool-call-parser qwen3_coder)
[ "${PARSERS:-1}" = "1" ] || PARSER_ARG=()

export FLASHINFER_DISABLE_VERSION_CHECK="${FLASHINFER_DISABLE_VERSION_CHECK:-1}"
export SGLANG_ENABLE_JIT_DEEPGEMM="${SGLANG_ENABLE_JIT_DEEPGEMM:-0}"

python3 /workspace/serve/sglang/patch_sglang_dflash_modelopt_lmhead.py

exec python3 -m sglang.launch_server \
  --model-path "$MODEL" \
  --served-model-name "$SERVED" \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path "$DRAFTER" \
  --speculative-num-draft-tokens "$NUM_SPEC" \
  "${DWIN_ARG[@]}" \
  --tp-size 1 \
  --attention-backend "$ATTN_BACKEND" \
  --mem-fraction-static "$MEM_FRACTION" \
  --cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS" \
  --max-running-requests "$MAX_RUNNING_REQUESTS" \
  --chunked-prefill-size "$CHUNKED_PREFILL_SIZE" \
  --max-prefill-tokens "$MAX_PREFILL_TOKENS" \
  --context-length "$MAXLEN" \
  "${CUDA_GRAPH_ARG[@]}" \
  "${CACHE_ARG[@]}" \
  --trust-remote-code \
  --enable-metrics \
  --host 0.0.0.0 --port "$PORT" \
  "${PARSER_ARG[@]}"

#!/usr/bin/env bash
# SGLang launch for nvidia/Qwen3.6-27B-NVFP4 on RTX PRO 6000 Blackwell (WSL2).
# Requires the patched image built from SGLang PR #30078:
#   docker build -t sglang:modelopt-pr30078 -f Dockerfile.sglang-modelopt-pr30078 .
# Run via:
#   IMAGE=sglang:modelopt-pr30078 ./serve.sh sglang nvidia_qwen36_27b_nvfp4
set -euo pipefail

MODEL="${MODEL_PATH:?MODEL_PATH is required}"
SERVED="${SERVED:?SERVED is required}"
MAXLEN="${MAXLEN:-262144}"
MEM_FRACTION="${MEM_FRACTION:-0.85}"
ATTN_BACKEND="${ATTN_BACKEND:-flashinfer}"
CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-8}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-4}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-2048}"
MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-32768}"
PORT=8000

PARSER_ARG=(--reasoning-parser qwen3 --tool-call-parser qwen3_coder)
[ "${PARSERS:-1}" = "1" ] || PARSER_ARG=()

# Keep the first working target plain/non-speculative. Prior DFlash pairing with
# this ModelOpt-mixed NVFP4 checkpoint hit a shape mismatch in SGLang's DFlash
# worker, so benchmark the base server before any speculative experiment.
CACHE_ARG=()
[ "${RADIX:-1}" = "1" ] || CACHE_ARG=(--disable-radix-cache)

CUDA_GRAPH_ARG=()
# On local SM120, CUDA graphs have previously produced corrupted output on some
# SGLang configs. Default to disabled for correctness; set DISABLE_CUDA_GRAPH=0
# later if we want a measured speed run with graphs enabled.
[ "${DISABLE_CUDA_GRAPH:-1}" = "1" ] && CUDA_GRAPH_ARG=(--disable-cuda-graph)

export SGLANG_ENABLE_JIT_DEEPGEMM="${SGLANG_ENABLE_JIT_DEEPGEMM:-0}"
export FLASHINFER_DISABLE_VERSION_CHECK="${FLASHINFER_DISABLE_VERSION_CHECK:-1}"

exec python3 -m sglang.launch_server \
  --model-path "$MODEL" \
  --served-model-name "$SERVED" \
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
  "${PARSER_ARG[@]}" \
  "$@"

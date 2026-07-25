#!/usr/bin/env bash
# SGLang launch for Qwen3.6-27B FP8 target + EAGLE-3 drafter on RTX PRO 6000
# Blackwell (SM120, WSL2). Run via the hardened profile runtime:
#   ./serve.sh sglang qwen36_27b_fp8_eagle3
#
# Matches the `qwen36_27b_fp8_dflash` package except for the speculative
# decoding algorithm.
#
# Drafter: Ex0bit/Qwen3.6-27B-PRISM-EAGLE3/compressed (~1.1GB, 32K vocab).
# The package metadata resolves its local read-only mount.
#
# The pinned package image applies the Qwen3.6 EAGLE-3 capture patch at build
# time, so the read-only runtime never mutates site-packages.
set -euo pipefail

MODEL="${MODEL_PATH:?MODEL_PATH is required}"
DRAFTER="${DRAFTER_PATH:?DRAFTER_PATH is required}"
SERVED="${SERVED:?SERVED is required}"
MAXLEN="${MAXLEN:-262144}"

# EAGLE-3 knobs. Defaults are the values from Ex0bit's validated fast config.
# Chain (topk=1) is recommended; tree (topk>1) raises accept length but costs
# tree-build overhead on the GatedDeltaNet hybrid target.
NUM_SPEC="${NUM_SPEC:-4}"   # --speculative-num-draft-tokens
TOPK="${TOPK:-1}"           # --speculative-eagle-topk
STEPS="${STEPS:-3}"         # --speculative-num-steps

MEM_FRACTION="${MEM_FRACTION:-0.82}"
ATTN_BACKEND="${ATTN_BACKEND:-flashinfer}"
CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-8}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-2}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-16384}"
MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-32768}"
PORT=8000

PARSER_ARG=(--reasoning-parser qwen3 --tool-call-parser qwen3_coder)
[ "${PARSERS:-1}" = "1" ] || PARSER_ARG=()

if [ "${RADIX:-1}" = "1" ]; then
  CACHE_ARG=()
  MAMBA_ARG=(--mamba-scheduler-strategy "${MAMBA_STRATEGY:-extra_buffer}")
  export SGLANG_ENABLE_SPEC_V2="${SGLANG_ENABLE_SPEC_V2:-1}"
else
  CACHE_ARG=(--disable-radix-cache)
  MAMBA_ARG=(--mamba-scheduler-strategy "${MAMBA_STRATEGY:-auto}")
fi


exec python3 -m sglang.launch_server \
  --model-path "$MODEL" \
  --served-model-name "$SERVED" \
  --speculative-algorithm EAGLE3 \
  --speculative-draft-model-path "$DRAFTER" \
  --speculative-num-steps "$STEPS" \
  --speculative-eagle-topk "$TOPK" \
  --speculative-num-draft-tokens "$NUM_SPEC" \
  --tp-size 1 \
  --attention-backend "$ATTN_BACKEND" \
  --mem-fraction-static "$MEM_FRACTION" \
  --cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS" \
  --max-running-requests "$MAX_RUNNING_REQUESTS" \
  --chunked-prefill-size "$CHUNKED_PREFILL_SIZE" \
  --max-prefill-tokens "$MAX_PREFILL_TOKENS" \
  --context-length "$MAXLEN" \
  "${MAMBA_ARG[@]}" \
  "${CACHE_ARG[@]}" \
  --trust-remote-code \
  --enable-metrics \
  --host 0.0.0.0 --port "$PORT" \
  "${PARSER_ARG[@]}" \
  "$@"

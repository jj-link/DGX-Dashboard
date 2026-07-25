#!/usr/bin/env bash
# vLLM launch for Qwen3.6-35B-A3B FP8 target + DFlash speculative drafter
# (z-lab block-diffusion 0.5B drafter) on RTX PRO 6000 Blackwell (WSL2).
# Requires the qwen-vllm:dflash image (vLLM PR #40898).
set -u

# Target = the FP8 model you actually run (spec-decode validates against it).
# The DFlash card nominally pairs with the bf16 base; FP8 target still
# produces correct output, acceptance length may differ slightly.
MODEL="${MODEL:-Qwen/Qwen3.6-35B-A3B-FP8}"
DRAFTER="${DRAFTER:-z-lab/Qwen3.6-35B-A3B-DFlash}"
SERVED="${SERVED:-Qwen3.6-35B-A3B}"
MAXLEN="${MAXLEN:-262144}"
NUM_SPEC="${NUM_SPEC:-15}"         # 8-16 per DFlash benchmarks; sweep this
# DFlash mandates the flash_attn backend, which does NOT support fp8 KV
# cache in this vLLM build -> default to bf16 KV (auto). fp8 would crash
# at engine init ("kv_cache_dtype not supported").
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
PORT="${PORT:-8000}"

# WSL2+Blackwell: cuMemSetAccess (VMM API) unsupported.
unset PYTORCH_CUDA_ALLOC_CONF
unset VLLM_BUILD_URL VLLM_IMAGE_TAG VLLM_BUILD_PIPELINE VLLM_BUILD_COMMIT
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

SPEC_CFG="{\"method\": \"dflash\", \"model\": \"${DRAFTER}\", \"num_speculative_tokens\": ${NUM_SPEC}}"

# Optional long-context drafter window (DFlash-specific).
DWIN_ARG=()
[ -n "${DFLASH_DRAFT_WINDOW:-}" ] && DWIN_ARG=(--speculative-dflash-draft-window-size "${DFLASH_DRAFT_WINDOW}")

# Prefix caching is ON by default for agentic/benchmark runs with shared prompt
# structure. Set PREFIX_CACHING=0 for throughput sweeps where prefix reuse is
# deliberately unwanted.
CACHE_ARG=()
[ "${PREFIX_CACHING:-1}" = "1" ] && CACHE_ARG+=(--enable-prefix-caching) || CACHE_ARG+=(--no-enable-prefix-caching)

PARSER_ARG=(--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder)
[ "${PARSERS:-1}" = "1" ] || PARSER_ARG=()

exec vllm serve "$MODEL" \
  --speculative-config "$SPEC_CFG" \
  "${DWIN_ARG[@]}" \
  --attention-backend flash_attn \
  --max-num-batched-tokens "${MAX_NUM_BATCHED:-32768}" \
  "${CACHE_ARG[@]}" \
  --served-model-name "$SERVED" \
  --host 0.0.0.0 --port "$PORT" \
  --max-model-len "$MAXLEN" \
  --gpu-memory-utilization 0.92 \
  --kv-cache-dtype "$KV_CACHE_DTYPE" \
  --disable-custom-all-reduce \
  --limit-mm-per-prompt '{"image": 0, "video": 0}' \
  --generation-config vllm \
  "${PARSER_ARG[@]}"

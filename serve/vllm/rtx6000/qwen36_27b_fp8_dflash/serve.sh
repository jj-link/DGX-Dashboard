#!/usr/bin/env bash
# vLLM launch for Qwen3.6-27B FP8 target + DFlash speculative drafter
# (z-lab block-diffusion drafter) on RTX PRO 6000 Blackwell (WSL2).
# Requires the qwen-vllm:dflash image (vLLM PR #40898).
#
# Optimal config from the 2026-05-22 DFlash sweep (single-stream coding,
# temp 0.6, 3 problems x 3 reps; see benchmarks/results/DFLASH_OPTIMAL.md):
#   fp8  target -> NUM_SPEC=12  (~152 tok/s, flat plateau 12-28; 2.8x vs no-DFlash)
#   bf16 target -> NUM_SPEC=20  (run: MODEL=Qwen/Qwen3.6-27B NUM_SPEC=20 ; ~111 tok/s)
set -euo pipefail

# Target = the model spec-decode validates against (output quality == target's).
MODEL="${MODEL_PATH:?MODEL_PATH is required}"
DRAFTER="${DRAFTER_PATH:?DRAFTER_PATH is required}"
SERVED="${SERVED:?SERVED is required}"
MAXLEN="${MAXLEN:-262144}"
NUM_SPEC="${NUM_SPEC:-12}"          # optimal for fp8 coding @ temp 0.6 (plateau 12-28)
# SPEC_MAXLEN: if set, adds the draft model's max_model_len to the spec config so
# vLLM's input_fits_in_drafter gate skips the drafter (-> plain decode) once a
# sequence exceeds it (i.e. auto-fallback to the plain model past N tokens).
SPEC_MAXLEN="${SPEC_MAXLEN:-}"
# DFlash mandates the flash_attn backend, which does NOT support fp8 KV cache
# in this vLLM build -> bf16 KV (auto). fp8 KV crashes at engine init.
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
PORT=8000

# SAMPLING (set by the CLIENT per request; this is what coding agents send).
# Qwen3.6-27B precise-coding recommendation:
# temperature=0.6, top_p=0.95, top_k=20, min_p=0.0,
# presence_penalty=0.0, repetition_penalty=1.0.
# We deliberately do NOT pass --generation-config vllm, so the model's
# Do NOT use greedy (temp 0): it triggers repetition loops on this reasoning model.

# WSL2+Blackwell: cuMemSetAccess (VMM API) unsupported.
unset PYTORCH_CUDA_ALLOC_CONF
unset VLLM_BUILD_URL VLLM_IMAGE_TAG VLLM_BUILD_PIPELINE VLLM_BUILD_COMMIT
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

if [ -n "$SPEC_MAXLEN" ]; then
  SPEC_CFG="{\"method\": \"dflash\", \"model\": \"${DRAFTER}\", \"num_speculative_tokens\": ${NUM_SPEC}, \"max_model_len\": ${SPEC_MAXLEN}}"
else
  SPEC_CFG="{\"method\": \"dflash\", \"model\": \"${DRAFTER}\", \"num_speculative_tokens\": ${NUM_SPEC}}"
fi

# Prefix caching ON by default â€” essential for interactive/agentic sessions: it
# reuses the growing conversation prefix's KV across turns. Without it, every turn
# re-prefills the ENTIRE accumulated history from scratch, so responses get slower
# and slower as the session grows (the "fine at first, crawls after a few minutes"
# failure). Set PREFIX_CACHING=0 to disable.
CACHE_ARG=()
[ "${PREFIX_CACHING:-1}" = "1" ] && CACHE_ARG+=(--enable-prefix-caching) || CACHE_ARG+=(--no-enable-prefix-caching)

PARSER_ARG=(--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder)
[ "${PARSERS:-1}" = "1" ] || PARSER_ARG=()

exec vllm serve "$MODEL" \
  --speculative-config "$SPEC_CFG" \
  --attention-backend flash_attn \
  --max-num-batched-tokens "${MAX_NUM_BATCHED:-32768}" \
  "${CACHE_ARG[@]}" \
  --served-model-name "$SERVED" \
  --host 0.0.0.0 --port "$PORT" \
  --max-model-len "$MAXLEN" \
  --gpu-memory-utilization "${GPU_MEM_UTIL:-0.83}" \
  --kv-cache-dtype "$KV_CACHE_DTYPE" \
  --disable-custom-all-reduce \
  --limit-mm-per-prompt '{"image": 0, "video": 0}' \
  "${PARSER_ARG[@]}" \
  "$@"

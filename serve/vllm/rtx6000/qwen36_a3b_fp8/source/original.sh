#!/usr/bin/env bash
# vLLM launch for Qwen3.6-35B-A3B FP8 (MoE, 3B active) on RTX PRO 6000 Blackwell (WSL2).
set -u
MODEL="${MODEL:-Qwen/Qwen3.6-35B-A3B-FP8}"
SERVED="${SERVED:-Qwen3.6-35B-A3B}"
MAXLEN="${MAXLEN:-262144}"   # Model tokenizer advertises 256K context (256*1024)
PORT="${PORT:-8000}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"

# WSL2+Blackwell: cuMemSetAccess (VMM API) is unsupported — expandable_segments crashes.
unset PYTORCH_CUDA_ALLOC_CONF
# These image-build metadata variables are useful outside vLLM, but vLLM warns
# on unknown VLLM_* env vars during startup. Drop them from the server process.
unset VLLM_BUILD_URL VLLM_IMAGE_TAG VLLM_BUILD_PIPELINE VLLM_BUILD_COMMIT
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# FP8 MoE backend. SM120 has native FP8 tensor cores, so backend choice
# matters. Default = vLLM auto-select. Override with MOE_BACKEND=<name>
# (deepgemm, triton, flashinfer_cutlass, cutlass).
MOE_ARG=()
[ -n "${MOE_BACKEND:-}" ] && MOE_ARG=(--moe-backend "${MOE_BACKEND}")

# Prefix caching is ON by default for agentic/benchmark runs with shared prompt
# structure. Set PREFIX_CACHING=0 for single-stream throughput sweeps where
# prefix reuse is deliberately unwanted.
CACHE_ARG=()
[ "${PREFIX_CACHING:-1}" = "1" ] && CACHE_ARG+=(--enable-prefix-caching) || CACHE_ARG+=(--no-enable-prefix-caching)
[ "${CHUNKED_PREFILL:-1}" = "1" ] && CACHE_ARG+=(--enable-chunked-prefill) || CACHE_ARG+=(--no-enable-chunked-prefill)

# Leave KV cache dtype at vLLM/model default for benchmark quality. Override
# with KV_CACHE_DTYPE=fp8 only for capacity/throughput sweeps.
KV_CACHE_ARG=()
[ -n "${KV_CACHE_DTYPE:-}" ] && KV_CACHE_ARG=(--kv-cache-dtype "${KV_CACHE_DTYPE}")

# SSM-state cache precision. Model config intends float32; A/B (2026-05-17 on
# 27B-FP8) showed fp32 = +1/34 polyglot, zero regressions, ~3% slower, no OOM
# at 128K. Default native fp32; MAMBA_DTYPE=float16 for max-throughput sweeps.
MAMBA_DTYPE="${MAMBA_DTYPE:-float32}"
MAMBA_ARG=(--mamba-ssm-cache-dtype "${MAMBA_DTYPE}" --mamba-cache-dtype float16)

# INDUCTOR_MAX_AUTOTUNE=1 -> torch.compile autotunes compute kernels
# (dequant, Mamba, elementwise). Not spec-decode/MTP; just kernel tuning.
COMPILE_ARG=()
[ "${INDUCTOR_MAX_AUTOTUNE:-0}" = "1" ] && COMPILE_ARG=(--compilation-config '{"inductor_compile_config":{"max_autotune":true}}')

# Reasoning/tool parsers do per-token incremental parsing in the stream
# path. PARSERS=0 drops them (raw streaming) to measure/remove that
# per-token overhead. Not spec-decode/MTP; an API-feature toggle.
if [ "${PARSERS:-1}" = "1" ]; then
  PARSER_ARG=(--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder)
else
  PARSER_ARG=()
fi

exec vllm serve "$MODEL" \
  "${MOE_ARG[@]}" \
  "${CACHE_ARG[@]}" \
  --served-model-name "$SERVED" \
  --host 0.0.0.0 --port "$PORT" \
  --max-model-len "$MAXLEN" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --gpu-memory-utilization 0.92 \
  "${KV_CACHE_ARG[@]}" \
  "${MAMBA_ARG[@]}" \
  --disable-custom-all-reduce \
  --limit-mm-per-prompt '{"image": 0, "video": 0}' \
  --generation-config vllm \
  "${PARSER_ARG[@]}" \
  ${FI_AUTOTUNE_ARG:---enable-flashinfer-autotune} \
  "${COMPILE_ARG[@]}"
  # FI_AUTOTUNE_ARG=" " (a space) to drop --enable-flashinfer-autotune
  # MoE backend sweep: re-run with each to compare tok/s
  # --moe-backend flashinfer_cutlass   # alternatives: triton, deepgemm

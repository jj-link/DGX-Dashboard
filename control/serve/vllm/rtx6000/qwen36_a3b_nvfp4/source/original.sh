#!/usr/bin/env bash
# vLLM launch for Qwen3.6-35B-A3B NVFP4 (MoE, 3B active) on RTX PRO 6000 Blackwell (WSL2).
# NOTE: RedHatAI repo ships tokenizer_config.json with tokenizer_class: TokenizersBackend
# (not a transformers class) — override tokenizer to the official FP8 repo.
set -u
MODEL="${MODEL:-RedHatAI/Qwen3.6-35B-A3B-NVFP4}"
TOKENIZER="${TOKENIZER:-Qwen/Qwen3.6-35B-A3B-FP8}"
SERVED="${SERVED:-Qwen3.6-35B-A3B}"
MAXLEN="${MAXLEN:-262144}"
PORT="${PORT:-8000}"

# WSL2+Blackwell: cuMemSetAccess (VMM API) is unsupported — expandable_segments crashes.
unset PYTORCH_CUDA_ALLOC_CONF
unset VLLM_BUILD_URL VLLM_IMAGE_TAG VLLM_BUILD_PIPELINE VLLM_BUILD_COMMIT
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# NvFp4 MoE backend. Defaults to marlin: on SM120 (RTX PRO 6000) marlin
# benchmarks ~14% faster than the auto-selected FLASHINFER_CUTLASS
# (184 -> 210 tok/s), because the CUTLASS FP4 fast-path is broken on SM120
# and TRTLLM/CuteDSL are SM100-only. Override with MOE_BACKEND=<name>.
MOE_BACKEND="${MOE_BACKEND:-marlin}"
MOE_ARG=(--moe-backend "${MOE_BACKEND}")

# Leave KV cache dtype at vLLM/model default for benchmark quality. Override
# with KV_CACHE_DTYPE=fp8 only for capacity/throughput sweeps.
KV_CACHE_ARG=()
[ -n "${KV_CACHE_DTYPE:-}" ] && KV_CACHE_ARG=(--kv-cache-dtype "${KV_CACHE_DTYPE}")

# SSM-state cache precision. Model config intends float32; A/B (2026-05-17 on
# 27B-FP8) showed fp32 = +1/34 polyglot, zero regressions, ~3% slower, no OOM
# at 128K. Default native fp32; MAMBA_DTYPE=float16 for max-throughput sweeps.
MAMBA_DTYPE="${MAMBA_DTYPE:-float32}"

exec vllm serve "$MODEL" \
  "${MOE_ARG[@]}" \
  --tokenizer "$TOKENIZER" \
  --served-model-name "$SERVED" \
  --host 0.0.0.0 --port "$PORT" \
  --max-model-len "$MAXLEN" \
  --gpu-memory-utilization 0.92 \
  "${KV_CACHE_ARG[@]}" \
  --mamba-ssm-cache-dtype "$MAMBA_DTYPE" \
  --mamba-cache-dtype float16 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --disable-custom-all-reduce \
  --limit-mm-per-prompt '{"image": 0, "video": 0}' \
  --generation-config vllm \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --enable-flashinfer-autotune
  # --moe-backend flashinfer_trtllm  # alternatives: flashinfer_cutlass, flashinfer_cutedsl

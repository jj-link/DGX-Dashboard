#!/usr/bin/env bash
# vLLM launch for Qwen3.5-122B-A10B NVFP4 (MoE, ~10B active) on RTX PRO 6000
# Blackwell SM120 (WSL2). ~72GB NVFP4 weights â€” needs the full 96GB GPU
# (cannot co-exist with the FP8 agent; stop that container first).
#
# NOTE: RedHatAI repo ships tokenizer_config.json tokenizer_class=
# TokenizersBackend and no clean sibling tokenizer with the right vocabulary.
# The corrected tokenizer is a separately resolved read-only local artifact.
set -euo pipefail
MODEL="${MODEL_PATH:?MODEL_PATH is required}"
TOKENIZER="${TOKENIZER_PATH:?TOKENIZER_PATH is required}"
SERVED="${SERVED:?SERVED is required}"
# OPTIMAL DEFAULT (measured): cudagraphs need VRAM for capture, so 72GB
# weights cap usable context. 8192 fits capture with headroom; raising it
# may require EAGER=1 or lower GPU_MEM_UTIL. (eager baseline 14.9 ->
# cudagraph 103.5 tok/s, ~6.9x; capture cost 2.11 GiB.)
MAXLEN="${MAXLEN:-131072}"
PORT=8000

# WSL2+Blackwell: cuMemSetAccess (VMM API) unsupported â€” expandable_segments crashes.
unset PYTORCH_CUDA_ALLOC_CONF
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# flashinfer attention workspace: default ~413MB overflows for this 122B
# (needs ~536MB at cudagraph plan). 768MB has headroom.
export VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE="${VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE:-805306368}"

# SM120 NVFP4: cutlass FP4 fast-path is broken, TRTLLM/CuteDSL are SM100-only
# -> marlin is the working MoE backend (same finding as the 35B NVFP4 script).
MOE_BACKEND="${MOE_BACKEND:-marlin}"

# OPTIMAL DEFAULT: cudagraphs ON (EAGER=0) â€” 6.9x over --enforce-eager.
# Set EAGER=1 only if cudagraph capture OOMs (e.g. raising MAXLEN).
EAGER_ARG=()
[ "${EAGER:-0}" = "1" ] && EAGER_ARG=(--enforce-eager)

# Leave KV cache dtype at vLLM/model default for benchmark quality. Override
# with KV_CACHE_DTYPE=fp8 only for capacity/throughput sweeps.
KV_CACHE_ARG=()
[ -n "${KV_CACHE_DTYPE:-}" ] && KV_CACHE_ARG=(--kv-cache-dtype "${KV_CACHE_DTYPE}")

# Hybrid Mamba model: each decode seq needs a Mamba cache block. The 72GB
# weights leave room for only a few hundred blocks, and cudagraph capture
# requires max_num_seqs <= that. 256 is ample for batch-1 decode.
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"

exec vllm serve "$MODEL" \
  --moe-backend "$MOE_BACKEND" \
  --tokenizer "$TOKENIZER" \
  --served-model-name "$SERVED" \
  --host 0.0.0.0 --port "$PORT" \
  --max-model-len "$MAXLEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --gpu-memory-utilization "${GPU_MEM_UTIL:-0.92}" \
  "${KV_CACHE_ARG[@]}" \
  "${EAGER_ARG[@]}" \
  --trust-remote-code \
  --disable-custom-all-reduce \
  --limit-mm-per-prompt '{"image": 0, "video": 0}' \
  --generation-config vllm \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  "$@"

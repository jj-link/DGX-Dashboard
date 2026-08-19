#!/usr/bin/env bash
set -euo pipefail

# Qwen3.8-27B-Uncensored-FP8 on SGLang (RTX PRO 6000, 96GB, SM120/Blackwell)
#
# Unabliterated/uncensored block-FP8 (e4m3, wb128) derivative of Qwen3.8-27B,
# served with the DSpark speculative decoder against the trained BF16 drafter
# RadixArk/Qwen3.8-27B-DSpark (same recipe as the MiaAI RTX PRO 6000 server and
# the RadixArk NVFP4 base).
#
# Serves over the dashboard hardened profile runtime (read-only FS, cap-drop,
# bind-mounted HF snapshot) via:
#   ./serve.sh local sglang qwen38_27b_uncensored_fp8
#   ./serve.sh local sglang qwen38_27b_uncensored_fp8 status|logs|verify|stop
#
# Budget (measured on this box):
#   - FP8 weights ~28.5GB -> bf16 KV pool ~515K tokens (full 256K single stream).
#   - bf16 KV cache (--kv-cache-dtype bf16), 2x fp8 bytes/token.
#   - 4 concurrent requests, DSpark block 7 -> mamba cache 4*12 = 48 slots.
#   - --mm-feature-transport cpu required on this WSL2 box (CUDA-IPC crashes).
MODEL="${MODEL_PATH:?MODEL_PATH is required}"
DRAFTER="${DRAFTER_PATH:?DRAFTER_PATH is required}"
SERVED="${SERVED:?SERVED is required}"
MAXLEN="${MAXLEN:-262144}"
MEM_FRACTION="${MEM_FRACTION:-0.85}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-4}"
MAX_MAMBA_CACHE_SIZE="${MAX_MAMBA_CACHE_SIZE:-48}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-4096}"
MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-4096}"
PORT=8000
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-bf16}"

export DGX_MODEL_CAPABILITIES_PATH=/run/inference/package/capabilities.json
export MAX_MODEL_LEN="$MAXLEN"

exec python3 -m sglang_with_capabilities \
  --model-path "$MODEL" \
  --served-model-name "$SERVED" \
  --trust-remote-code \
  --mem-fraction-static "$MEM_FRACTION" \
  --attention-backend flashinfer \
  --mm-feature-transport cpu \
  --chunked-prefill-size "$CHUNKED_PREFILL_SIZE" \
  --max-prefill-tokens "$MAX_PREFILL_TOKENS" \
  --kv-cache-dtype "$KV_CACHE_DTYPE" \
  --mamba-ssm-dtype bfloat16 \
  --mamba-full-memory-ratio 4.21 \
  --context-length "$MAXLEN" \
  --mamba-radix-cache-strategy extra_buffer_lazy \
  --max-mamba-cache-size "$MAX_MAMBA_CACHE_SIZE" \
  --max-running-requests "$MAX_RUNNING_REQUESTS" \
  --linear-attn-verify-backend triton \
  --speculative-algorithm DSPARK \
  --speculative-draft-model-path "$DRAFTER" \
  --speculative-dspark-block-size 7 \
  --speculative-draft-model-quantization unquant \
  --speculative-draft-attention-backend flashinfer \
  --min-free-slots-delay 1 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --sampling-defaults model \
  --enable-metrics \
  --host 0.0.0.0 --port "$PORT" \
  "$@"

#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-nvidia/Qwen3.6-27B-NVFP4}"
DRAFTER="${DRAFTER:-z-lab/Qwen3.6-27B-DFlash}"
DEPTH="${DEPTH:-0}"
SPEC_METHOD="${SPEC_METHOD:-dflash}"
SERVED="${SERVED:-qwen27_hybrid_depth${DEPTH}}"
PORT="${PORT:-8000}"

KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8_e4m3}"
MAX_NUM_BATCHED="${MAX_NUM_BATCHED:-8352}"
FLASHINFER_AUTOTUNE="${FLASHINFER_AUTOTUNE:-1}"
AUTOTUNE_ARGS=(--enable-flashinfer-autotune)
[[ "$FLASHINFER_AUTOTUNE" == "1" ]] || AUTOTUNE_ARGS=(--no-enable-flashinfer-autotune)
SPEC_ARGS=()
if [[ "$DEPTH" != "0" ]]; then
  case "$SPEC_METHOD" in
    dflash)
      SPEC_ARGS=(
        --speculative-config
        "{\"method\":\"dflash\",\"model\":\"${DRAFTER}\",\"num_speculative_tokens\":${DEPTH}}"
      )
      ;;
    mtp)
      SPEC_ARGS=(
        --speculative-config
        "{\"method\":\"mtp\",\"num_speculative_tokens\":${DEPTH}}"
      )
      ;;
    *)
      printf 'Unsupported SPEC_METHOD=%s\n' "$SPEC_METHOD" >&2
      exit 2
      ;;
  esac
fi

exec vllm serve "$MODEL" \
  --served-model-name "$SERVED" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.923 \
  --kv-cache-dtype "$KV_CACHE_DTYPE" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED" \
  --max-num-seqs 16 \
  --mamba-ssm-cache-dtype float32 \
  --mamba-cache-dtype float16 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --disable-custom-all-reduce \
  --linear-backend cutlass \
  --limit-mm-per-prompt '{"image": 0, "video": 0}' \
  --generation-config vllm \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  "${AUTOTUNE_ARGS[@]}" \
  --tool-call-parser qwen3_coder \
  "${SPEC_ARGS[@]}"

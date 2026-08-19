#!/usr/bin/env bash
set -euo pipefail

ENGINE="${1:?engine is required}"
ARTIFACT="${2:?artifact is required}"
RANK="${NODE_RANK:?NODE_RANK is required}"
WORLD_SIZE="${WORLD_SIZE:?WORLD_SIZE is required}"
MASTER_ADDR="${MASTER_ADDR:?MASTER_ADDR is required}"
MASTER_PORT="${MASTER_PORT:?MASTER_PORT is required}"
MODEL_PATH="${MODEL_PATH:?MODEL_PATH is required}"
SERVED="${SERVED:?SERVED is required}"
API_HOST="${API_HOST:?API_HOST is required}"
API_PORT="${API_PORT:?API_PORT is required}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:?MAX_MODEL_LEN is required}"

printf 'topology rank=%s world_size=%s master=%s:%s dist_if=%s rdma_hca=%s\n' \
  "$RANK" "$WORLD_SIZE" "$MASTER_ADDR" "$MASTER_PORT" \
  "${NCCL_SOCKET_IFNAME:?NCCL_SOCKET_IFNAME is required}" \
  "${NCCL_IB_HCA:?NCCL_IB_HCA is required}"

headless=()
[[ "$ENGINE" != vllm || "$RANK" == 0 ]] || headless=(--headless)

case "$ENGINE/$ARTIFACT" in
  sglang/unsloth_qwen36_27b_nvfp4_dflash_tp2)
    DRAFTER_PATH="${DRAFTER_PATH:?DRAFTER_PATH is required}"
    args=(
      python3 -m sglang.launch_server
      --model-path "$MODEL_PATH"
      --host "$API_HOST"
      --port "$API_PORT"
      --served-model-name "$SERVED"
      --trust-remote-code
      --tp-size 2
      --nnodes "$WORLD_SIZE"
      --node-rank "$RANK"
      --dist-init-addr "$MASTER_ADDR:$MASTER_PORT"
      --attention-backend flashinfer
      --context-length "$MAX_MODEL_LEN"
      --mem-fraction-static 0.85
      --enable-metrics
      --max-running-requests 8
      --chunked-prefill-size 2048
      --tool-call-parser qwen3_coder
      --reasoning-parser qwen3
      --speculative-algorithm DFLASH
      --speculative-draft-model-path "$DRAFTER_PATH"
      --speculative-num-draft-tokens 20
      --speculative-draft-window-size 4096
      --mamba-scheduler-strategy extra_buffer
      --disable-cuda-graph
      "${headless[@]}"
    )
    ;;
  vllm/deepseek_ai_deepseek_v4_flash_dspark_tp2|\
  vllm/drowzeys_keys_deepseek_v4_flash_dspark_abliterated_32_32)
    speculative_config="{\"method\":\"dspark\",\"num_speculative_tokens\":${MTP_NUM_TOKENS:-5},\"draft_sample_method\":\"probabilistic\"}"
    CAPABILITIES_PATH="${DGX_MODEL_CAPABILITIES_PATH:?DGX_MODEL_CAPABILITIES_PATH is required}"
    CAPABILITIES_SHA256="${DGX_MODEL_CAPABILITIES_SHA256:?DGX_MODEL_CAPABILITIES_SHA256 is required}"
    [[ -f "$CAPABILITIES_PATH" && ! -L "$CAPABILITIES_PATH" ]] || {
      printf 'error: model capability profile is unavailable\n' >&2
      exit 1
    }
    [[ "$(sha256sum "$CAPABILITIES_PATH" | cut -d' ' -f1)" == "$CAPABILITIES_SHA256" ]] || {
      printf 'error: model capability profile digest mismatch\n' >&2
      exit 1
    }
    args=(
      /usr/local/bin/vllm serve "$MODEL_PATH"
      --served-model-name "$SERVED"
      --host "$API_HOST"
      --port "$API_PORT"
      --middleware model_capabilities.ModelCapabilitiesMiddleware
      --trust-remote-code
      --tensor-parallel-size 2
      --pipeline-parallel-size 1
      --kv-cache-dtype "${KV_CACHE_DTYPE:-fp8_ds_mla}"
      --block-size 256
      --max-model-len "$MAX_MODEL_LEN"
      --max-num-seqs "${MAX_NUM_SEQS:-6}"
      --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-8192}"
      --long-prefill-token-threshold "${LONG_PREFILL_TOKEN_THRESHOLD:-0}"
      --scheduler-cls decode_aware_scheduler.DecodeAwareScheduler
      --max-cudagraph-capture-size "$(( ${MAX_NUM_SEQS:-6} * (${MTP_NUM_TOKENS:-5} + 1) ))"
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.85}"
      --enable-prefix-caching
      --async-scheduling
      --enable-chunked-prefill
      --speculative-config "$speculative_config"
      --tokenizer-mode deepseek_v4
      --distributed-executor-backend mp
      --moe-backend flashinfer_b12x
      --tool-call-parser deepseek_v4
      --enable-auto-tool-choice
      --reasoning-parser deepseek_v4
      --reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"<think>","reasoning_end_str":"</think>"}'
      --default-chat-template-kwargs '{"thinking":false}'
      --generation-config vllm
      --enable-flashinfer-autotune
      --nnodes "$WORLD_SIZE"
      --node-rank "$RANK"
      --master-addr "$MASTER_ADDR"
      --master-port "$MASTER_PORT"
      "${headless[@]}"
    )
    ;;
  *)
    printf "error: unsupported cluster artifact '%s/%s'\n" "$ENGINE" "$ARTIFACT" >&2
    exit 1
    ;;
esac

exec "${args[@]}"

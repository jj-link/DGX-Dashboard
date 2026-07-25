#!/usr/bin/env bash
# SGLang launch for Qwen3.6-27B FP8 target + DFlash drafter on RTX PRO 6000
# Blackwell (SM120, WSL2). Run via the hardened wrapper:
#   ./run_sglang_docker.sh serve/sglang/serve_qwen36_27b_fp8_dflash.sh
#   ./serve.sh sglang 27b_fp8_dflash
#
# WHY SGLANG (vs the vLLM script): the DFlash drafter is trained at ~3-4k context
# and its acceptance collapses to ~0% past ~64k. vLLM keeps drafting anyway
# (full draft+verify cost, zero benefit). SGLang's --speculative-draft-window-size
# clamps the drafter to the last N tokens so it stays in its trained range at long
# context. vLLM has no equivalent -- that flag is the entire reason this exists.
#
# !! UNVERIFIED ON SM120 !! No source confirms this exact stack (Qwen3.6-27B-FP8
# + DFLASH + consumer-Blackwell SM120) running. SGLang's fast attention/FP8-GEMM
# backends target datacenter Blackwell (SM100); SM120 has open bugs (sglang
# #16816, #15342, #24633). Defaults below pick the SM120-safe paths. If launch
# fails, see the FALLBACKS block below.
#
# Requires SGLang >= 0.5.10 (Qwen3.6 support lives in qwen3_5.py).
set -u

MODEL="${MODEL:-Qwen/Qwen3.6-27B-FP8}"          # block-wise FP8, auto-detected from the checkpoint
DRAFTER="${DRAFTER:-z-lab/Qwen3.6-27B-DFlash}"
SERVED="${SERVED:-Qwen3.6-27B-FP8-DFLASH}"       # same served name as the vLLM script -> clients/benches are drop-in
MAXLEN="${MAXLEN:-250000}"                      # full native 256k window (max_position_embeddings=262144, rope_scaling: none)
NUM_SPEC="${NUM_SPEC:-16}"                      # local 27B staged DFlash sweep winner
MEM_FRACTION="${MEM_FRACTION:-0.82}"            # WSL2: leave real VRAM headroom. At 0.92 the
# draft model (~3GB) + draft KV (~6.6GB) + cuda graphs stacked on top filled the card to ~0 free
# (nvidia-smi 96544/97887 MiB), which on WSL2 spills to slow shared sysmem. 0.85 = bigger KV pool
# for the 256k window (~6GB headroom); watch the "Maximum concurrency for 262144" line at boot.
ATTN_BACKEND="${ATTN_BACKEND:-flashinfer}"      # SM120: flashinfer works; trtllm_mha/fa3/fa4 are SM100/SM90.
# Single-stream: cap cuda-graph batch sizes. SGLang's default captures up to 256
# (50+ sizes) x spec draft tokens, reserving huge memory that starved the
# KV/mamba pool ("Not enough memory" at pool init). 8 is ample for batch=1.
CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-8}"
# Single-stream: cap concurrent requests. CRITICAL on this hybrid+spec setup --
# the mamba state cache + spec-dec intermediate states scale with this value
# (mcpr * max_running_requests * num_draft_tokens). SGLang resets it to 48 for
# spec decoding, which consumed the ENTIRE memory budget and made KV-pool init
# fail ("Not enough memory"). 2 is plenty for one interactive user and frees
# tens of GB for the context pool.
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-4}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-16384}" # match local A3B prefill sweep winner
MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-32768}"     # match local A3B prefill sweep winner
PORT="${PORT:-8000}"

# THE lever this script exists for: clamp the drafter's attention to the last N
# tokens so it stays in its ~3-4k trained range at long context. Empty = full
# context = the broken (collapsing) baseline. Measured on SM120 2026-05-23:
# Older long-context sweep picked 6144 for consistency, but the local staged
# 27B tuning run on this launch shape picked 2048 with NUM_SPEC=16.
# (benchmarks/results/dflash-tune-27b-*.tsv has the latest sweep.)
DRAFT_WINDOW="${DRAFT_WINDOW:-2048}"
DWIN_ARG=()
# DRAFT_WINDOW=off|0|none -> omit the flag = SGLang full-context drafter (the
# "collapses at long context" baseline, for A/B against the window).
case "${DRAFT_WINDOW}" in off|0|none) ;; *) DWIN_ARG=(--speculative-draft-window-size "${DRAFT_WINDOW}") ;; esac

# Parsers: SGLang uses qwen3 reasoning + qwen3_coder tool-calls (NOT vLLM's qwen3_xml).
PARSER_ARG=(--reasoning-parser qwen3 --tool-call-parser qwen3_coder)
[ "${PARSERS:-1}" = "1" ] || PARSER_ARG=()

# Mamba-hybrid + radix cache + spec-decode conflict: SGLang ERRORS at init if
# radix cache is on with --mamba-scheduler-strategy no_buffer (the 'auto' default
# for this model). Two ways out: disable radix cache, OR extra_buffer +
# SGLANG_ENABLE_SPEC_V2=1 -- which is the DEFAULT here. Prefix caching is ON:
# essential for agentic multi-turn (reuses the growing conversation prefix instead
# of re-prefilling it every turn). Set RADIX=0 to disable (smaller mamba
# reservation, bigger KV pool; fine for single-request benchmarks).
if [ "${RADIX:-1}" = "1" ]; then
  CACHE_ARG=()
  MAMBA_ARG=(--mamba-scheduler-strategy "${MAMBA_STRATEGY:-extra_buffer}")
  export SGLANG_ENABLE_SPEC_V2="${SGLANG_ENABLE_SPEC_V2:-1}"
else
  CACHE_ARG=(--disable-radix-cache)
  MAMBA_ARG=(--mamba-scheduler-strategy "${MAMBA_STRATEGY:-auto}")
fi

# ---- FALLBACKS if it errors on SM120 (community-sourced, unofficial) ----
#  FP8-GEMM error:          ATTN_BACKEND unchanged; add `--fp8-gemm-backend triton`
#                           and `export SGLANG_ENABLE_DEEP_GEMM=0` (sglang #16816)
#  attention shmem error:   ATTN_BACKEND=triton
#  garbage output:          do NOT enable fp8 KV cache (we keep bf16 -- no flag set)
#  radix/mamba assert:      keep MAMBA_STRATEGY=auto, or `export SGLANG_ENABLE_SPEC_V2=1`
#  last resort image:       IMAGE=voipmonitor/llm-pytorch-blackwell:nightly (unofficial SM120 build)

exec python3 -m sglang.launch_server \
  --model-path "$MODEL" \
  --served-model-name "$SERVED" \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path "$DRAFTER" \
  --speculative-num-draft-tokens "$NUM_SPEC" \
  "${DWIN_ARG[@]}" \
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
  "${PARSER_ARG[@]}"

#!/usr/bin/env bash
# SGLang launch for Qwen3.6-35B-A3B FP8 target + DFlash drafter on RTX PRO 6000
# Blackwell (SM120, WSL2). Run via the hardened profile runtime:
#   ./serve.sh sglang qwen36_a3b_fp8_dflash
#
# Local twin of the Spark script sweep/serve_sglang_a3b_dflash_remote.sh (which
# is a bare `docker run` for GB10/sm_121a). Same model + DFlash config; this one
# is the in-container inner script the wrapper execs. DFlash on this stack hit
# ~77 tok/s @ accept_len ~4.4 on the Spark; expect similar-or-better here (the
# RTX 6000 has far more memory bandwidth than GB10).
#
# WHY SGLANG (vs the vLLM a3b script): --speculative-draft-window-size clamps the
# DFlash drafter to its trained ~3-4k range so acceptance doesn't collapse at
# long context. (vLLM PR #40898 grew --speculative-dflash-draft-window-size too,
# but this keeps an apples-to-apples SGLang path next to the 27B script.)
#
# !! UNVERIFIED ON SM120 !! Same caveat as the 27B script: SGLang's fast
# attention/FP8-GEMM backends target datacenter Blackwell (SM100); SM120 has open
# bugs. Defaults below pick the SM120-safe paths; see FALLBACKS if launch fails.
#
# !! 256k ON A 96GB CARD IS TIGHT !! The Spark holds 256k KV in 128GB unified mem
# at 0.82. This card is 96GB. If SGLang dies at pool init ("Not enough memory"),
# drop MAXLEN (e.g. MAXLEN=131072 ./serve.sh sglang a3b_fp8_dflash). Do NOT chase
# it with a higher MEM_FRACTION -- 0.82 is the WSL2-safe ceiling; above it the KV
# pool spills to slow shared sysmem (see the 27B script's notes).
#
# Requires SGLang >= 0.5.10 (Qwen3.6 lives in qwen3_5.py). Both weights are cached
# locally: Qwen/Qwen3.6-35B-A3B-FP8 and z-lab/Qwen3.6-35B-A3B-DFlash.
# NOTE: shares port 8000 + the GPU with the 27B server -- stop that first
#   (docker rm -f sglang-serve_qwen36_27b_fp8_dflash).
set -euo pipefail

MODEL="${MODEL_PATH:?MODEL_PATH is required}"
DRAFTER="${DRAFTER_PATH:?DRAFTER_PATH is required}"
SERVED="${SERVED:?SERVED is required}"
MAXLEN="${MAXLEN:-262144}"                      # 256k = native max. Lower to 131072 if 96GB can't hold the KV pool.
NUM_SPEC="${NUM_SPEC:-12}"                       # local tune winner: 12 + 2048
MEM_FRACTION="${MEM_FRACTION:-0.82}"            # WSL2 ceiling; 0.92 spilled to slow sysmem on the 27B run
ATTN_BACKEND="${ATTN_BACKEND:-flashinfer}"      # SM120: flashinfer works; trtllm_mha/fa3/fa4 are SM100/SM90
CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-8}"     # single-stream: small graph set (batch=1)
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-2}"  # single-stream: caps mamba + spec-dec state growth
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-16384}" # local prefill sweep winner: 16k + 32k
MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-32768}"     # local prefill sweep winner: 16k + 32k
PORT=8000

# THE lever this script exists for: clamp the drafter's attention to the last N
# tokens so it stays in its trained range at long context. off|0|none omits
# it = full-context baseline. Local A3B tune winner: NUM_SPEC=12, window=2048.
DRAFT_WINDOW="${DRAFT_WINDOW:-2048}"
DWIN_ARG=()
case "${DRAFT_WINDOW}" in off|0|none) ;; *) DWIN_ARG=(--speculative-draft-window-size "${DRAFT_WINDOW}") ;; esac

# SGLang parsers: qwen3 reasoning + qwen3_coder tool-calls (NOT vLLM's qwen3_xml).
PARSER_ARG=(--reasoning-parser qwen3 --tool-call-parser qwen3_coder)
[ "${PARSERS:-1}" = "1" ] || PARSER_ARG=()

# Mamba-hybrid + radix cache + spec-decode: SGLang errors at init if radix cache
# is on with the 'no_buffer' mamba default. extra_buffer + SGLANG_ENABLE_SPEC_V2=1
# is the default here (prefix caching ON -- reuses the growing conversation prefix
# across agentic turns). RADIX=0 disables radix cache (bigger KV pool; good for
# single-request benches). NB: with DFLASH, SGLang logs that spec-v2 overlap is
# "not supported yet" and disables it -- the env is harmless either way.
if [ "${RADIX:-1}" = "1" ]; then
  CACHE_ARG=()
  MAMBA_ARG=(--mamba-scheduler-strategy "${MAMBA_STRATEGY:-extra_buffer}")
  export SGLANG_ENABLE_SPEC_V2="${SGLANG_ENABLE_SPEC_V2:-1}"
else
  CACHE_ARG=(--disable-radix-cache)
  MAMBA_ARG=(--mamba-scheduler-strategy "${MAMBA_STRATEGY:-auto}")
fi

# ---- FALLBACKS if it errors on SM120 (community-sourced, unofficial) ----
#  OOM at pool init:        MAXLEN=131072 ./serve.sh sglang a3b_fp8_dflash  (256k KV won't fit 96GB)
#  FP8-GEMM error:          add `--fp8-gemm-backend triton` and `export SGLANG_ENABLE_DEEP_GEMM=0`
#  attention shmem error:   ATTN_BACKEND=triton
#  garbage output:          do NOT enable fp8 KV cache (we keep bf16 -- DFlash needs it anyway)
#  radix/mamba assert:      RADIX=0, or keep MAMBA_STRATEGY=auto

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
  "${PARSER_ARG[@]}" \
  "$@"

# 5-Model Head-to-Head on Spark (Tier 1)

Compares 5 models served on the GB10/spark via vLLM:

| Alias                   | Repo                                              | Quant   | Notes |
|-------------------------|---------------------------------------------------|---------|-------|
| `minimax`               | saricles/MiniMax-M2.7-REAP-172B-A10B-NVFP4-GB10   | NVFP4   | reasoning model, slow |
| `qwen36-27b-nvfp4`      | sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP          | NVFP4   | served w/o MTP for fair compare |
| `qwen36-27b-fp8`        | Qwen/Qwen3.6-27B-FP8                              | FP8     | |
| `qwen36-35b-a3b-nvfp4`  | RedHatAI/Qwen3.6-35B-A3B-NVFP4                    | NVFP4   | MoE, 3B active |
| `qwen36-35b-a3b-fp8`    | Qwen/Qwen3.6-35B-A3B-FP8                          | FP8     | MoE, 3B active |

All run on the same GB10 endpoint at `http://gx10-a6c7.lan:8000/v1` — only
one at a time. The sweep script swaps them via docker.

## Benches

### Tier 1a: Polyglot-style coding bench (READY — custom harness)

- Problem set: 34 Python exercises from
  `aider/tmp.benchmarks/polyglot-benchmark/python/exercises/practice/`
- Driver: **`oneshot_bench.py`** — a slim ~150-line script (NOT aider's
  benchmark.py). It assembles instructions + stub file into a single
  user message, sends it to `/v1/chat/completions` with bounded
  `max_tokens`, extracts the largest python fence from the response, writes
  it to the stub path, and runs pytest. Output: per-problem
  `.oneshot.results.json` + a summary at
  `results/<alias>-oneshot-<ts>.json`.
- Orchestration: **`sweep_oneshot.sh`** swaps the spark container between
  models and runs `oneshot_bench.py` against each in turn. Same problem
  set across all 5 models for direct comparison.
- Aggregation: **`aggregate_minimal.py`** flattens the latest result per
  alias into a table (pass count, pass %, avg latency, avg output tokens).

**Why not aider's benchmark.py:** the upstream benchmark harness hangs on
these specific reasoning models — the response is received by vLLM
(metrics confirm tokens are generated), but aider's openai-compatible
client never returns from the call. Direct `aider --no-stream --message
...` calls work fine, so the issue is harness-specific (likely related to
how it handles `reasoning_content` in non-streaming responses). The
custom harness sidesteps it.

### Tier 1b: BFCL v3 Live (STUB)

Tool-use benchmark: ~250 user-contributed function-calling tasks, AST-graded.

Status: **harness installed, run-script written, but not yet usable** because:

1. BFCL evaluates in *prompt mode*: it formats the chat template + tools spec
   client-side, sends raw text to `/v1/completions`, and parses tool calls
   from the response text. This **bypasses vLLM's native `--tool-call-parser`**.
2. Our 5 models are not in BFCL's upstream registry — Qwen3.6 series and
   MiniMax-M2 are too new. To get faithful scores we'd need to:
   - Add new entries to `gorilla/.../bfcl_eval/constants/model_config.py`
     and `.../supported_models.py` (one per model, paired with a handler).
   - Download tokenizer files locally so BFCL can `apply_chat_template`.
   - Possibly write a MiniMax-specific handler if the existing
     `QuickTestingOSSHandler` falls short on the tool-call format.

`run_bfcl.sh` documents the approach but exits with "tokenizer dir missing"
until step 2 is done. Tracked as follow-up.

## Layout

```
benchmarks/
├── README.md                  ← you are here
├── sweep.sh                   ← top-level orchestrator
├── aggregate.sh               ← collect _stats.yml across all run dirs into a table
├── run_aider_spark.sh               ← per-model aider runner
├── run_bfcl.sh                ← per-model BFCL runner (stub)
├── results/                   ← sweep logs + aggregated tables
├── spark-serve/
│   ├── models.env             ← alias → repo/served-name/parser config
│   └── swap.sh                ← ssh spark → docker rm vllm → docker run …
├── aider/                     ← editable install + .venv
└── gorilla/.../leaderboard/   ← BFCL repo + .venv
```

## How to run

```bash
# Smoke a single model (1 problem):
python3 oneshot_bench.py qwen36-27b-fp8 --num-tests 1 --keywords robot-name

# Full 5-model sweep (default: 3 problems per model, same problems for all):
./sweep_oneshot.sh

# Smaller / bigger / different selection:
NUM_TESTS=5 ./sweep_oneshot.sh
NUM_TESTS=10 ./sweep_oneshot.sh
KEYWORDS="robot-name,hangman,wordy" NUM_TESTS=3 ./sweep_oneshot.sh

# Just aggregate the latest results table:
python3 aggregate_minimal.py
```

## Known caveats

- **GB10 decode speeds are slow for this workload.** Measured during setup:
  - `minimax` (172B-A10B, NVFP4): ~22 tok/s — large MoE benefits from low active params (10B).
  - `qwen36-27b-nvfp4`: ~12 tok/s — dense 27B fully active, no MoE advantage.
  - `qwen36-27b-fp8`: ~7.5 tok/s — FP8 path on GB10 is slower than NVFP4 (Blackwell is optimized for NVFP4 first).
  - The 35B-A3B variants haven't been measured yet but A3B's 3B active params should make them the fastest of the lot.

  Per-problem latency with reasoning on is several minutes. Start with `NUM_TESTS=5` for a first pass.
- **NVFP4 tokenizer hack**: both `sakamakismile` and `RedHatAI` NVFP4 repos
  ship `tokenizer_config.json` with `tokenizer_class: TokenizersBackend` which
  isn't a transformers class. `models.env` works around this by passing
  `--tokenizer Qwen/Qwen3.6-27B` (or the matching official Qwen repo for the
  35B-A3B). Weights still come from the quantized repo.
- **Quant ≠ model**: NVFP4 vs FP8 results may differ; that's intentional —
  the goal is to see how much each quantization costs in real coding quality.
- **Sakamakismile NVFP4 has MTP**: we serve it *without* `--speculative-config`
  so we measure base accuracy, not the MTP-accelerated path.
- **Reasoning content leaks**: all 5 models are reasoning models. Aider's
  `whole` edit format is robust to leading `<think>` blocks (it scans for the
  code fence), but if you see a `well_formed%` cliff for any model, that's
  the likely culprit.
- **Single-host serial**: only one model loaded at a time. A failed swap
  leaves the previous model live — `swap.sh` waits for `/v1/models` to
  confirm the new name before returning.

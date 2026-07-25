# Sweep status snapshot

Last update: 2026-05-13 12:44

## What's running

- **Background task:** `sweep_oneshot.sh` retry pass on the 2 failing
  models (minimax + qwen36-27b-fp8) with `TIMEOUT=1800 MAX_TOKENS=4096`.
- Log: `results/sweep-oneshot-<latest-ts>.log`

## Results so far (partial)

```
alias                      |   pass |   rate |  avg_lat_s | avg_out_tok | notes
------------------------------------------------------------------------------------------
minimax                    |  0/3   |   0.0% |          0 |           0 | 3 timeouts at 600s — retry pending
qwen36-27b-nvfp4           |  1/3   |  33.3% |      231.5 |        2344 | 2 timeouts on beer-song/book-store
qwen36-27b-fp8             |  0/3   |   0.0% |          0 |           0 | 3 timeouts at 600s — retry pending
qwen36-35b-a3b-nvfp4       |  2/3   |  66.7% |      155.9 |        5336 |
qwen36-35b-a3b-fp8         |  2/3   |  66.7% |       67.0 |        2747 |
```

Run on the FIRST 3 alphabetical Python polyglot problems (same 3 for all
models so scores are directly comparable):
  1. affine-cipher
  2. beer-song
  3. book-store

## Notes on results

- **35B-A3B (both quants) clearly leads** — fastest *and* most accurate.
  FP8 is ~2.3× faster than NVFP4 for this MoE on GB10 (67s vs 155s avg).
  Both got the same problems right and wrong, so quantization didn't move
  the needle on quality.
- **27B-NVFP4 (dense)** is mid-tier. The two timeouts were
  infrastructure-bound, not capability-bound — the 600s HTTP cap was hit
  while the model was still happily decoding at ~12 tok/s. Retry pending.
- **MiniMax / 27B-FP8 timed out on every problem.** Both are slow paths
  on GB10 (minimax has 10B active params; FP8 path is unoptimized vs.
  NVFP4 on Blackwell). Retry with `TIMEOUT=1800 MAX_TOKENS=4096` is in
  flight — that drops the thinking budget while extending the wall clock,
  which should land real scores instead of timeouts.

## Layout (artifacts)

- **Harness:** `oneshot_bench.py` (custom — sidesteps aider/benchmark.py
  hangs on these specific reasoning models, see README §Tier 1a)
- **Orchestrator:** `sweep_oneshot.sh`
- **Aggregator:** `aggregate_minimal.py`
- **Serve scripts:** `spark-serve/{models.env, swap.sh}`
- **Per-model results:** `results/<alias>-oneshot-<ts>.json`
- **Per-problem dumps:** `aider/tmp.benchmarks/polyglot-benchmark/python/exercises/practice/<problem>/.oneshot.results.json`

## To re-run / extend

```bash
cd /mnt/c/Users/josep/Projects/personal/test/benchmarks

# Re-run a single model (5 problems):
python3 oneshot_bench.py qwen36-27b-fp8 --num-tests 5 \
  --max-tokens 4096 --timeout 1800

# Re-run the whole sweep across all 5 with the same 3 problems:
NUM_TESTS=3 MAX_TOKENS=4096 TIMEOUT=1200 ./sweep_oneshot.sh

# Pin all models to specific problems:
KEYWORDS="robot-name,hangman,wordy" NUM_TESTS=3 ./sweep_oneshot.sh

# See current results table:
python3 aggregate_minimal.py
```

## To stop the sweep mid-way

```bash
pkill -9 -f 'sweep_oneshot\|oneshot_bench'
```

## Known issues / follow-ups

1. **BFCL Live AST not yet runnable** — see README. Need to register custom
   model entries in `gorilla/.../bfcl_eval/constants/model_config.py` and
   `.../supported_models.py` for Qwen3.6 + MiniMax-M2.

2. **Aider polyglot upstream harness hangs** — `benchmark.py` opens a
   `/v1/chat/completions` stream that vLLM finishes (metrics confirm
   completion tokens) but aider/LiteLLM never returns. Direct
   `aider --no-stream --message ...` calls work fine, so it's harness-
   specific. Worth filing upstream once we have more data.

3. **FP8 path is unexpectedly slow on dense 27B GB10** — ~7.5 tok/s for
   Qwen-27B-FP8 vs ~12 tok/s for the NVFP4 variant. Interesting that for
   the MoE 35B-A3B the FP8 path was *faster* than NVFP4. Worth digging
   into the `--attention-backend flashinfer` + `kv-cache-dtype fp8_e4m3`
   interaction for dense FP8 weights specifically.

4. **MiniMax compilation-config dropped** — `--compilation-config={JSON}`
   didn't survive the `ssh → docker run` quoting chain in `swap.sh`. The
   model still loads correctly but goes through the default compile path.
   If you want it back, run docker directly on spark with the original
   args.

5. **NVFP4 tokenizer hack** — both `sakamakismile` and `RedHatAI` NVFP4
   repos ship `tokenizer_config.json` with `tokenizer_class:
   TokenizersBackend` which isn't a transformers class. `models.env`
   works around this by passing `--tokenizer Qwen/Qwen3.6-27B` (or the
   matching official Qwen repo for the 35B-A3B).

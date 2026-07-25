# 5-model spark/local head-to-head — final results

_Generated 2026-05-17 08:00:02 from the latest sweep per alias._

## Setup

- **34 Python polyglot exercises**, one-shot prompts (instructions + stub).
- `max_tokens=32768`, `temperature=0.0`, streaming SSE.
- Pytest scores each: pass = full test suite passes, anything else = fail.
- Spark (GB10): minimax-m2.7-REAP-172B-A10B-NVFP4
- Local (RTX PRO 6000 Blackwell, WSL2): the 4 Qwens

## Pass rate

| Model | Quant | Pass | Rate | Avg latency | Avg out tok |
|---|---|---|---|---:|---:|
| minimax | NVFP4 | 6/23 *(PARTIAL)* | 26.1% | 1529.3s | 24,508 |
| qwen36-27b-nvfp4 | NVFP4 | 13/34 | 38.2% | 122.8s | 6,572 |
| qwen36-27b-fp8 | FP8 | 13/34 | 38.2% | 119.1s | 6,455 |
| qwen36-35b-a3b-nvfp4 | NVFP4 | 2/3 | 66.7% | 155.9s | 5,336 |
| qwen36-35b-a3b-fp8 | FP8 | 10/34 | 29.4% | 30.7s | 7,249 |

## Token cost

| Model | Total in | Total out | Median out | p95 out | Avg out (✓) | Avg out (✗) |
|---|---:|---:|---:|---:|---:|---:|
| minimax | 26,415 | 563,673 | 32,768 | 32,768 | 27,591 | 23,419 |
| qwen36-27b-nvfp4 | 40,430 | 223,442 | 4,857 | 12,354 | 5,835 | 7,028 |
| qwen36-27b-fp8 | 41,122 | 219,473 | 5,182 | 13,810 | 5,852 | 6,829 |
| qwen36-35b-a3b-nvfp4 | 6,118 | 16,009 | 4,451 | 4,451 | 4,342 | 7,324 |
| qwen36-35b-a3b-fp8 | 34,172 | 246,459 | 7,192 | 11,420 | 5,582 | 7,943 |

## Per-problem pass matrix

| problem | mmax | 27N | 27F | 35aN | 35aF |
|---|---|---|---|---|---|
| affine-cipher | ✗ | ✓ | ✓ | ✓ | ✓ |
| beer-song | ✓ | ✓ | ✓ | ✓ | ✓ |
| book-store | ✗ | ✗ | ✗ | ✗ | ✗ |
| bottle-song | ✗ | ✗ | ✗ | ? | ✗ |
| bowling | ✓ | ✓ | ✓ | ? | ✗ |
| connect | ✗ | ✗ | ✗ | ? | ✗ |
| dominoes | ✓ | ✓ | ✓ | ? | ✓ |
| dot-dsl | ✗ | ✗ | ✗ | ? | ✗ |
| food-chain | ✗ | ✓ | ✓ | ? | ✗ |
| forth | ✗ | ✗ | ✗ | ? | ✗ |
| go-counting | ✗ | ✗ | ✗ | ? | ✗ |
| grade-school | ✓ | ✓ | ✓ | ? | ✗ |
| grep | ✗ | ✗ | ✗ | ? | ✗ |
| hangman | ✗ | ✗ | ✗ | ? | ✓ |
| list-ops | ✗ | ✗ | ✗ | ? | ✗ |
| paasio | ✗ | ✗ | ✗ | ? | ✗ |
| phone-number | ✗ | ✗ | ✗ | ? | ✗ |
| pig-latin | ✓ | ✓ | ✓ | ? | ✓ |
| poker | ✗ | ✗ | ✗ | ? | ✗ |
| pov | ✗ | ✗ | ✗ | ? | ✗ |
| proverb | ✓ | ✓ | ✓ | ? | ✓ |
| react | ✗ | ✗ | ✗ | ? | ✗ |
| rest-api | ✗ | ✗ | ✗ | ? | ✗ |
| robot-name | ? | ✗ | ✗ | ? | ✗ |
| scale-generator | ? | ✓ | ✓ | ? | ✗ |
| sgf-parsing | ? | ✗ | ✗ | ? | ✗ |
| simple-linked-list | ? | ✗ | ✗ | ? | ✗ |
| transpose | ? | ✓ | ✓ | ? | ✓ |
| tree-building | ? | ✗ | ✗ | ? | ✗ |
| two-bucket | ? | ✓ | ✓ | ? | ✓ |
| variable-length-quantity | ? | ✓ | ✓ | ? | ✓ |
| wordy | ? | ✗ | ✗ | ? | ✗ |
| zebra-puzzle | ? | ✓ | ✓ | ? | ✓ |
| zipper | ? | ✗ | ✗ | ? | ✗ |

## BFCL v3 Live AST (tool-use)

**Parser note:** BFCL here runs in *prompt mode* (`bfcl generate` formats
the prompt client-side from the model's tokenizer chat template, hits
`/v1/completions`, and parses tool calls itself). It **bypasses vLLM's
`--tool-call-parser` entirely** — so the served parser (hermes vs
qwen3_xml) does not affect these numbers; they are valid as-is. What does
govern BFCL tool parsing is the BFCL registry entry
(`Qwen/Qwen3-30B-A3B-Instruct-2507-FC` for 35b-a3b) + the local tokenizer
template. The vLLM `--tool-call-parser` only matters for the live serving
path (e.g. the Hermes agent's native tool calls), which the serve scripts
set to `qwen3_xml`. Polyglot results above are also parser-independent.

Accuracy = correct / total across all live categories (simple, multiple, parallel,
parallel_multiple, relevance, irrelevance). 2251 scored cases.

| Model | Quant | Correct | Total | Accuracy | Parser |
|---|---|---:|---:|---:|---|
| minimax | NVFP4 | — | — | — | no BFCL run |
| qwen36-27b-nvfp4 | NVFP4 | 1,729 | 2,251 | 76.8% | hermes |
| qwen36-27b-fp8 | FP8 | 1,807 | 2,251 | 80.3% | hermes |
| qwen36-35b-a3b-nvfp4 | NVFP4 | — | — | — | no BFCL run |
| qwen36-35b-a3b-fp8 | FP8 | 1,764 | 2,251 | 78.4% | hermes |

## Source data

- minimax polyglot: `results/minimax-oneshot-20260513-203654.json`
- qwen36-27b-nvfp4 polyglot: `results/qwen36-27b-nvfp4-oneshot-20260513-211927.json`
- qwen36-27b-fp8 polyglot: `results/qwen36-27b-fp8-oneshot-20260513-223549.json`
- qwen36-35b-a3b-nvfp4 polyglot: `results/qwen36-35b-a3b-nvfp4-oneshot-20260513-115524.json`
- qwen36-35b-a3b-fp8 polyglot: `results/qwen36-35b-a3b-fp8-oneshot-20260513-201915.json`

- minimax BFCL: (no run or no scores)
- qwen36-27b-nvfp4 BFCL: `results/bfcl/qwen36-27b-nvfp4-20260514-034906/`
- qwen36-27b-fp8 BFCL: `results/bfcl/qwen36-27b-fp8-20260514-044122/`
- qwen36-35b-a3b-nvfp4 BFCL: (no run or no scores)
- qwen36-35b-a3b-fp8 BFCL: `results/bfcl/qwen36-35b-a3b-fp8-20260514-050614/`

## Reproduce

```bash
cd /mnt/c/Users/josep/Projects/personal/test/benchmarks
# Spark (minimax only):
ALIASES=minimax NUM_TESTS=34 MAX_TOKENS=32768 TIMEOUT=3600 ./sweep_oneshot.sh
ALIAS=minimax ./sweep_bfcl_spark.sh
# Local (4 Qwens):
NUM_TESTS=34 MAX_TOKENS=32768 TIMEOUT=3600 ./sweep_oneshot_local.sh
./sweep_bfcl_local.sh
# Aggregate:
python3 finalize_results.py
```
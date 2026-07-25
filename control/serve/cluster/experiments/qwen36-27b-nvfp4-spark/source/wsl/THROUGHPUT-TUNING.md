# Single-request throughput tuning notes

Target: maximize single-request generation throughput for `qwen36-27b-nvfp4` on Spark2+Spark3 TP=2 while preserving correctness: both GPUs must stay active during long generation.

Benchmark shape used during tuning:

- API: `http://spark2-ts:8888/v1/chat/completions`
- Model: `qwen36-27b-nvfp4`
- Request: one chat request, `max_tokens=512`, `temperature=0.0`
- GPU activity sampled from both Spark2 and Spark3 via `nvidia-smi`
- Runtime: `vllm/vllm-openai:v0.23.0`, non-Ray `mp`, Spark3 `--headless`

## Results

| Variant | Result | Throughput | GPU behavior | Decision |
|---|---:|---:|---|---|
| Stable baseline: `--enforce-eager`, `--no-enable-flashinfer-autotune`, `--no-async-scheduling`, `MAX_NUM_SEQS=8` | Success | 16.19 tok/s cold-ish; 16.26 tok/s warm second run | Both GPUs active | Keep |
| Remove `--enforce-eager`, keep FlashInfer autotune disabled | Failed | timed out after 300s | Spark3 ~95% avg, Spark2 ~2.6% avg | Reject |
| Keep eager, enable FlashInfer autotune | Success | 14.91 tok/s | Both GPUs active | Reject: slower and longer startup |
| Keep eager/no-autotune, add `--generation-config vllm` | Success | 14.41 tok/s | Both GPUs active | Reject: slower |
| Keep eager/no-autotune, `MAX_NUM_SEQS=1` | Success | 14.40 tok/s | Both GPUs active | Reject: slower |
| Keep eager/no-autotune, enable async scheduling | Success | 15.21 tok/s | Both GPUs active | Reject: slower for single request |

## Current best config

The best tested single-request config is the stable baseline:

```text
--enforce-eager
--no-enable-flashinfer-autotune
--no-async-scheduling
MAX_NUM_SEQS=8
```

`--enforce-eager` is still required for correctness/stability on this official vLLM 0.23.0 runtime. Removing it reproduced the bad long-generation failure mode: one node pegged while the other idled, and the request timed out.

FlashInfer autotune is not helpful here: it increases startup time and produced lower measured single-request throughput.

Async scheduling is also not helpful for this specific single-request benchmark. It may still matter for multi-request throughput, but that was not the target of this tuning pass.

## Practical note

The first request after restart may be slower because vLLM still JIT-compiles some Triton kernels during inference. Warm steady-state single-request throughput on the kept config was about `16.26 tok/s` with both GPUs active.

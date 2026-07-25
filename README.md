# vLLM Qwen serving / benchmarking workspace

Layout after the 2026-05-17 reorg. The one rule: **`serve.sh`, `run_vllm_docker.sh`, `run_sglang_docker.sh`,
`moe_configs/`, `moe_tune_out/`, and `tok-qwen35-122b/` stay at the root** —
the hub script computes `PROJECT_DIR` from its own location, mounts the whole
tree at `/workspace`, and references those dirs by hardcoded path. Moving them
breaks the container.

```
serve.sh               Dispatcher (easiest entry): engine + short name.
                         ./serve.sh 27b_fp8_dflash         # engine defaults to vllm
                         ./serve.sh sglang 27b_fp8_dflash
                       Short name = unique fragment of a serve/<engine>/ script;
                       full paths still work. Env (DETACH=1 etc.) passes through.
run_vllm_docker.sh     vLLM hub: runs a serve/vllm/ script in the hardened
                       container. Full paths still work:
                         ./run_vllm_docker.sh serve/vllm/serve_qwen36_a3b_fp8.sh
run_sglang_docker.sh   SGLang hub: same hardening, for serve/sglang/ scripts.
                       SGLang >=0.5.10 image; SM120/RTX-Blackwell support is
                       still rough -- see the serve/sglang script header.

serve/                 Launch scripts, split by engine:
  vllm/                  vLLM scripts, one per model/quant/feature combo.
  sglang/                SGLang scripts (DFlash drafter window for long context).
bench/                 Benchmark clients + bench_suite.sh. No cross-refs out.
sweep/                 Drivers that relaunch containers and run benches.
                         sweep_*.sh    local, call ../run_vllm_docker.sh
                         *_remote.sh   self-contained, run ON the Spark box
results/               *_tps_results.md measurement writeups.
docker/                Dockerfile (qwen-vllm:pinned), Dockerfile.dflash (:dflash).
                       Build context changed: docker build -f docker/Dockerfile .
moe_configs/           Tuned fused-MoE tile configs bind-mounted by the hub.
moe_tune_out/          Output of MoE autotuning runs.
tok-qwen35-122b/       Tokenizer mounted at /workspace/tok-qwen35-122b.
benchmarks/            Pre-existing, already self-organized (has its own README).
```

Unsloth Qwen3.6 NVFP4 finalists (all bind only to `127.0.0.1:8000`):
```bash
./serve.sh vllm unsloth_qwen36_27b_nvfp4
./serve.sh vllm unsloth_qwen36_35b_a3b_nvfp4_fast
./serve.sh sglang unsloth_qwen36_27b_nvfp4
./serve.sh sglang unsloth_qwen36_35b_a3b_nvfp4_fast
```
The vLLM launchers use native MTP with four speculative tokens. The SGLang launchers use the pinned matching DFlash drafter with eight speculative tokens.

## bench/ contents

The 50k-decode benchmark had four redundant iterations; consolidated 2026-05-17
to one. `bench_50k.py` is the former `bench_true50k.py` (tokenize-verified
prompt size, per-run nonce to defeat prefix caching, aggregate stats) — the
correct version. The old char-estimate `bench_50k.py`, `bench_50k_decode.py`,
and `bench_50k_decode_v2.py` were deleted.

| File | Role |
|---|---|
| `bench_tps.py` | Short-prompt decode tok/s. Used by every sweep + `bench_suite.sh`. |
| `bench_50k.py` | Verified 50k-context decode tok/s. Used by `bench_suite.sh`. |
| `bench_50k_coding.py` | Sustained 50k decode on a coding workload. Used by `sweep_dflash.sh` + `fp8_coding_mtp_remote.sh`. |
| `bench_suite.sh` | Runs `bench_tps.py` + `bench_50k.py` back to back. |
| `snake_test_qwen_32b.py` | Not a benchmark — a pygame Snake script output by an old Qwen-32B run (Feb 2025). Kept as a sample. |

# Grouped DFlash K-normalization experiment

This directory is a clean backport of vLLM PR [#46761](https://github.com/vllm-project/vllm/pull/46761), commit `02a1f23711c5bdbff81eb8a610dde39e1141d036`, to the deployed vLLM source/API at `23002d3f368a5a24641301bc71e4ae15dae89a24` (`0.20.2rc1.dev206+g23002d3f3`). It changes only DFlash's context-K normalization path. The target model, draft model, quantization, cache layout, RoPE, and cache-write paths are unchanged.

The upstream PR modified the newer `csrc/libtorch_stable/layernorm_kernels.cu` optional-weight API. The deployed commit still implements the registered op in `csrc/layernorm_kernels.cu` with mandatory `torch::Tensor&` arguments. `0001-grouped-dflash-knorm.patch` ports the same indexing contract to that deployed API rather than copying incompatible stable-ABI code.

## Exact source anchors

Base repository: `https://github.com/vllm-project/vllm.git`

| Path | Git blob at base commit | Anchors before patch | Backport |
|---|---|---|---|
| `csrc/layernorm_kernels.cu` | `e617e45dc58b7d90a57efc26492e6da09f7ed348` | `rms_norm_kernel` line 14; weight argument line 22; rank-2/3/4 row addressing lines 27-44; `rms_norm` wrapper lines 192-235; launch lines 226-231 | Adds a zero-or-row `weight_stride`; selects weight by the outermost input index for ranks 2/3/4; validates 1-D and 2-D weights; passes the stride through the CUDA launch. A 1-D weight keeps stride zero and therefore preserves broadcast semantics. |
| `vllm/model_executor/models/qwen3_dflash.py` | `0587fa62bfb409e0eb5fa7c87c97d07dde6b75b3` | `_build_fused_kv_buffers` line 354; K weights lines 377-378; `precompute_and_store_context_kv` line 409; per-layer K norm lines 460-468 | Materializes one contiguous `[num_layers, head_dim]` weight tensor and replaces `L` custom-op launches with one launch over `[L, num_ctx, num_kv_heads, head_dim]`. |

The Dockerfile checks the full base commit and both Git blob IDs before applying the patch. A source mismatch fails the build before compilation.

## Artifacts

- `0001-grouped-dflash-knorm.patch`: C++/CUDA kernel and Qwen DFlash integration.
- `Dockerfile`: two-stage source-wheel build on `qwen-vllm:dflash`; the runtime stage starts from a fresh copy of the same production image and installs the patched wheel with `--no-deps`.
- `build.sh`: guarded build wrapper using a distinct candidate tag.
- `probe_grouped_knorm.py`: GPU equivalence probe covering the DFlash rank-4 layout, rank-3/rank-4 edge layouts, FP16/BF16/FP32, non-power-of-two hidden size, legacy 1-D broadcast behavior, and invalid grouped-weight shapes.
- `run_probe.sh`: guarded probe wrapper using a distinct container name.

## Build

From NVIDIA-Workbench WSL:

```bash
cd /home/workbench/inference/experiments/grouped-knorm
./build.sh
```

Defaults:

- base image: `qwen-vllm:dflash`
- candidate image: `qwen-vllm:dflash-grouped-knorm-02a1f237`
- CUDA architecture: `12.0` (Blackwell)
- build parallelism: `MAX_JOBS=8`, `NVCC_THREADS=2`

Optional resource/tag overrides do not change the production tag:

```bash
CANDIDATE_IMAGE=qwen-vllm:dflash-grouped-knorm-test \
MAX_JOBS=4 NVCC_THREADS=1 ./build.sh
```

The wheel is compiled against the base image's existing Torch/CUDA environment with `VLLM_USE_PRECOMPILED=0` and installed into the candidate with `--no-deps`. That prevents pip from replacing Torch 2.11 cu130, FlashInfer 0.6.8.post1, or CUTLASS DSL 4.4.2. The production image is only read as a base layer; it is not retagged or modified.

## Equivalence probe

The full probe requires a GPU and is intentionally separate from the image build:

```bash
cd /home/workbench/inference/experiments/grouped-knorm
./run_probe.sh
```

A short DFlash-layout-only check is:

```bash
./run_probe.sh --quick
```

Success requires bitwise identity (`torch.equal`) between the grouped launch and the former per-layer loop for every case, unchanged 1-D weight behavior, and `RuntimeError` for row-count, hidden-size, and weight-rank mismatches.

## Candidate serving

Use the candidate image in an experiment-specific invocation while preserving the exact production model arguments and the existing host patch mounts under `/home/workbench/inference/patches/vllm`. Do not edit `run_vllm_docker.sh` or the active serve script. The image and container must remain distinct, for example:

```bash
docker run --gpus all --rm \
  --name qwen-vllm-dflash-grouped-knorm-02a1f237 \
  <the same runtime mounts, environment, and ports as production> \
  qwen-vllm:dflash-grouped-knorm-02a1f237 \
  <the same model path and vLLM serve arguments as production>
```

The placeholder is deliberate: copy the active production invocation's model/mount/serve arguments verbatim so this experiment changes only the grouped K-normalization implementation. Do not reuse the production container name.

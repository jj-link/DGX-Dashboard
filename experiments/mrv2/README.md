# ModelRunnerV2 DFlash experiment

This directory is isolated from the production launcher and image. It does not modify `run_vllm_docker.sh`, the active serve scripts, `qwen-vllm:dflash`, the target model, or the DFlash model.

## Exact pins

| Component | Pin |
|---|---|
| Base image | `qwen-vllm:dflash` (read-only build parent) |
| Candidate image | `qwen-vllm:mrv2-a0f6d767-fi0614` |
| vLLM source and precompiled wheel | `a0f6d767e42acf09f94e7af2bacca7f4c268c264` (`0.23.1rc1.dev996+ga0f6d767e`) |
| FlashInfer Python and cubin packages | `0.6.14` |
| Target model | `nvidia/Qwen3.6-27B-NVFP4@0893e1606ff3d5f97a441f405d5fc541a6bdf404` |
| DFlash model | `z-lab/Qwen3.6-27B-DFlash@0919688658996800f86b895034249700e9481106` |

The vLLM pin is after all three required upstream changes:

| Upstream PR | Merged commit |
|---|---|
| `#44586` ModelRunnerV2 DFlash/full CUDA graphs | `0bae1d38480374365ad77bbea50be225237572ea` |
| `#46761` fused DFlash context-K RMSNorm | `02a1f23711c5bdbff81eb8a610dde39e1141d036` |
| `#46781` ModelRunnerV2 block verification | `db808b39614384a0349378268a46a1a0feabcec3` |

The pinned vLLM commit normally declares FlashInfer `0.6.13`; this experiment deliberately upgrades both `flashinfer-python` and `flashinfer-cubin` to `0.6.14`. `verify_image.py` makes a missing or substituted 0.6.14 package a build failure.

## Model support at the pin

Support exists in-tree at the exact vLLM pin:

- `Qwen3_5ForConditionalGeneration` is registered to `qwen3_5.Qwen3_5ForConditionalGeneration`. The target repository declares that architecture and `model_type: qwen3_5` despite the product name Qwen3.6.
- `DFlashDraftModel` is registered to `qwen3_dflash.DFlashQwen3ForCausalLM`. The pinned DFlash repository declares `DFlashDraftModel` and `model_type: qwen3`.
- `SpeculativeConfig` includes `dflash`, `rejection_sample_method=block`, draft revisions, and draft attention backend selection.
- ModelRunnerV2 routes DFlash through its draft speculator and `ModelCudaGraphManager`; PR `#44586` added the corresponding full-CUDA-graph path.

This is source-level availability, not a claim that this unbuilt experiment has passed a GPU launch.

## Reapplying the existing Qwen ModelOpt fixes

The four host fixes remain the source of truth under `../../patches/vllm`:

- `model_executor/layers/quantization/modelopt.py`
- `model_executor/models/qwen3_5.py`
- `model_executor/parameter.py`
- `model_executor/layers/vocab_parallel_embedding.py`

The Docker build does **not** copy those old files over modern vLLM or replay their textual diff. Static reconciliation against the exact pin found that modern vLLM already contains the three behavioral fixes: native `W4A16_NVFP4` mixed-precision dispatch with Qwen prefix/fused-projection lookup, quantization-aware Qwen `ParallelLMHead`, and vocabulary-loader dtype/scalar handling. Replaying the old `modelopt.py` would actually delete modern native W4A16 support. The fourth host delta only prints shapes immediately before the same existing loader exceptions, so it is recorded as diagnostic-only and is not forward-ported. `reconcile_modelopt_fixes.py` fingerprints all four supplied host files, asserts every required modern equivalent, syntax-checks the reconciled sources, and writes `/opt/modelopt-reapply/manifest.tsv`. A missing/changed host prerequisite, missing modern behavior, or syntax error fails the build explicitly.

## Build

Prerequisites: Docker Buildx, local production image `qwen-vllm:dflash`, network access to GitHub/Python wheel indexes/vLLM wheel storage, and all four host patch files above.

```bash
cd /home/workbench/inference/experiments/mrv2
./build.sh
```

`build.sh` uses the experiment directory as its small Docker context and passes `../../patches/vllm` as a read-only named BuildKit context. It loads the candidate under `qwen-vllm:mrv2-a0f6d767-fi0614`. The Dockerfile pins both the source checkout and `VLLM_PRECOMPILED_WHEEL_COMMIT` to prevent a source/binary commit mismatch.

Only the separate output tag is overrideable; base, source, wheel commit, and FlashInfer version remain fixed:

```bash
IMAGE=qwen-vllm:mrv2-a0f6d767-fi0614-test ./build.sh
```

## Launch variants

Both variants force `VLLM_USE_V2_MODEL_RUNNER=1`, use `FULL_DECODE_ONLY` CUDA graph mode, retain the Qwen3.6 NVFP4 target and Qwen3.6 DFlash drafter at exact Hub revisions, use FlashAttention for target/draft attention, and set probabilistic draft sampling so the rejection methods are directly comparable.

Standard rejection (container `vllm-mrv2-dflash-standard-a0f6d767`, host port `8001`):

```bash
cd /home/workbench/inference/experiments/mrv2
./run_standard.sh
```

Block rejection (container `vllm-mrv2-dflash-block-a0f6d767`, host port `8002`):

```bash
cd /home/workbench/inference/experiments/mrv2
./run_block.sh
```

The launchers default to offline, read-only Hugging Face cache access and fail before `docker run` unless both exact snapshots are present. For an explicit first download, permit network/cache writes for that invocation:

```bash
ALLOW_ONLINE=1 ./run_standard.sh
ALLOW_ONLINE=1 ./run_block.sh
```

Useful isolated overrides include `IMAGE`, `PORT`, `CONTAINER_NAME`, `HF_CACHE`, `GPU_MEM_UTIL`, `MAXLEN`, `NUM_SPEC`, `MAX_NUM_BATCHED`, `MAX_NUM_SEQS`, and `DETACH=1`. Standard and block use separate default container names, ports, and compilation-cache volumes.

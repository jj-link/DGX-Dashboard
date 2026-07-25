# MiMo-V2.5-NVFP4 on Spark Cluster - Research Tracker

## Goal
Get https://huggingface.co/lukealonso/MiMo-V2.5-NVFP4 working on the spark cluster (spark2-ts + spark3-ts, GB10 SM121).

## Key Facts From Model Card
- Model: lukealonso/MiMo-V2.5-NVFP4 (NVFP4 quant of XiaomiMiMo/MiMo-V2.5)
- 179B params, multimodal (text, images, audio, video)
- Quantized: only non-shared MoE expert MLP projections to NVFP4. Attention/dense MLPs/shared experts stay BF16
- Docker image: docker.io/lukealonso/sglang-cuda13-b12x (CUSTOM, contains custom kernels)
- Env: CUTE_DSL_ARCH="sm_120a"
- Requirements: "only supported on RTX 6000 (SM120)"
  - Minimum: 2x RTX PRO 6000 Blackwell 96GB
  - Recommended: 4x RTX PRO 6000 Blackwell 96GB
- SGLang-based, uses b12x backends for moe-runner, attention, mm-attention, fp4-gemm
- EAGLE speculative decoding (num-steps 3, topk 1, draft-tokens 4)
- KV cache dtype fp8_e4m3
- reasoning-parser mimo, tool-call-parser mimo
- 2x config: tp=2, mem-fraction=0.95, context-length=131072, cuda-graph-max-bs=4, max-running-requests=4, chunked-prefill=2048
- 4x config: tp=4, mem-fraction=0.85, no context-length limit, max-running-requests=8, chunked-prefill=8192

## Hardware Available
- WSL host: RTX PRO 6000 (96GB, SM120) + RTX 4090 (24GB) -- only 1 SM120 GPU, not enough alone
- spark2-ts: GB10 SoC, 128GB unified, SM121
- spark3-ts: GB10 SoC, 128GB unified, SM121
- Currently running DeepSeek-V4-Flash-DSpark on spark2+spark3 (DO NOT TOUCH)

## Core Problem
Model card says SM120 only. Spark cluster is SM121. Need to determine:
1. Can b12x kernels run on SM121?
2. Is there an sm_121a variant of CUTE_DSL_ARCH?
3. Has anyone gotten it working on GB10?
4. Can the WSL RTX PRO 6000 be used (only 1 GPU, need 2 minimum)?

## Research Questions
- [x] Q1: Read all community discussions for GB10/SM121 attempts (see below)
- [x] Q2: Inspect docker image lukealonso/sglang-cuda13-b12x for SM121 support
- [x] Q3: Check lukealonso's other repos for SM121 variants
- [x] Q4: Check SGLang source for b12x backend SM121 support
- [x] Q5: Check model files structure (shard count, sizes)
- [x] Q6: Is there a vLLM alternative? NO — discussion #9 confirms no vLLM support, community wants it
- [x] Q7: Can multi-node TP work across spark2+spark3 for this model?
- [x] Q8: Memory math: can 2x GB10 (256GB total) hold 183.5GB NVFP4 model?

## Things Tried
1. Read model card (full text via console)
2. Read community discussions #8, #9, #10, #11
3. Read SGLang GitHub issue #24321 (full comments)

## Findings

### Community Discussion Summary
- #11 (2 days ago): "2x RTX P6000BW produces garbage" — even on SM120, garbage after ~100 tokens. Links to SGLang #24321.
- #10 (May 24): TP4 doesn't work — AssertionError n=3392 not divisible by 128
- #9 (May 19): Community wants vLLM support — no response from lukealonso
- #8 (May 12): Kernel version issue — sglang-kernel needs >=0.4.2.post1, workaround Dockerfile provided

### SGLang Issue #24321 — ROOT CAUSE FOUND
- Title: "MiMo-V2.5 NVFP4 produces garbage tokens on consumer Blackwell (SM12.0a / RTX PRO 6000)"
- Root cause (found by lacvapps): Fused qkv_proj loader bug. Checkpoint stores QKV in plain [Q;K;V] order, but loader shards by blind row-chunking → corrupts attention on every rank.
- Fix requires THREE PRs + offline dequant:
  1. PR #28100: section-aware QKV loader (delegates to QKVParallelLinear)
  2. PR #28099: MIXED_PRECISION detection (content-based, not arch-based)
  3. PR #27916: expert routing fix (RoutingMethodType.DeepSeekV3)
  4. Offline dequant of 63 MXFP8 dense tensors to bf16 (weight_scale_inv u8 = UE8M0 exponent)
- Validated on B200 (SM100) TP4: GSM8K-200 greedy = 0.990 (vs 0.995 original FP8). EAGLE MTP accept 3.95-4.0.
- On SM120 (RTX PRO 6000): unit tests pass, but full serving needs 2+ GPUs (model is 183.5GB, >96GB single)
- Model on disk: 183.5 GB, 178.56B params

### Critical Question: Does lukealonso b12x image fix this?
- Model card updated 5/4/26 (day after issue opened) — "new docker image"
- b12x image uses CUSTOM backends: --moe-runner-backend b12x, --attention-backend b12x, --mm-attention-backend b12x, --fp4-gemm-backend b12x
- These are NOT the standard flashinfer_cutlass/triton backends that had the QKV bug
- The b12x backends may have their own weight loading path that doesn't have the bug
- Need to check if b12x image includes the QKV fix or uses a different codepath

### SM121 (GB10) Compatibility — FEASIBLE IN PRINCIPLE
- Model card says "only supported on RTX 6000 (SM120)"
- BUT: b12x README says "SM120/SM121 CuTe DSL kernel library" — explicitly supports SM121!
- pyproject.toml description says "SM120-only" (stale), but README says SM120/SM121
- b12x is a PURE PYTHON wheel (py3-none-any) — JIT-compiled via CuTe DSL at runtime (like Triton)
- b12x version 0.23.0 on PyPI, deps: torch>=2.12.0, nvidia-cutlass-dsl>=4.5.2, cuda-python, etc.
- CUTE_DSL_ARCH=sm_120a is used throughout — need to check if sm_121a exists or if sm_120a works on SM121
- require_sm120() in benchmarks does NOT actually check arch — just checks cuda.is_available()

### Docker Image Architecture — HARD BLOCKER (but workaround exists)
- docker.io/lukealonso/sglang-cuda13-b12x is AMD64 ONLY
- Spark nodes are aarch64 (ARM64) — confirmed: spark2 is aarch64, SM121
- The image CANNOT run on spark nodes directly
- WORKAROUND: b12x is pip-installable (pure Python), and all deps have aarch64 wheels:
  - nvidia-cutlass-dsl: pure Python (any)
  - nvidia-cutlass-dsl-libs-cu13: has manylinux_2_28_aarch64 wheels
  - nvidia-cutlass-dsl-libs-base: has manylinux_2_28_aarch64 wheels
  - cuda-python: needs checking but likely has aarch64
- Need to build a custom SGLang aarch64 image with b12x installed

### b12x GitHub Repo (github.com/lukealonso/b12x)
- Actively maintained (commits within minutes of research)
- 114 stars, 20 forks, 779 commits, 10 branches
- Source for all custom kernels: GEMM, attention, MoE, sparse MLA, residual, quantization, PCIe allreduce
- Has integration with sglang/vllm (b12x/integration/ directory)
- Kernel inventory includes: NVFP4 GEMM, MXFP8 linear, paged attention (BF16/FP8), MoE (FP4/W4A16/W4A8), sparse MLA, NSA/MSA indexer, PCIe one-shot allreduce
- Open issues: DeepSeek V4 Flash SGLang code request, MiniMax-M2.7 NVFP4 problem, etc.

### Model Files Structure
- 35 safetensors shards, ~5GB each, total 184GB
- config.json (6.18MB), hf_quant_config.json (4.23MB), amax_checkpoint.safetensors (8.62MB)
- audio_tokenizer directory (multimodal)
- tokenizer files (merges.txt 1.67MB)
- Model on disk: 183.5 GB, 178.56B params

### b12x Issue #10 — GB10/SM121/aarch64 CONFIRMED WORKING
- arcticjoe tried running DeepSeek-V4-Flash-NVFP4 on 2x DGX Spark (GB10, SM121, ARM64)
- Hit issue: public SGLang fork calls prepare_w4a16_mxfp4_native_weights, renamed to prepare_w4a16_modelopt_native_weights in b12x on May 25
- lukealonso's private SGLang fork has the updated glue
- SeedSource confirmed b12x kernels RUN CORRECTLY on GB10/SM121/aarch64 with ZERO changes:
  - NVFP4 dense GEMM: numerically exact vs FlashInfer CUTLASS FP4 reference (cos=1.0, max_abs=0)
  - Compressed sparse-MLA: all 10 cases correct (cos~0.99999)
  - Setup: pip install nvidia-cutlass-dsl==4.5.2 + b12x with torch 2.12 in CUDA-13 aarch64 container
  - cap (12, 1) detected automatically, zero kernel changes
  - pyproject "SM120-only" is stale; README "SM120/SM121" is accurate

### local-inference-lab/blackwell-llm-docker — BUILD RECIPE
- Source repo for all Docker images: github.com/local-inference-lab/blackwell-llm-docker
- Dockerfile.sglang-cu130: CUDA 13.0, torch 2.11, SGLang + b12x + PCIe allreduce
- Dockerfile.sglang-cu132: CUDA 13.2, torch 2.12 from source, SGLang + b12x
- All Dockerfiles target x86_64 + SM120a (TORCH_CUDA_ARCH_LIST="12.0a", CMAKE_CUDA_ARCHITECTURES="120a")
- Would need modification for aarch64 + SM121a
- Base image: nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04 (available for both amd64 and aarch64)
- Has patches/ directory with fix scripts for various issues

### local-inference-lab/sglang — SGLANG FORK WITH b12x INTEGRATION
- Full MiMo-V2.5 model support: mimo_v2.py, mimo_v2_nextn.py, mimo_mtp.py, mimo_audio.py, multimodal processors
- b12x backend registered in server_args.py: moe_runner, attention, fp4_gemm, mm_attention, nsa backends
- b12x integration in deepseek_v4.py, qwen2_moe.py, multimodal VIT runners
- mimo_v2.py uses generic moe_runner_backend dispatch (not hardcoded b12x calls)
- Model card launch flags all supported: --moe-runner-backend b12x, --attention-backend b12x, etc.

### MEMORY MATH — 2x GB10 (spark2 + spark3)
- Model on disk: 183.5 GB (NVFP4 quantized, 178.56B params)
- GB10 unified memory: 128GB per node, 256GB total across 2 nodes
- Model weights alone: ~92GB per node (183.5 / 2) with TP=2
- Per node after weights: 128 - 92 = 36GB for KV cache + CUDA graphs + overhead
- With mem-fraction-static 0.95 (from 2x config): 128 * 0.95 = 121.6GB available
  - Weights: ~92GB
  - Remaining: ~30GB for KV cache + graphs + EAGLE draft
- This is TIGHT but may work with:
  - cuda-graph-max-bs 4, max-running-requests 4 (from 2x config)
  - context-length 131072 (limited)
  - chunked-prefill-size 2048
- Alternative: 2x RTX PRO 6000 (192GB total, 96GB each) — model fits more comfortably

### WSL RTX PRO 6000 Option
- WSL has 1x RTX PRO 6000 (96GB, SM120) + 1x RTX 4090 (24GB)
- Model needs minimum 2x 96GB GPUs — WSL has only 1 Blackwell GPU
- RTX 4090 is SM89 (Ada) — does NOT support NVFP4
- Cannot run on WSL alone

## ACTION PLAN

### Path Forward: Build Custom aarch64 SGLang+b12x Docker Image for Spark Cluster

The path is clear but requires significant build effort:

1. Build a custom Docker image on spark2 (aarch64) based on Dockerfile.sglang-cu130
   - Change TORCH_CUDA_ARCH_LIST from "12.0a" to "12.1a"
   - Change CMAKE_CUDA_ARCHITECTURES from "120a" to "121a"
   - Change FLASHINFER_CUDA_ARCH_LIST from "12.0a" to "12.1a"
   - Set CUTE_DSL_ARCH=sm_121a (confirmed supported by test_cutlass_45_provides_sm121a_blockscaled_mma)

2. Install from local-inference-lab/sglang fork (has b12x integration + MiMo-V2.5 support)
   - Use local-inference-lab/sglang@main as the SGLang source
   - Install b12x from PyPI (pip install b12x) — pure Python, JIT-compiled

3. Download model to spark2: hf download lukealonso/MiMo-V2.5-NVFP4 (184GB, 35 shards)

4. Launch with TP=2 across spark2+spark3 (multi-node, like existing DSpark setup)
   - Use NCCL_NET=Socket + NCCL_IB_DISABLE=1 (same as DSpark)
   - GLOO_SOCKET_IFNAME=TP_SOCKET_IFNAME=enp1s0f1np1
   - Launch rank 0 on spark2, rank 1 on spark3 separately
   - Port: different from 8888 (DSpark uses 8888)

5. Known risks:
   - QKV loader bug (SGLang #24321): May need PRs #28100 + #28099 + #27916 + MXFP8 dequant
     - BUT: b12x uses custom backends, may have different loading path
     - The model card was updated 5/4/26 with "new docker image" — may already be fixed in lukealonso's image
   - sglang-kernel version issue (discussion #8): Need sglang-kernel >= 0.4.2.post1
   - Build will take several hours (torch from source, FlashInfer, SGLang, etc.)
   - Currently running DSpark on spark2+spark3 — must not interfere

## BUILD STATUS (started July 1, 2026)

### Docker Image Build
- Dockerfile: /home/jjlink/inference/mimo-v25-build/Dockerfile.mimo-aarch64
- Based on: Dockerfile.sglang-cu130 from local-inference-lab/blackwell-llm-docker
- Modifications: SM120a -> SM121a, x86_64-linux -> aarch64-linux, CUTE_DSL_ARCH=sm_121a
- SGLang fork: voipmonitor/sglang blackwell-cu130 branch (has b12x integration + MiMo-V2.5)
- Build log: /tmp/mimo-docker-build.log on spark2
- Status: BLOCKED — spark2 power spike shutdown during sgl-kernel compilation
- Notes:
  - First attempt failed with transient HTTP 520 from ports.ubuntu.com (apt-get)
  - Second attempt running successfully
  - torch 2.11.0+cu130 aarch64 wheel confirmed available
  - torch 2.12.1+cu130 aarch64 downloaded and installed
  - FlashInfer compiled from source successfully
  - sgl-kernel CMake configuration done (223s), Ninja build started
  - Build got to step 22/24 (sgl-kernel compilation) before spark2 power spike
  - Docker build cache should have all stages up to sgl-kernel build cached

### Model Download
- Path: ~/models/hub/models--lukealonso--MiMo-V2.5-NVFP4/
- Log: /tmp/mimo-download.log on spark2
- Status: BLOCKED — spark2 down, was at ~88GB of 184GB
- Notes:
  - Had stale lock file issue — fixed by killing old process and removing locks
  - Download restarted successfully

### NEXT STEPS (when spark2 is back)
1. Check if spark2 rebooted (uptime, dmesg for power events)
2. Check if docker build survived (docker images, docker build cache)
3. If build cache intact: restart build from sgl-kernel step
4. If build cache lost: restart full build (should be faster due to downloaded packages)
5. Restart model download (hf download resumes from cache)
6. Apply clock limit to prevent future power spikes: sudo nvidia-smi -lgc 2125

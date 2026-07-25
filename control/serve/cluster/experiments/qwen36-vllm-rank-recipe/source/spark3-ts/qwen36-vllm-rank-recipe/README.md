# Qwen3.6-27B-NVFP4 vLLM rank recipe

Persistent recipe for serving `nvidia/Qwen3.6-27B-NVFP4` on Spark2+Spark3 GB10 with vLLM OpenAI API, TP=2, Docker, DSpark-style rank0/rank1 launch discipline, and DSpark-style RDMA/RoCE/NCCL settings.

Final target exclusions: no Ray, no MiMo image, no MiMo patches, no SGLang, no single-node fallback, no temporary deployment scripts.

## Recipe path

Remote authoritative path on both nodes:

- `/home/jjlink/inference/qwen36-vllm-rank-recipe`

Local control copy:

- `/home/workbench/inference/serve/cluster/qwen36-vllm-rank-recipe`

## Runtime

Image:

- `vllm/vllm-openai:v0.21.0`
- Expected vLLM: `0.21.0`
- Expected commit: `ad7125a431e176d4161099480a66f0169609a690`

The first tested official release image, `vllm/vllm-openai:latest` / v0.24.0 / `ee0da84ab9e04ac7610e28580af62c365e898389`, loaded the checkpoint and initialized NCCL over IB but failed the strict non-Ray rank path with `AssertionError: collective_rpc should not be called on follower node`.

Updated hypothesis: the DSpark rank method is likely tied to older pre-regression vLLM multiprocessing behavior. We therefore moved the baseline back to official v0.21.0 before attempting source/backport work. If v0.21.0 can do rank launch but lacks enough ModelOpt/Qwen support, build a custom image from this older baseline plus targeted ModelOpt/Qwen loader backports rather than starting from v0.24/nightly.

If rank launch is unstable, the next intended step is a custom vLLM source image from a newer commit that preserves the same scripts/env/flags. Do not pivot to Ray as the final path.

## Exact mounts

Each container uses:

- `/home/jjlink/models/hub/hub:/models/hub:ro`
- `/home/jjlink/inference/qwen36-vllm-rank-recipe:/workspace/recipe:ro`
- `/dev/infiniband:/dev/infiniband`

The model path inside the container is:

- `/models/hub/models--nvidia--Qwen3.6-27B-NVFP4`

## Exact vLLM shape

- rank0 container: Spark2, `--node-rank 0`, `VLLM_HOST_IP=10.0.0.1`
- rank1 container: Spark3, `--node-rank 1`, `--headless`, `VLLM_HOST_IP=10.0.0.2`
- `--nnodes 2`
- `--master-addr 10.0.0.1`
- `--master-port 25000`
- `--tensor-parallel-size 2`
- `--distributed-executor-backend mp`
- `--enforce-eager` diagnostic/final-candidate flag: added after the first successful headless-mp startup passed `/v1/models` and short generation but stalled during long generation in the compile/autotune/CUDA-graph path with one node busy and the other idle.
- `--no-enable-flashinfer-autotune` diagnostic/final-candidate flag: added after `--enforce-eager` disabled compile/CUDA graphs but startup still entered repeated FlashInfer fp8_gemm autotuning.
- `--served-model-name qwen36-27b-nvfp4`
- API: `http://spark2-ts:8888/v1`

## Commands

From the local control copy:

```bash
cd /home/workbench/inference/serve/cluster/qwen36-vllm-rank-recipe
./start.sh
./status.sh
./logs.sh both 100
./verify.sh
./stop.sh
```

From Spark2 after sync:

```bash
cd /home/jjlink/inference/qwen36-vllm-rank-recipe
./status.sh
./logs.sh both 100
./stop.sh
```

## Caveats

- This intentionally tests the strict non-Ray vLLM `mp` multi-node path. Rank1 must be launched with `--headless`, matching the DSpark compose recipe. Without `--headless`, official vLLM v0.23/v0.24 loads the checkpoint and initializes NCCL/RoCE but crashes on the follower with `collective_rpc should not be called on follower node`.
- The complete model cache is the double-hub directory. The single-hub path exists but is incomplete/stale on these nodes.

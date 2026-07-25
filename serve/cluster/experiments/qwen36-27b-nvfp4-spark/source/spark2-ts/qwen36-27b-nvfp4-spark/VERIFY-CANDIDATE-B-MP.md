# Candidate B verification: official vLLM 0.23.0 + non-Ray mp/rank TP=2

Recipe path:

- Local: `/home/workbench/inference/serve/cluster/qwen36-vllm-rank-recipe`
- Remote: `/home/jjlink/inference/qwen36-vllm-rank-recipe`

Runtime:

- Image: `vllm/vllm-openai:v0.23.0`
- Executor: `--distributed-executor-backend mp`
- Rank shape: Spark2 rank0, Spark3 rank1, `--nnodes 2`, `--node-rank 0/1`, `--master-addr 10.0.0.1`, `--master-port 25000`
- Model: `/models/hub/models--nvidia--Qwen3.6-27B-NVFP4`
- Served name: `qwen36-27b-nvfp4`

## What passed

The official v0.23.0 mp/rank candidate successfully reached model inspection and ModelOpt/NVFP4 detection on both nodes:

- `Resolved architecture: Qwen3_5ForConditionalGeneration`
- `Detected ModelOpt fp8 checkpoint (quant_algo=FP8)`
- `Detected ModelOpt NVFP4 checkpoint (quant_algo=NVFP4)`
- `Detected ModelOpt NVFP4 checkpoint (quant_algo=W4A16_NVFP4)`

It also initialized NCCL over RoCE on both ranks:

- Spark2/rank0: `NCCL INFO NET/IB : Using [0]rocep1s0f1:1/RoCE [RO]; OOB enp1s0f1np1:10.0.0.1`
- Spark3/rank1: `NCCL INFO NET/IB : Using [0]rocep1s0f1:1/RoCE [RO]; OOB enp1s0f1np1:10.0.0.2`

Both workers loaded checkpoint shards and reached TP assignment:

- rank0: `rank 0 in world size 2 ... TP rank 0`
- rank1: `rank 1 in world size 2 ... TP rank 1`

## What failed

The follower/rank1 process crashed during EngineCore startup with the known official-vLLM V1 mp executor failure:

```text
AssertionError: collective_rpc should not be called on follower node
```

Stack location:

```text
vllm/v1/engine/core.py -> _initialize_kv_caches
vllm/v1/executor/abstract.py -> get_kv_cache_specs
vllm/v1/executor/multiproc_executor.py -> collective_rpc
```

The rank1 container exited and rank0 never exposed a healthy API.

## Decision

Candidate B is not viable with official vLLM v0.23.0 as-is. This matches the earlier v0.24.0 mp/rank failure pattern. Per the project rule, do not keep looping on official mp/rank without a concrete executor patch.

The next path is Candidate C: a custom/runtime-engineering path. The likely approach is either:

1. Patch the newer vLLM V1 mp executor so follower nodes do not call leader-only `collective_rpc`, or
2. Start from the older DSpark-compatible vLLM lineage and surgically backport the minimum Qwen3.6/ModelOpt NVFP4 loader support.

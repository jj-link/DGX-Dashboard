# Final verification: Qwen3.6-27B-NVFP4 Spark2+Spark3 vLLM deployment

Final recipe path:

- Local: `/home/workbench/inference/serve/cluster/qwen36-27b-nvfp4-spark`
- Remote: `/home/jjlink/inference/qwen36-27b-nvfp4-spark`

## Final runtime

- Image: `vllm/vllm-openai:v0.23.0`
- vLLM: `0.23.0`
- vLLM build commit: `91df0fad4dc98a67c7659d9dbd915245d5c43d96`
- Executor: non-Ray `mp`
- Rank shape:
  - Spark2/rank0: API/head process
  - Spark3/rank1: `--headless` worker
- API: `http://spark2-ts:8888/v1`
- Served model: `qwen36-27b-nvfp4`
- Model path in container: `/models/hub/models--nvidia--Qwen3.6-27B-NVFP4`
- Host model root: `/home/jjlink/models/hub/hub`

## Critical launch flags

```text
--tensor-parallel-size 2
--distributed-executor-backend mp
--nnodes 2
--node-rank 0/1
--master-addr 10.0.0.1
--master-port 25000
--enforce-eager
--no-enable-flashinfer-autotune
--enable-prefix-caching
--enable-chunked-prefill
--no-async-scheduling
```

Rank1/Spark3 must include `--headless`. Without it, official vLLM v0.23/v0.24 starts a follower API/EngineCore and crashes with:

```text
AssertionError: collective_rpc should not be called on follower node
```

`--enforce-eager` and `--no-enable-flashinfer-autotune` are currently required for healthy long generation. Without them, the official runtime can pass short tests but stall in compile/autotune/CUDA-graph paths with one node busy and the other idle.

## Networking/RDMA discipline

The final recipe preserves the GB10 DSpark/MiMo RoCE discipline:

```text
--network host
--ipc host
--device=/dev/infiniband:/dev/infiniband
NCCL_NET=IB
NCCL_IB_DISABLE=0
NCCL_IB_HCA=rocep1s0f1
NCCL_SOCKET_IFNAME=enp1s0f1np1
GLOO_SOCKET_IFNAME=enp1s0f1np1
TP_SOCKET_IFNAME=enp1s0f1np1
NCCL_IB_GID_INDEX=3
NCCL_CROSS_NIC=1
NCCL_CUMEM_ENABLE=0
NCCL_NVLS_ENABLE=0
NCCL_NET_GDR_LEVEL=LOC
NCCL_SOCKET_FAMILY=AF_INET
NCCL_IGNORE_CPU_AFFINITY=1
NCCL_DEBUG=INFO
```

## Verification results

Final verification artifacts in this directory:

- `models.json`
- `short.json`
- `long.json`
- `gpu-sample.tsv`

### `/v1/models`

Passed. Returned model id:

```text
qwen36-27b-nvfp4
```

with `max_model_len=262144`.

### Short generation

Passed. `short.json` is an HTTP 200 chat completion response for model `qwen36-27b-nvfp4`.

### Long generation

Passed. `long.json` is an HTTP 200 chat completion response for model `qwen36-27b-nvfp4`.

vLLM logs during the long request showed steady generation throughput around 16.5-16.8 tokens/s and request completion:

```text
POST /v1/chat/completions HTTP/1.1 200 OK
Engine 000: Avg generation throughput: 16.6 tokens/s, Running: 1 reqs
```

### Both-node GPU activity during long generation

Recorded from `gpu-sample.tsv`:

```text
spark2-ts: samples=64 active_samples=35 max_util=89 avg_util=47.8
spark3-ts: samples=64 active_samples=33 max_util=88 avg_util=43.5
```

This satisfies the primary success criterion: both Spark2 and Spark3 GPUs stayed active during long generation. The earlier Spark2-busy/Spark3-idle and Spark3-busy/Spark2-idle failure modes were not present in the final run.

### NCCL/RoCE logs

Spark2/rank0:

```text
NCCL INFO NCCL_IB_HCA set to rocep1s0f1
NCCL INFO NET/IB : Using [0]rocep1s0f1:1/RoCE [RO]; OOB enp1s0f1np1:10.0.0.1
```

Spark3/rank1:

```text
NCCL INFO NCCL_IB_HCA set to rocep1s0f1
NCCL INFO NET/IB : Using [0]rocep1s0f1:1/RoCE [RO]; OOB enp1s0f1np1:10.0.0.2
```

## Commands

Start:

```bash
cd /home/workbench/inference/serve/cluster/qwen36-27b-nvfp4-spark && ./start.sh
```

Status:

```bash
cd /home/workbench/inference/serve/cluster/qwen36-27b-nvfp4-spark && ./status.sh
```

Logs:

```bash
cd /home/workbench/inference/serve/cluster/qwen36-27b-nvfp4-spark && ./logs.sh both 200
```

Verify:

```bash
cd /home/workbench/inference/serve/cluster/qwen36-27b-nvfp4-spark && ./verify.sh
```

Stop:

```bash
cd /home/workbench/inference/serve/cluster/qwen36-27b-nvfp4-spark && ./stop.sh
```

# Candidate A verification: official vLLM 0.24.0 + Ray TP=2

Recipe path:

- Local: `/home/workbench/inference/serve/cluster/qwen36-27b-nvfp4-spark`
- Remote: `/home/jjlink/inference/qwen36-27b-nvfp4-spark`

Runtime:

- Image: `vllm/vllm-openai:latest`
- Observed vLLM: `0.24.0`
- Observed Ray: `2.56.0`
- Executor: `--distributed-executor-backend ray`
- Model: `/models/hub/models--nvidia--Qwen3.6-27B-NVFP4`
- Served name: `qwen36-27b-nvfp4`

## What passed

`/v1/models` passed and returned `qwen36-27b-nvfp4` with `max_model_len=262144`.

Short generation passed after setting:

- `GPU_MEMORY_UTILIZATION=0.75`
- `VLLM_USE_RAY_WRAPPED_PP_COMM=0`
- `CUDA_VISIBLE_DEVICES=0`

Short generation evidence:

- `verify-output/20260706-004431/short-response.json` contains HTTP 200 response for model `qwen36-27b-nvfp4`.
- Short GPU monitor showed both nodes active briefly:
  - Spark2: reached 92-93% GPU util.
  - Spark3: reached 91-92% GPU util.

RoCE/NCCL evidence in vLLM logs included both nodes using IB/RoCE:

- Spark2/rank0: `NCCL INFO NET/IB : Using [0]rocep1s0f1:1/RoCE [RO]; OOB enp1s0f1np1:10.0.0.1`
- Spark3/rank1: `NCCL INFO NET/IB : Using [0]rocep1s0f1:1/RoCE [RO]; OOB enp1s0f1np1:10.0.0.2`

## What failed

Long generation failed the primary success criterion.

During the long generation monitor in `verify-output/20260706-004431`:

- `gpu-long-spark2-ts.csv`: 141 samples, max 96%, average ~95.3%, last 10 samples all 96%.
- `gpu-long-spark3-ts.csv`: 141 samples, max 91%, average ~0.65%, last 10 samples all 0%.
- `long-response.json` stayed empty while Spark2 remained busy and Spark3 remained idle.

This is exactly the unhealthy behavior the goal warned about: Spark2 busy while Spark3 idles during generation.

Candidate A therefore is not acceptable as the final deployment despite successful `/v1/models`, model load, NCCL/RoCE initialization, and short generation.

## Decision

Do not declare Ray candidate healthy. Move to Candidate B/C evaluation:

1. Candidate B: official/newer vLLM with non-Ray mp/rank executor. If it fails with `collective_rpc should not be called on follower node`, document and do not loop without an executor patch.
2. Candidate C: custom old-lineage vLLM image/backport if official Ray and official mp both fail verification.

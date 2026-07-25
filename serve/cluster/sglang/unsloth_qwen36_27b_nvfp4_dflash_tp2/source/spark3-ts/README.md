# Qwen3.6-27B Unsloth NVFP4 + SGLang DFlash TP=2 on Spark2/Spark3

Persistent recipe for serving `unsloth/Qwen3.6-27B-NVFP4` with SGLang and the `z-lab/Qwen3.6-27B-DFlash` drafter across two GB10 nodes.

- Local recipe: `/home/workbench/inference/serve/cluster/qwen36-27b-unsloth-nvfp4-sglang-dflash`
- Remote recipe: `/home/jjlink/inference/qwen36-27b-unsloth-nvfp4-sglang-dflash`
- Spark2/rank0: OpenAI-compatible API on port 8888
- Spark3/rank1: worker node

The final DFlash candidate preserves the Spark1 DFlash baseline flags, except `DISABLE_CUDA_GRAPH=1` is now the default because both DFlash and non-DFlash TP=2 stalled after prefill/decode with CUDA graphs enabled on GB10: `DFLASH`, draft model `z-lab/Qwen3.6-27B-DFlash`, draft tokens `20`, draft window `4096`, `extra_buffer`, FlashInfer, `mem-fraction-static=0.85`, `max-running-requests=8`, `cuda-graph-max-bs=16`, `chunked-prefill-size=2048`.

It also preserves the GB10 RoCE/RDMA/NCCL discipline: host networking, `/dev/infiniband`, `NCCL_NET=IB`, `NCCL_IB_DISABLE=0`, `NCCL_IB_HCA=rocep1s0f1`, `NCCL_SOCKET_IFNAME/GLOO_SOCKET_IFNAME/TP_SOCKET_IFNAME=enp1s0f1np1`, `NCCL_IB_GID_INDEX=3`, and `NCCL_CROSS_NIC=1`.

Default mode is the final verified SGLang + DFlash + TP=2 deployment. The failed CUDA-graph candidate and the fix are documented in `DFLASH-INCOMPATIBILITY.md`. Set `ENABLE_DFLASH=0 ./start.sh` only for the non-DFlash diagnostic fallback.

Commands:

```bash
cd /home/workbench/inference/serve/cluster/qwen36-27b-unsloth-nvfp4-sglang-dflash
./start.sh          # final SGLang + DFlash + TP=2 deployment
./start-dflash.sh   # same as start.sh; explicit DFlash entry point
ENABLE_DFLASH=0 ./start.sh   # non-DFlash diagnostic fallback
./status.sh
./verify.sh
./logs.sh both 200
./stop.sh
```

Important: `start.sh` refuses to use incomplete HF cache blobs by checking for `model.safetensors` on both Spark2 and Spark3 before launch. It removes only this recipe's container name, not unrelated vLLM/SGLang servers.

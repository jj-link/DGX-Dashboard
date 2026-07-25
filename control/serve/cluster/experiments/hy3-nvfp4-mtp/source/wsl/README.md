# Hy3-295B NVFP4 MTP on Spark2+Spark3

Source of truth: https://github.com/tonyd2wild/Hy3-295B-NVFP4-MTP-2x-DGX-Spark at commit c6e2947b048ced0f0caa2596960cc580c5294493.

This directory contains the Spark2/Spark3 operational wrapper. It keeps the upstream recipe shape and changes only cluster-specific values:

- head: Spark2, fabric IP 10.0.0.1
- worker: Spark3, fabric IP 10.0.0.2
- fabric interface: enp1s0f1np1
- RoCE HCA: rocep1s0f1, GID index 3
- API port: 8888, served model: hy3
- model path on both nodes: /home/jjlink/models/hy3-nvfp4-w4a16, mounted read-only as /models
- runtime image: hy3-vllm-ray:v0.23.0, built locally from vllm/vllm-openai:v0.23.0 plus Ray 2.49.2. The base image was verified to include vllm.model_executor.models.hy_v3; upstream used a local vLLM 0.23.1 custom image that is not present on this cluster.

Weights:

- Hugging Face repo: kodelow/Hy3-NVFP4-W4A16
- upstream size: about 181GB, 99 safetensor shards
- layout expected by scripts: flat local-dir download at /home/jjlink/models/hy3-nvfp4-w4a16 on both Spark nodes

Launch flags:

```text
vllm serve /models \
  --served-model-name hy3 --host 0.0.0.0 --port 8888 \
  --tensor-parallel-size 2 \
  --distributed-executor-backend ray \
  --max-model-len 131072 \
  --max-num-seqs 6 \
  --kv-cache-dtype fp8_e4m3 \
  --gpu-memory-utilization 0.90 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
  --trust-remote-code --enforce-eager
```

Operations from Spark2:

```bash
cd /home/jjlink/inference/serve/cluster/hy3-nvfp4-mtp
./build-image.sh
./start.sh
./status.sh
./logs.sh head 200
./logs.sh worker 200
./verify.sh
./stop.sh
```

Verification criteria:

1. `./status.sh` shows `hy3-head` on Spark2 and `hy3-worker` on Spark3, Ray has 2 active nodes, and both GB10s have model memory allocated.
2. `curl http://127.0.0.1:8888/v1/models` returns model id `hy3`.
3. `./verify.sh` completes a 512-token chat completion.
4. Head logs contain MTP/speculative configuration and acceptance/throughput lines.
5. During a long generation, Spark2 and Spark3 both show GPU activity/memory use.

Measured throughput:

- Pending final run after the 181GB model download completes on Spark2 and is rsynced to Spark3.
- Upstream reference for the same recipe: 21.8 tok/s single-stream, 59.7 tok/s aggregate over 6-way concurrency with enforce-eager + MTP spec-1.

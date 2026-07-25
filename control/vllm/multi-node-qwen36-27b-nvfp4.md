# Multi-Node vLLM Configuration for Qwen3.6-27B-NVFP4

## Research Summary

### 1. Official vLLM Docker Hub Tags

Only `vllm/vllm-openai:latest` and `vllm/vllm-openai:gemma` exist on Docker Hub.
**No versioned tags** (0.7.3, 0.8.0, etc.) are published. This means you cannot
pull a specific older version from Docker Hub directly.

### 2. V0 vs V1 Engine Timeline

| Version | `VLLM_USE_V1` default | Multi-node MP executor |
|---------|----------------------|----------------------|
| ≤0.7.3  | OFF (V0 engine)     | Ray only (mp is single-node) |
| 0.8.4+  | ON (V1 engine)      | Ray only (mp is single-node) |
| 0.21.0  | OFF (V0 engine)     | Ray only (mp is single-node) |

**Key finding**: `VLLM_USE_V1=0` env var disables V1 engine and forces V0.
This works in BOTH v0.8.4+ and `:latest` images.

### 3. Multi-Node MP Executor

The `MultiprocessingDistributedExecutor` (mp backend) explicitly hardcodes
`127.0.0.1` and states: *"Multiprocessing-based executor does not support
multi-node setting."*

**For multi-node TP, you MUST use the Ray executor:**
```
--distributed-executor-backend ray
```

The `external_launcher` executor was removed before v0.7.3 and is not available.

### 4. Working Configuration

**Option A (Recommended): Use existing qwen-vllm image**
Your `Dockerfile.vllm-latest` already builds vLLM 0.21.0 from the NGC GB10
base (`nvcr.io/nvidia/vllm:26.03.post1-py3`) — this is the correct V0 engine
version with proper GB10 CUDA 13 / SM121 support.

Build: `docker build -f docker/Dockerfile.vllm-latest -t vllm-multi-node:0.21 .`

**Option B: Use `:latest` with V1 disabled**
Set `VLLM_USE_V1=0` to force V0 engine. Requires Ray to be installed.

### 5. Multi-Node Launch Configuration

The existing DeepSeek V4 Flash script (`serve_deepseek_v4_flash_tp2.sh`) shows
the working pattern:

```bash
# Rank 0 (spark2-ts, 10.0.0.1):
export RANK=0
vllm serve "$MODEL" \
  --tensor-parallel-size 2 \
  --distributed-executor-backend mp \
  --host 0.0.0.0 --port 8000

# Rank 1 (spark3-ts, 10.0.0.2):
export RANK=1
vllm serve "$MODEL" \
  --tensor-parallel-size 2 \
  --distributed-executor-backend mp \
  --host 0.0.0.0 --port 8000
```

**CRITICAL**: The existing script uses `--distributed-executor-backend mp` which
works for the DSpark model because DSpark applies a patch that enables multi-node
mp support. For the standard vLLM binary, this would fail. You need to verify
whether the NGC base image or your custom build has this patch applied.

### 6. Environment Variables for NCCL (IB Transport)

```bash
export NCCL_IB_HCA=rocep1s0f1
export NCCL_IB_GID_INDEX=3
export NCCL_CROSS_NIC=1
export NCCL_DEBUG=INFO
export MASTER_ADDR=10.0.0.1
export MASTER_PORT=29500
export WORLD_SIZE=2
export RANK=<0 or 1>
```

### 7. Known Pitfalls

1. **V1 engine bug**: `collective_rpc should not be called on follower node` —
   ensure `VLLM_USE_V1=0` is set.

2. **MP executor is single-node**: The standard `--distributed-executor-backend mp`
   does NOT work for multi-node unless you have a patched vLLM (like the DSpark
   build). Use `--distributed-executor-backend ray` for standard vLLM, OR verify
   your NGC image has the multi-node mp patch.

3. **Ray dependency**: The `:latest` image may not include Ray. The NGC-based
   `Dockerfile.vllm-latest` may include it — check with `pip list | grep ray`.

4. **IB through Docker**: Use `--network host` mode so containers can access
   RDMA devices directly.

5. **Model path**: Mount the HF cache as read-only:
   `-v /home/jjlink/models:/models:ro` with `HF_HOME=/models`

6. **NVFP4 specific**: May need `--disable-custom-all-reduce` (same as existing
   serve scripts) and `--mamba-ssm-cache-dtype float32`.

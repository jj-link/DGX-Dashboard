ARG BASE_IMAGE=vllm/vllm-openai@sha256:251eba5cc7c12fed0b75da22a9240e582b1c9e39f6fbc064f86781b963bd814f
FROM ${BASE_IMAGE}
COPY patches/vllm-0.24-dflash-depth-matrix/vllm /usr/local/lib/python3.12/dist-packages/vllm
COPY patches/vllm-0.24-align-upstream/vllm/v1/worker/mamba_utils.py /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/mamba_utils.py
COPY patches/vllm-0.24-align-upstream/vllm/v1/worker/gpu/model_states/mamba_hybrid.py /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/model_states/mamba_hybrid.py
COPY patches/vllm-0.24-align-upstream/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py
COPY patches/vllm-0.24-align-upstream/vllm/model_executor/layers/mamba/ops/causal_conv1d.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/mamba/ops/causal_conv1d.py
COPY patches/vllm-0.24-align-upstream/vllm/model_executor/layers/fla/ops/fused_recurrent.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fla/ops/fused_recurrent.py
COPY patches/vllm-0.24-align-upstream/vllm/model_executor/layers/fla/ops/fused_sigmoid_gating.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fla/ops/fused_sigmoid_gating.py
COPY patches/vllm-0.24-align-upstream/vllm/model_executor/kernels/linear/nvfp4/cutlass.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/kernels/linear/nvfp4/cutlass.py
COPY patches/vllm-0.24-align-upstream/vllm/model_executor/warmup/deep_gemm_warmup.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/warmup/deep_gemm_warmup.py
COPY patches/vllm-0.24-align-upstream/vllm/forward_context.py /usr/local/lib/python3.12/dist-packages/vllm/forward_context.py
COPY patches/vllm-0.24-align-upstream/vllm/config/compilation.py /usr/local/lib/python3.12/dist-packages/vllm/config/compilation.py
COPY patches/vllm-0.24-align-upstream/vllm/model_executor/models/qwen3_next.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_next.py
COPY patches/vllm-0.24-align-upstream/vllm/model_executor/models/qwen3_5.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_5.py
COPY patches/vllm-0.24-align-upstream/vllm/v1/core/sched/scheduler.py /usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py
COPY patches/vllm-0.24-align-upstream/vllm/v1/attention/backends/gdn_attn.py /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/gdn_attn.py
COPY patches/vllm-0.24-align-upstream/vllm/v1/attention/backends/flash_attn.py /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/flash_attn.py
COPY patches/vllm-0.24-align-upstream/vllm/platforms/interface.py /usr/local/lib/python3.12/dist-packages/vllm/platforms/interface.py
COPY patches/vllm-0.24-align-upstream/vllm/v1/core/kv_cache_utils.py /usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_utils.py
COPY moe_configs/ /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/configs/
COPY patches/qwen3-dflash-v1-mixed.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_dflash.py
COPY patches/vllm-0.24-align-upstream/vllm/v1/spec_decode/dflash.py /usr/local/lib/python3.12/dist-packages/vllm/v1/spec_decode/dflash.py
COPY patches/vllm-0.24-align-upstream/vllm/v1/cudagraph_dispatcher.py /usr/local/lib/python3.12/dist-packages/vllm/v1/cudagraph_dispatcher.py
COPY patches/vllm-0.24-align-upstream/vllm/v1/worker/gpu_model_runner.py /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py
RUN python3 - <<'PY'
from pathlib import Path
path = Path('/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py')
source = path.read_text()
old = 'stride = torch.empty(target_shape).stride()'
assert source.count(old) == 1
path.write_text(source.replace(old, 'stride = torch.empty(target_shape, device="meta").stride()'))
PY

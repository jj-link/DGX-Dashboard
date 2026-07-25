ARG BASE_IMAGE=vllm/vllm-openai@sha256:251eba5cc7c12fed0b75da22a9240e582b1c9e39f6fbc064f86781b963bd814f
FROM ${BASE_IMAGE}
COPY vllm-0.24-dflash-depth-matrix/vllm /usr/local/lib/python3.12/dist-packages/vllm
COPY vllm-0.24-align-upstream/vllm/v1/worker/mamba_utils.py /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/mamba_utils.py
COPY vllm-0.24-align-upstream/vllm/v1/worker/gpu/model_states/mamba_hybrid.py /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/model_states/mamba_hybrid.py
COPY vllm-0.24-align-upstream/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py
COPY vllm-0.24-align-upstream/vllm/config/compilation.py /usr/local/lib/python3.12/dist-packages/vllm/config/compilation.py
COPY vllm-0.24-align-upstream/vllm/model_executor/models/qwen3_next.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_next.py
COPY vllm-0.24-align-upstream/vllm/model_executor/models/qwen3_5.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_5.py
COPY vllm-0.24-align-upstream/vllm/v1/core/sched/scheduler.py /usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py
COPY vllm-0.24-align-upstream/vllm/v1/attention/backends/gdn_attn.py /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/gdn_attn.py
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

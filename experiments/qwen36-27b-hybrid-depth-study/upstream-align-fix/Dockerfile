FROM vllm:qwen27-reference-uva
COPY mamba_utils.py /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/mamba_utils.py
COPY mamba_hybrid.py /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/model_states/mamba_hybrid.py
COPY interface.py /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/model_states/interface.py
COPY warmup.py /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/warmup.py
COPY model_runner.py /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/model_runner.py

ARG BASE_IMAGE=vllm/vllm-openai@sha256:a230095847e93bd4df9888b33dab956fa9504537b828a23657d2b26fed57b5c9
FROM ${BASE_IMAGE}
ENV VLLM_USE_PRECOMPILED=1
RUN python3 -m pip install --no-cache-dir --no-deps --force-reinstall     "vllm @ git+https://github.com/vllm-project/vllm.git@23002d3f368a5a24641301bc71e4ae15dae89a24"

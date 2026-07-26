ARG BASE_IMAGE=ghcr.io/anemll/dspark-vllm-gx10@sha256:a83948492cf13df455170fb42885f5ef4db54fefe0feff0f841ecbff464ac9d8
FROM ${BASE_IMAGE}
RUN python3 -m pip install --no-cache-dir --upgrade --break-system-packages \
    "flashinfer-python==0.6.15.dev20260712" \
    "flashinfer-cubin==0.6.15.dev20260712" \
    "flashinfer-jit-cache==0.6.15.dev20260712" \
    --extra-index-url https://flashinfer.ai/whl/nightly/ \
    --extra-index-url https://flashinfer.ai/whl/nightly/cu130/

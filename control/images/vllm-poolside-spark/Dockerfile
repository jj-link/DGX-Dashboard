ARG BASE_IMAGE=ghcr.io/anemll/dspark-vllm-gx10@sha256:a83948492cf13df455170fb42885f5ef4db54fefe0feff0f841ecbff464ac9d8
FROM ${BASE_IMAGE}
RUN python3 -m pip install --no-cache-dir --upgrade --break-system-packages --no-deps \
    "flashinfer-python==0.6.13" \
    && python3 -c "import importlib.metadata as m; assert m.version('flashinfer-python') == '0.6.13'; assert m.version('flashinfer-cubin') == '0.6.13'"

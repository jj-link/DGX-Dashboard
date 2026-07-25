# syntax=docker/dockerfile:1.7

ARG CUDA_DEVEL_IMAGE=nvidia/cuda:13.0.2-devel-ubuntu22.04
ARG RUNTIME_IMAGE=qwen-vllm:dflash

FROM ${CUDA_DEVEL_IMAGE} AS wheel-builder

ARG VLLM_REPOSITORY=https://github.com/vllm-project/vllm.git
ARG VLLM_REVISION=23002d3f368a5a24641301bc71e4ae15dae89a24
ARG PYTHON_VERSION=3.12
ARG MAX_JOBS=2
ARG NVCC_THREADS=8
ARG TORCH_CUDA_ARCH_LIST=12.0

ENV DEBIAN_FRONTEND=noninteractive \
    UV_HTTP_TIMEOUT=500 \
    UV_INDEX_STRATEGY=unsafe-best-match \
    UV_LINK_MODE=copy \
    VLLM_TARGET_DEVICE=cuda \
    VLLM_DOCKER_BUILD_CONTEXT=1 \
    VLLM_USE_PRECOMPILED= \
    MAX_JOBS=${MAX_JOBS} \
    NVCC_THREADS=${NVCC_THREADS} \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ccache \
        curl \
        git \
        gcc-10 \
        g++-10 \
        libibverbs-dev \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 110 \
        --slave /usr/bin/g++ g++ /usr/bin/g++-10 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && /root/.local/bin/uv venv /opt/venv --python ${PYTHON_VERSION}
ENV PATH=/opt/venv/bin:/root/.local/bin:${PATH}

WORKDIR /workspace
RUN git clone --filter=blob:none "${VLLM_REPOSITORY}" . \
    && git checkout --detach "${VLLM_REVISION}" \
    && test "$(git rev-parse HEAD)" = "${VLLM_REVISION}"

COPY 0001-padded-nvfp4-output.patch /tmp/0001-padded-nvfp4-output.patch
COPY probe_weights_padding_cols.py /tmp/probe_weights_padding_cols.py

# Probe the deployed source contract before patching, require a clean apply, then
# probe the post-patch data flow before any expensive compilation begins.
RUN python /tmp/probe_weights_padding_cols.py --phase pre /workspace \
    && git apply --check /tmp/0001-padded-nvfp4-output.patch \
    && git apply /tmp/0001-padded-nvfp4-output.patch \
    && python /tmp/probe_weights_padding_cols.py --phase post /workspace

# This intentionally performs a real CUDA source build. VLLM_USE_PRECOMPILED is
# empty, so nvfp4_quant_kernels.cu is compiled into the wheel rather than being
# inherited from an unpatched precompiled extension.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv/bin/python \
        -r requirements/build/cuda.txt \
        --extra-index-url https://download.pytorch.org/whl/cu130

RUN --mount=type=cache,target=/root/.cache/ccache \
    --mount=type=cache,target=/root/.cache/uv \
    python setup.py bdist_wheel --dist-dir=/wheelhouse --py-limited-api=cp38 \
    && python -c "from pathlib import Path; wheels=list(Path('/wheelhouse').glob('vllm-*.whl')); assert len(wheels)==1, wheels"

FROM ${RUNTIME_IMAGE} AS padded-quant

ARG VLLM_REVISION=23002d3f368a5a24641301bc71e4ae15dae89a24
LABEL org.opencontainers.image.title="qwen-vllm padded NVFP4 quant experiment" \
      org.opencontainers.image.revision="${VLLM_REVISION}" \
      org.opencontainers.image.description="DFlash vLLM with PR 42774 padded NVFP4 CUDA output"

COPY --from=wheel-builder /wheelhouse/vllm-*.whl /tmp/wheelhouse/
RUN python -m pip install --no-cache-dir --no-deps --force-reinstall /tmp/wheelhouse/vllm-*.whl \
    && rm -rf /tmp/wheelhouse \
    && python -c "import importlib.metadata as m; v=m.version('vllm'); assert v.startswith('0.20.2rc1.dev206+g23002d3f3'), v; print(v)"

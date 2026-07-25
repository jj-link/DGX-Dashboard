# syntax=docker/dockerfile:1.7

ARG CUDA_DEVEL_IMAGE=nvidia/cuda:13.0.2-devel-ubuntu22.04
ARG BASE_IMAGE=qwen-vllm:dflash

FROM ${CUDA_DEVEL_IMAGE} AS build

ARG VLLM_BASE_COMMIT=23002d3f368a5a24641301bc71e4ae15dae89a24
ARG MAX_JOBS=2
ARG NVCC_THREADS=8

ENV DEBIAN_FRONTEND=noninteractive \
    UV_HTTP_TIMEOUT=500 \
    UV_INDEX_STRATEGY=unsafe-best-match \
    UV_LINK_MODE=copy \
    VLLM_TARGET_DEVICE=cuda \
    VLLM_DOCKER_BUILD_CONTEXT=1 \
    VLLM_USE_PRECOMPILED= \
    SETUPTOOLS_SCM_PRETEND_VERSION=0.20.2rc1.dev206+g23002d3f3 \
    TORCH_CUDA_ARCH_LIST=12.0 \
    MAX_JOBS=${MAX_JOBS} \
    NVCC_THREADS=${NVCC_THREADS}

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ccache curl git gcc-10 g++-10 libibverbs-dev \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 110 \
        --slave /usr/bin/g++ g++ /usr/bin/g++-10 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && /root/.local/bin/uv venv /opt/venv --python 3.12
ENV PATH=/opt/venv/bin:/root/.local/bin:${PATH}

WORKDIR /opt/grouped-knorm/vllm
RUN git init . \
    && git remote add origin https://github.com/vllm-project/vllm.git \
    && git fetch --depth=1 origin "${VLLM_BASE_COMMIT}" \
    && git checkout --detach FETCH_HEAD \
    && test "$(git rev-parse HEAD)" = "${VLLM_BASE_COMMIT}" \
    && test "$(git hash-object csrc/layernorm_kernels.cu)" = \
        e617e45dc58b7d90a57efc26492e6da09f7ed348 \
    && test "$(git hash-object vllm/model_executor/models/qwen3_dflash.py)" = \
        0587fa62bfb409e0eb5fa7c87c97d07dde6b75b3

COPY 0001-grouped-dflash-knorm.patch /tmp/0001-grouped-dflash-knorm.patch
RUN git apply --check /tmp/0001-grouped-dflash-knorm.patch \
    && git apply /tmp/0001-grouped-dflash-knorm.patch

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv/bin/python \
        -r requirements/build/cuda.txt \
        --extra-index-url https://download.pytorch.org/whl/cu130

RUN --mount=type=cache,target=/root/.cache/ccache \
    --mount=type=cache,target=/root/.cache/uv \
    python setup.py bdist_wheel --dist-dir=/wheelhouse --py-limited-api=cp38

FROM ${BASE_IMAGE} AS runtime
USER root
COPY --from=build /wheelhouse /tmp/grouped-knorm-wheel
RUN python3 -m pip install \
        --no-cache-dir --no-deps --force-reinstall \
        /tmp/grouped-knorm-wheel/vllm-*.whl \
    && rm -rf /tmp/grouped-knorm-wheel
COPY probe_grouped_knorm.py /opt/grouped-knorm/probe_grouped_knorm.py

LABEL org.opencontainers.image.title="qwen-vllm DFlash grouped K-normalization experiment" \
      org.opencontainers.image.description="Backport of vLLM PR 46761 commit 02a1f237 to deployed commit 23002d3f3" \
      org.opencontainers.image.base.name="qwen-vllm:dflash" \
      org.opencontainers.image.revision="02a1f23711c5bdbff81eb8a610dde39e1141d036"

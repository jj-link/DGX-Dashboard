ARG BASE_IMAGE=qwen-vllm:dflash
FROM ${BASE_IMAGE}

USER root

ARG VLLM_COMMIT=23002d3f368a5a24641301bc71e4ae15dae89a24
ARG CUTLASS_COMMIT=e8ecfad75b44d1ad56264f5001d877e9e47fe080
ARG MAX_JOBS=2
ARG NVCC_THREADS=1

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
       build-essential ccache cmake git ninja-build \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --no-cache-dir "cmake>=3.26.1"

# The runtime image contains a precompiled-wheel environment switch. Clear it
# so the patched CUTLASS extension is actually rebuilt, and expose the CUDA
# development files installed by Torch's cu130 packages.
RUN ln -sfn libnvrtc.so.13 /usr/local/cuda/targets/x86_64-linux/lib/libnvrtc.so
ENV VLLM_USE_PRECOMPILED=0 \
    CPATH=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include \
    LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib \
    LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib

RUN git clone --filter=blob:none https://github.com/vllm-project/vllm.git /opt/vllm-src \
    && git -C /opt/vllm-src checkout "${VLLM_COMMIT}" \
    && test "$(git -C /opt/vllm-src rev-parse HEAD)" = "${VLLM_COMMIT}"

RUN git init /opt/cutlass-src \
    && git -C /opt/cutlass-src remote add origin https://github.com/NVIDIA/cutlass.git \
    && git -C /opt/cutlass-src fetch --depth=1 origin "${CUTLASS_COMMIT}" \
    && git -C /opt/cutlass-src checkout --detach FETCH_HEAD \
    && test "$(git -C /opt/cutlass-src rev-parse HEAD)" = "${CUTLASS_COMMIT}"

COPY patch_cutlass_blockwise_small_n.py /tmp/patch_cutlass_blockwise_small_n.py
RUN python3 /tmp/patch_cutlass_blockwise_small_n.py

COPY vllm-cutlass46-sm120-fp8-tiles.patch /tmp/vllm-cutlass46-sm120-fp8-tiles.patch
RUN git -C /opt/vllm-src apply --check /tmp/vllm-cutlass46-sm120-fp8-tiles.patch \
    && git -C /opt/vllm-src apply /tmp/vllm-cutlass46-sm120-fp8-tiles.patch
COPY generated_tactics.hpp /tmp/generated_tactics.hpp
RUN install -m 0644 /tmp/generated_tactics.hpp \
    /opt/vllm-src/csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_blockwise_sm120_fp8_tactics.hpp

ENV VLLM_CUTLASS_SRC_DIR=/opt/cutlass-src \
    TORCH_CUDA_ARCH_LIST=12.0f \
    VLLM_TARGET_DEVICE=cuda \
    MAX_JOBS=${MAX_JOBS} \
    NVCC_THREADS=${NVCC_THREADS}

WORKDIR /opt/vllm-src
RUN --mount=type=cache,target=/root/.cache/ccache \
    python3 use_existing_torch.py \
    && python3 -m pip install --no-build-isolation --no-deps --force-reinstall .

# Keep the production base's Torch and FlashInfer dependency set. The source
# install above uses --no-deps and builds only vLLM at the deployed DFlash SHA.
RUN python3 -c "import torch, vllm; assert torch.__version__.startswith('2.11'); assert vllm.__version__.startswith('0.20.2rc1.dev206')"

WORKDIR /workspace

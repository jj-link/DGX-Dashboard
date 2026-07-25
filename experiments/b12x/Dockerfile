# syntax=docker/dockerfile:1.7
# Experiment-only image: DFlash base + FlashInfer 0.6.14 b12x dense NVFP4.
ARG BASE_IMAGE=qwen-vllm:dflash@sha256:c3c88e6777b1107fcc4a0509fab6dabf9e2be2d34e00f4455e42255d2eee02cb
FROM ${BASE_IMAGE}

ARG SITE_PACKAGES=/usr/local/lib/python3.12/dist-packages
LABEL org.opencontainers.image.title="qwen-vllm b12x FlashInfer 0.6.14 candidate" \
      org.opencontainers.image.base.digest="sha256:c3c88e6777b1107fcc4a0509fab6dabf9e2be2d34e00f4455e42255d2eee02cb" \
      org.opencontainers.image.source.vllm="0.20.2rc1.dev206+g23002d3f3" \
      org.opencontainers.image.source.vllm-pr="https://github.com/vllm-project/vllm/pull/40082" \
      org.opencontainers.image.source.flashinfer="https://github.com/flashinfer-ai/flashinfer/releases/tag/v0.6.14"

COPY experiments/b12x/requirements.lock /opt/b12x/requirements.lock
RUN python3 -m pip install \
      --no-cache-dir \
      --no-deps \
      --force-reinstall \
      --require-hashes \
      -r /opt/b12x/requirements.lock

# Refuse to apply the API adaptation to any vLLM build other than the deployed
# 0.20.2rc1.dev206+g23002d3f3 source snapshot.
COPY experiments/b12x/base-vllm.sha256 /opt/b12x/base-vllm.sha256
COPY experiments/b12x/0001-b12x-nvfp4.patch /opt/b12x/0001-b12x-nvfp4.patch
RUN cd "${SITE_PACKAGES}" \
    && sha256sum -c /opt/b12x/base-vllm.sha256 \
    && git apply --check /opt/b12x/0001-b12x-nvfp4.patch \
    && git apply /opt/b12x/0001-b12x-nvfp4.patch

# Preserve the exact Qwen3.6 NVFP4 compatibility overlays used by production,
# but bake them into this isolated candidate instead of changing host launchers.
COPY patches/vllm /opt/b12x/model-overrides
COPY experiments/b12x/model-overrides.sha256 /opt/b12x/model-overrides.sha256
RUN cd /opt/b12x/model-overrides \
    && sha256sum -c /opt/b12x/model-overrides.sha256 \
    && install -m 0644 model_executor/layers/quantization/modelopt.py "${SITE_PACKAGES}/vllm/model_executor/layers/quantization/modelopt.py" \
    && install -m 0644 model_executor/models/qwen3_5.py "${SITE_PACKAGES}/vllm/model_executor/models/qwen3_5.py" \
    && install -m 0644 model_executor/parameter.py "${SITE_PACKAGES}/vllm/model_executor/parameter.py" \
    && install -m 0644 model_executor/layers/vocab_parallel_embedding.py "${SITE_PACKAGES}/vllm/model_executor/layers/vocab_parallel_embedding.py"

COPY experiments/b12x/verify_install.py /opt/b12x/verify_install.py
ENV VLLM_NVFP4_GEMM_BACKEND=flashinfer-b12x
RUN python3 /opt/b12x/verify_install.py

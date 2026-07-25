# syntax=docker/dockerfile:1.7
ARG BASE_IMAGE=qwen-vllm:dflash
ARG VLLM_COMMIT=a0f6d767e42acf09f94e7af2bacca7f4c268c264
ARG FLASHINFER_VERSION=0.6.14

FROM ${BASE_IMAGE} AS candidate
ARG VLLM_COMMIT
ARG FLASHINFER_VERSION

LABEL org.opencontainers.image.title="Qwen DFlash ModelRunnerV2 experiment" \
      org.opencontainers.image.source="https://github.com/vllm-project/vllm" \
      org.opencontainers.image.revision="${VLLM_COMMIT}" \
      io.flashinfer.version="${FLASHINFER_VERSION}" \
      io.vllm.experiment="mrv2"

ENV VLLM_USE_PRECOMPILED=1 \
    VLLM_PRECOMPILED_WHEEL_COMMIT=${VLLM_COMMIT} \
    VLLM_PRECOMPILED_WHEEL_VARIANT=cu130

RUN test -d /usr/local/lib/python3.12/dist-packages/vllm \
    || { echo 'fatal: qwen-vllm:dflash does not contain the expected Python 3.12 vLLM installation' >&2; exit 31; }
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# VLLM_PRECOMPILED_WHEEL_COMMIT makes setup.py fetch extension and Rust
# artifacts built from this exact source commit instead of silently using the
# current main wheel. Dependency resolution retains the commit's required
# Python packages; torch remains pinned by that commit to 2.11.0/cu130.
RUN python3 -m pip install --no-cache-dir --upgrade \
      "vllm @ git+https://github.com/vllm-project/vllm.git@${VLLM_COMMIT}" \
    && python3 -m pip install --no-cache-dir --upgrade \
      "flashinfer-python==${FLASHINFER_VERSION}" \
      "flashinfer-cubin @ https://github.com/flashinfer-ai/flashinfer/releases/download/v0.6.14/flashinfer_cubin-0.6.14-py3-none-any.whl#sha256=7bbed9f3851b59f3f6f6cb344810775bbce8a1c012ecf9a502bebd91cfb6433e" \
      "flashinfer-jit-cache @ https://github.com/flashinfer-ai/flashinfer/releases/download/v0.6.14/flashinfer_jit_cache-0.6.14%2Bcu130-cp39-abi3-manylinux_2_28_x86_64.whl#sha256=22b666dffe4cca7de0ea67668c16c0be819e8a67cdb34c404cf89abeb3d2b510"

# The old host overlays predate native W4A16_NVFP4 support in this pin and
# cannot safely be replayed as whole-file or textual patches. Fingerprint the
# four supplied overlays, then require their behavioral equivalents in modern
# vLLM. Missing or changed prerequisites and missing upstream equivalents are
# hard build failures. The reconciliation manifest records why each old delta
# is superseded or intentionally not forwarded.
COPY --from=modelopt-fixes /model_executor /opt/modelopt-fixes/vllm/model_executor
COPY reconcile_modelopt_fixes.py /opt/mrv2/reconcile_modelopt_fixes.py
RUN python3 /opt/mrv2/reconcile_modelopt_fixes.py \
      /opt/modelopt-fixes \
      /usr/local/lib/python3.12/dist-packages \
      /opt/modelopt-reapply

COPY verify_image.py /opt/mrv2/verify_image.py
RUN VLLM_EXPECTED_COMMIT="${VLLM_COMMIT}" \
    FLASHINFER_EXPECTED_VERSION="${FLASHINFER_VERSION}" \
    python3 /opt/mrv2/verify_image.py

ENV VLLM_USE_V2_MODEL_RUNNER=1
WORKDIR /workspace

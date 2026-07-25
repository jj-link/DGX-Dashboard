FROM lmsysorg/sglang@sha256:af911a303a12516adf23ab8bb89c8fdf161ec0ceafc278a436fa111f5c118988

ARG SGLANG_SOURCE_COMMIT=7de33ce806c12664b647604d61cf1403d2d18013
ARG SGLANG_PR_TIP=6affa83fa3ab675ffbf16e1eb6b833992d9d388b
ARG PATCH_SHA256=838341d55916a1d0f2d4dd819c0fef1ddb4b2b8f03490b2d32b8c7375343f50d
ARG FLASHINFER_VERSION=0.6.14
ARG SGL_DEEP_GEMM_VERSION=0.1.4.post1

LABEL org.opencontainers.image.title="Unsloth Qwen3.6 mixed FP8/NVFP4 SGLang" \
      org.opencontainers.image.source="https://github.com/sgl-project/sglang" \
      org.opencontainers.image.revision="${SGLANG_SOURCE_COMMIT}" \
      org.opencontainers.image.base.digest="sha256:af911a303a12516adf23ab8bb89c8fdf161ec0ceafc278a436fa111f5c118988" \
      io.unsloth.sglang.pr-tip="${SGLANG_PR_TIP}" \
      io.unsloth.sglang.patch-sha256="${PATCH_SHA256}" \
      io.unsloth.sglang.flashinfer-version="${FLASHINFER_VERSION}" \
      io.unsloth.sglang.deep-gemm-version="${SGL_DEEP_GEMM_VERSION}"

WORKDIR /sgl-workspace/sglang
RUN python3 -m pip install --no-cache-dir --upgrade \
        "flashinfer_python[cu13]==${FLASHINFER_VERSION}" \
        "flashinfer-cubin==${FLASHINFER_VERSION}" \
        "flashinfer-jit-cache==${FLASHINFER_VERSION}" \
        --extra-index-url https://flashinfer.ai/whl \
        --extra-index-url https://flashinfer.ai/whl/cu130
RUN python3 -m pip install --no-cache-dir --upgrade \
        "sgl-deep-gemm==${SGL_DEEP_GEMM_VERSION}"

RUN git fetch --no-tags --depth=1 origin "${SGLANG_SOURCE_COMMIT}" \
    && git checkout --detach "${SGLANG_SOURCE_COMMIT}" \
    && test "$(git rev-parse HEAD)" = "${SGLANG_SOURCE_COMMIT}"
RUN python3 -m pip install --no-cache-dir --no-build-isolation --no-deps \
        --editable ./python

COPY 0001-compressed-tensors-per-group-format.patch /tmp/sglang.patch
RUN echo "${PATCH_SHA256}  /tmp/sglang.patch" | sha256sum --check - \
    && git apply --check /tmp/sglang.patch \
    && git apply /tmp/sglang.patch \
    && rm /tmp/sglang.patch

COPY test_compressed_tensors_mixed_precision.py /tmp/test_compressed_tensors_mixed_precision.py
COPY test_runtime_smoke.py /tmp/test_runtime_smoke.py
COPY test_qwen35_quantized_load.py /tmp/test_qwen35_quantized_load.py
COPY test_dflash_quantized_head.py /tmp/test_dflash_quantized_head.py
RUN python3 -m pytest -q \
        /tmp/test_compressed_tensors_mixed_precision.py \
        /tmp/test_qwen35_quantized_load.py \
        /tmp/test_dflash_quantized_head.py \
        /tmp/test_runtime_smoke.py \
    && rm \
        /tmp/test_compressed_tensors_mixed_precision.py \
        /tmp/test_qwen35_quantized_load.py \
        /tmp/test_dflash_quantized_head.py \
        /tmp/test_runtime_smoke.py

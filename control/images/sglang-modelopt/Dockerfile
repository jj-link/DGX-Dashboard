ARG BASE_IMAGE=lmsysorg/sglang@sha256:42194170546745092e74cd5f81ad32a7c6e944c7111fe7bf13588152277ff356
FROM ${BASE_IMAGE}
ENV FLASHINFER_DISABLE_VERSION_CHECK=1
RUN python3 -m pip install --no-cache-dir --no-deps --force-reinstall     "sglang @ git+https://github.com/sgl-project/sglang.git@b34706ba8c8fe679888d8e9752c4faacbc80e444#subdirectory=python"

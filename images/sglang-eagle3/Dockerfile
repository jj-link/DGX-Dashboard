ARG BASE_IMAGE=lmsysorg/sglang@sha256:42194170546745092e74cd5f81ad32a7c6e944c7111fe7bf13588152277ff356
FROM ${BASE_IMAGE}
COPY serve/sglang/patch_sglang_eagle3.py /tmp/patch_sglang_eagle3.py
RUN python3 /tmp/patch_sglang_eagle3.py && rm /tmp/patch_sglang_eagle3.py

ARG BASE_IMAGE=nvcr.io/nvidia/vllm@sha256:13e327dad79e6e417f6687fec2ba76b0386d597082ec0ee003c1e964ec6ad0e7
FROM ${BASE_IMAGE}
ENV VLLM_USE_PRECOMPILED=1
RUN python3 -m pip install --no-cache-dir --upgrade --break-system-packages     "vllm @ git+https://github.com/vllm-project/vllm.git@ad7125a431e176d4161099480a66f0169609a690"     "transformers==5.8.1"

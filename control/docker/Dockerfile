# Pinned vLLM image for serving Qwen3.6 models on RTX PRO 6000 (SM120, WSL2).
#
# Why a Dockerfile and not just `docker run vllm/vllm-openai`:
#   - Pins the exact vLLM/torch/flashinfer dependency set so a poisoned or
#     yanked transitive PyPI package can't silently change what you run.
#   - Gives you one place to bake in any future site-packages patches
#     (currently NONE — the SM120 NVFP4 patches were reverted; the real
#     levers are runtime flags like --moe-backend marlin).
#
# Pin to a digest, not a floating tag, for real supply-chain protection.
# Get the digest with:  docker buildx imagetools inspect vllm/vllm-openai:<tag>
ARG VLLM_IMAGE=vllm/vllm-openai:latest
FROM ${VLLM_IMAGE}

# The official image's entrypoint is the API server. We override at `docker
# run` time with bash so the existing serve_*.sh scripts drive the launch
# (they set env + exec vllm serve themselves). Nothing to install here yet.
#
# If we ever need to re-apply a site-packages patch, do it here, e.g.:
#   COPY patches/ /patches/
#   RUN python - <<'PY' ... PY

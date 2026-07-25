FROM vllm/vllm-openai:latest

# Official vLLM 0.24.0 image loads NVIDIA Qwen3.6 ModelOpt NVFP4, but does not
# ship Ray in this local image. Bake Ray into a reproducible cluster image.
RUN python3 -m pip install --no-cache-dir "ray[default]==2.56.0"

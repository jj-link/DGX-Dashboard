#!/bin/bash
# Serve Gemma-4-12B-it with MTP (Multi-Token Prediction) on GPU 1 (RTX 4090)
# Configuration: F16 MTP heads, n_max=4 (fastest benchmarked)

docker run -d --name llama-gemma4-12b --gpus device=1 -e CUDA_VISIBLE_DEVICES=0 -p 127.0.0.1:8001:8080 -v /home/workbench/models/hub/unsloth/gemma-4-12b-it-GGUF:/models ghcr.io/ggml-org/llama.cpp:server-cuda -m /models/gemma-4-12b-it-UD-Q4_K_XL.gguf --host 0.0.0.0 --port 8080 --n-gpu-layers 999 --cache-ram 0 --no-warmup --spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-model /models/MTP/gemma-4-12b-it-F16-MTP.gguf

#!/usr/bin/env bash
# Serve the base model + abliterated LoRA adapter via vLLM's OpenAI-compatible API.
#
# Usage:
#   chmod +x serve_abliterated.sh
#   HF_TOKEN=<your_token> ./serve_abliterated.sh
#
# The server listens on http://localhost:8000
# Call it with the model name "abliterated" to use the LoRA adapter,
# or "meta-llama/Llama-2-7b-chat-hf" for the unmodified base model.

set -euo pipefail

ADAPTER_HOST_DIR="$(cd "$(dirname "$0")" && pwd)"
ADAPTER_CONTAINER_DIR="/adapters"
HF_CACHE_DIR="${HOME}/.cache/huggingface"

docker run --gpus all --ipc=host \
    -v "${HF_CACHE_DIR}:/root/.cache/huggingface" \
    -v "${ADAPTER_HOST_DIR}:${ADAPTER_CONTAINER_DIR}" \
    -p 8000:8000 \
    -e HF_TOKEN="${HF_TOKEN:?HF_TOKEN environment variable is required}" \
    vllm/vllm-openai:latest \
        --model meta-llama/Llama-2-7b-chat-hf \
        --enable-lora \
        --lora-modules abliterated="${ADAPTER_CONTAINER_DIR}/llama-2-7b-chat-hf-abliterated-lora" \
        --max-lora-rank 16

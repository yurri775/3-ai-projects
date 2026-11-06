#!/usr/bin/env bash
#
# Serve an open-source model with vLLM on a free Google Colab T4 GPU.
#
# In Colab:  Runtime → Change runtime type → T4 GPU, then run this in a cell:
#   !bash serve_colab.sh
#
# Llama-3.2-1B fits comfortably on a T4 (~2.5 GB in fp16). Rule of thumb:
# ~2 GB per billion parameters in fp16, plus room for the KV cache.

set -euo pipefail

MODEL="${MODEL:-unsloth/Llama-3.2-1B-Instruct}"
PORT="${PORT:-8000}"

echo "Installing vLLM ..."
pip install -q vllm

echo "Serving ${MODEL} on port ${PORT} ..."
# T4 has no bf16 support, so use half precision. Cap context to keep KV cache small.
nohup vllm serve "${MODEL}" \
  --dtype half \
  --max-model-len 4096 \
  --port "${PORT}" > vllm.log 2>&1 &

echo "Waiting for the server to come up ..."
until curl -s "localhost:${PORT}/health" >/dev/null 2>&1; do sleep 2; done

echo "Server is up. OpenAI-compatible endpoint: http://localhost:${PORT}/v1"
echo "Sample of exposed metrics:"
curl -s "localhost:${PORT}/metrics" | grep -E "time_to_first_token|inter_token|generation_tokens" | head

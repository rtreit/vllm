#!/bin/bash
# Script to build and test vLLM with CodeXEmbedModel2B support on CPU

# Exit on errors
set -e

# Build the Docker image with CPU support
echo "Building vLLM CPU Docker image..."
docker build -f docker/Dockerfile.cpu -t vllm-cpu-test --target vllm-dev .

# Run a container with the code mounted
echo "Running test container with your code..."
docker run --rm -it \
  -v $(pwd):/workspace/vllm \
  --name vllm-test-container \
  vllm-cpu-test \
  bash -c "cd /workspace/vllm && \
           pip install -e . && \
           python test_codex_embed_cpu.py --model hf-internal-testing/tiny-random-GemmaForCausalLM && \
           echo 'Basic test complete. To test with the actual model, run: python test_codex_embed_cpu.py --model SFR/SFR-Embedding-Code-2B_R'"

echo "Test complete!"

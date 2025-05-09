#!/bin/bash
# Script to build and deploy vLLM with CodeXEmbedModel2B support to k8s GPU pool

# Exit on errors
set -e

# Build the Docker image for GPU testing
echo "Building vLLM GPU test Docker image..."
docker build -f Dockerfile.gpu-test -t vllm-gpu-test:latest .

# Tag and push the image to your container registry
# Uncomment and modify these lines with your registry info
# REGISTRY="your-registry.io/username"
# docker tag vllm-gpu-test:latest $REGISTRY/vllm-gpu-test:latest
# docker push $REGISTRY/vllm-gpu-test:latest

echo "Deploying to Kubernetes GPU pool..."
# Uncomment this line when ready to deploy
# kubectl apply -f k8s-gpu-test.yaml

echo "Setup complete! You can test your model with:"
echo "kubectl port-forward service/vllm-codex-embed-service 8000:8000"
echo "And then run a test client: python test_api_client.py"

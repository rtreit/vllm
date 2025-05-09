# Testing CodeXEmbedModel2B Implementation

This document provides instructions for testing the `CodeXEmbedModel2B` implementation for the SFR-Embedding-Code-2B_R model in vLLM.

## CPU Testing (Local Development)

### Option 1: Using Docker Compose

The easiest way to test on CPU is using the provided Docker Compose setup:

```bash
# Build and run the CPU test container
docker-compose -f docker-compose-cpu-test.yml up --build
```

This will:
1. Build a CPU-compatible Docker image
2. Install your vLLM implementation with the CodeXEmbedModel2B support
3. Run the basic and comprehensive tests with a small test model

### Option 2: Using the Test Script

```bash
# Make the script executable
chmod +x test_on_cpu.sh

# Run the test script
./test_on_cpu.sh
```

## Testing with the Actual SFR Model

To test with the actual SFR-Embedding-Code-2B_R model:

```bash
# Using Docker Compose (modify the command in docker-compose-cpu-test.yml first to use the real model)
docker-compose -f docker-compose-cpu-test.yml up

# Or using Python directly (if you have vLLM installed)
python test_codex_embed_comprehensive.py --model SFR/SFR-Embedding-Code-2B_R
```

Note: The actual model is 2B parameters, so CPU inference will be very slow.

## GPU Testing (Kubernetes)

For proper performance testing, deploy to a Kubernetes cluster with GPU support:

```bash
# Edit deploy_to_k8s.sh to configure your container registry
nano deploy_to_k8s.sh

# Make the script executable
chmod +x deploy_to_k8s.sh

# Run the deployment script
./deploy_to_k8s.sh
```

After deployment:

1. Forward the service port:
   ```bash
   kubectl port-forward service/vllm-codex-embed-service 8000:8000
   ```

2. Test with the OpenAI-compatible API:
   ```bash
   python -c '
   import openai

   client = openai.Client(base_url="http://localhost:8000/v1", api_key="dummy")

   code_snippets = [
       "def hello_world():\n    print(\"Hello, world!\")",
       "function greet() {\n    console.log(\"Hello, world!\");\n}"
   ]

   response = client.embeddings.create(input=code_snippets, model="SFR/SFR-Embedding-Code-2B_R")

   print(f"Embedding dimensions: {len(response.data[0].embedding)}")
   print(f"First 5 values of embedding 1: {response.data[0].embedding[:5]}")
   '
   ```

## Expected Results

The `CodeXEmbedModel2B` model should:

1. Correctly load the model architecture
2. Generate embeddings with the expected dimensionality
3. Produce normalized embeddings (L2 norm ≈ 1.0)
4. Show reasonable cosine similarity between related code snippets
5. Successfully use last token pooling

## Troubleshooting

If you encounter issues:

1. **Model not found**: Make sure the model architecture is properly registered in `vllm/model_executor/models/registry.py`
2. **Import errors**: Verify that the implementation files are in the correct location
3. **Memory issues on CPU**: Reduce the model size or batch size for testing
4. **GPU errors**: Check GPU availability and CUDA version compatibility

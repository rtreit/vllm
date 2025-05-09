# Test script for CodeXEmbedModel2B on CPU
import argparse
import numpy as np
import torch
import os

from vllm import LLM, SamplingParams

def test_embedding_model(model_name='hf-internal-testing/tiny-random-GemmaForCausalLM'):
    """
    Test CodeXEmbedModel2B implementation or a similar embedding model on CPU.
    
    For initial testing, we use a tiny model with the right architecture to validate
    the code paths work, without needing to download the full SFR/SFR-Embedding-Code-2B_R model.
    
    For the actual model testing, use model_name='SFR/SFR-Embedding-Code-2B_R'
    """
    print(f"Testing embedding model: {model_name}")
    print(f"Running on device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    # Initialize the LLM for embedding
    # The trust_remote_code may be needed for the actual SFR model
    # The 'cpu' tensor_parallel_devices specifies CPU-only execution
    llm = LLM(
        model=model_name,
        task="embed",  # Important to specify embedding task
        trust_remote_code=False, 
        dtype="float32",  # Use float32 on CPU
        tensor_parallel_devices=["cpu"],
        max_model_len=128,  # Keep small for testing
    )
    
    # Create test code snippets
    code_snippets = [
        "def hello_world():\n    print('Hello, world!')",
        "function greet() {\n    console.log('Hello, world!');\n}"
    ]
    
    # Generate embeddings
    print("Generating embeddings...")
    embeddings = llm.encode(code_snippets)
    
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embeddings sample (first 5 elements):\n{embeddings[0, :5]}")
    
    # Check if embeddings are normalized (L2 norm should be close to 1.0)
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"L2 norms: {norms}")
    
    # Calculate similarity between the two code snippets
    similarity = np.dot(embeddings[0], embeddings[1]) / (norms[0] * norms[1])
    print(f"Cosine similarity between snippets: {similarity}")
    
    return embeddings

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test CodeXEmbedModel2B implementation on CPU")
    parser.add_argument("--model", type=str, default="hf-internal-testing/tiny-random-GemmaForCausalLM",
                        help="Model name or path, use SFR/SFR-Embedding-Code-2B_R for actual model")
    args = parser.parse_args()
    
    test_embedding_model(args.model)

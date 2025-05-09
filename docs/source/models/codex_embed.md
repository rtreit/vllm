# SFR-Embedding-Code-2B_R - Code Embeddings Model

## Overview

The SFR-Embedding-Code-2B_R model is a specialized embedding model designed for code representation. It uses a Gemma2 architecture as its backbone and implements last token pooling to generate embeddings that are particularly effective for code representation tasks.

## Features

- Based on the Gemma2 model architecture
- Uses last token pooling and normalization
- Optimized for Code Information Retrieval tasks
- One of the top performers on the COIR benchmark

## Usage

To use this model for generating code embeddings:

```python
from vllm import LLM

# Initialize the model - make sure to specify the task as "embed"
llm = LLM(
    model="SFR/SFR-Embedding-Code-2B_R", 
    task="embed",
    trust_remote_code=False
)

# Generate embeddings for code snippets
code_snippets = [
    "def hello_world():\n    print('Hello, world!')",
    "function greet() {\n    console.log('Hello, world!');\n}"
]

# Generate embeddings
embeddings = llm.encode(code_snippets)
print(f"Embeddings shape: {embeddings.shape}")
```

## Performance

The SFR-Embedding-Code-2B_R model is specifically designed for code embedding tasks and performs exceptionally well on the COIR (Code Information Retrieval) benchmark, making it an excellent choice for:

- Code search and retrieval
- Code similarity comparison 
- Code understanding tasks
- Repository exploration
- Code recommendation systems

## Model Architecture

This model extends the Gemma2 architecture and adds specialized pooling for code embeddings:

- It uses last token pooling from the output hidden states
- Applies L2 normalization to the embeddings
- Strips out the language model head weights that aren't needed for embeddings

## Implementation Details

The vLLM implementation uses the `CodeXEmbedModel2B` class to generate embeddings. The model is configured for tensor parallelism and pipeline parallelism, ensuring efficient inference across multiple GPUs.

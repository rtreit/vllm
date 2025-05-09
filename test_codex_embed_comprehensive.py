# Comprehensive test script for CodeXEmbedModel2B
import argparse
import numpy as np
import torch
from typing import List, Dict, Any, Optional, Union, Tuple
import logging
import sys
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("CodeXEmbedTest")

# Import vLLM components
try:
    from vllm import LLM, SamplingParams
    from vllm.config import VllmConfig, ModelConfig, PoolerConfig
    from vllm.model_executor.layers.pooler import PoolingType
    
    # Test direct import of your model
    try:
        from vllm.model_executor.models.codex_embed import CodeXEmbedModel2B
        from vllm.model_executor.pooling_metadata import PoolingMetadata
        logger.info("Successfully imported CodeXEmbedModel2B directly")
    except ImportError:
        logger.warning("Could not import CodeXEmbedModel2B directly - this is expected if running in a docker container before installing vllm")
except ImportError:
    logger.error("Failed to import vLLM - make sure it's installed")
    sys.exit(1)

def test_model_loading(model_name: str) -> None:
    """Test that we can load the model architecture."""
    logger.info(f"Testing model loading: {model_name}")
    
    try:
        llm = LLM(
            model=model_name,
            task="embed",
            trust_remote_code=False,
            dtype="float32",  # Use float32 on CPU
            tensor_parallel_devices=["cpu"],
            max_model_len=128,  # Keep small for testing
        )
        logger.info("✓ Successfully loaded model")
        
        # Check model type
        model_type = type(llm.llm_engine.model_executor.model).__name__
        logger.info(f"Model type: {model_type}")
        if model_name == "SFR/SFR-Embedding-Code-2B_R" and model_type != "CodeXEmbedModel2B":
            logger.warning(f"Expected CodeXEmbedModel2B but got {model_type}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

def test_embedding_generation(model_name: str, code_snippets: List[str]) -> np.ndarray:
    """Test generating embeddings for code snippets."""
    logger.info(f"Testing embedding generation with model: {model_name}")
    
    try:
        llm = LLM(
            model=model_name,
            task="embed",
            trust_remote_code=False,
            dtype="float32",
            tensor_parallel_devices=["cpu"],
            max_model_len=128,
        )
        
        logger.info(f"Generating embeddings for {len(code_snippets)} code snippets...")
        embeddings = llm.encode(code_snippets)
        
        logger.info(f"Embeddings shape: {embeddings.shape}")
        logger.info(f"First 5 values of first embedding: {embeddings[0, :5]}")
        
        # Check if embeddings are normalized
        norms = np.linalg.norm(embeddings, axis=1)
        logger.info(f"L2 norms: {norms}")
        if not np.allclose(norms, 1.0, rtol=1e-3):
            logger.warning(f"Embeddings are not normalized (norms not close to 1.0)")
        else:
            logger.info("✓ Embeddings are normalized (L2 norm ≈ 1.0)")
        
        return embeddings
        
    except Exception as e:
        logger.error(f"Failed to generate embeddings: {e}")
        raise

def test_similarity_calculation(embeddings: np.ndarray, expected_similarities: Dict[Tuple[int, int], str] = None) -> None:
    """Test calculating similarities between embeddings."""
    logger.info("Testing similarity calculations...")
    
    try:
        n_embeddings = embeddings.shape[0]
        
        # Calculate all pairwise similarities
        similarity_matrix = np.zeros((n_embeddings, n_embeddings))
        for i in range(n_embeddings):
            for j in range(n_embeddings):
                similarity_matrix[i, j] = np.dot(embeddings[i], embeddings[j])
        
        logger.info(f"Similarity matrix:\n{similarity_matrix}")
        
        # Check for expected similarity patterns
        if expected_similarities:
            for (i, j), expectation in expected_similarities.items():
                similarity = similarity_matrix[i, j]
                logger.info(f"Similarity between snippets {i} and {j}: {similarity:.4f} - {expectation}")
        
        logger.info("✓ Similarity calculation complete")
        
    except Exception as e:
        logger.error(f"Failed to calculate similarities: {e}")
        raise

def run_comprehensive_test(model_name: str = "hf-internal-testing/tiny-random-GemmaForCausalLM") -> None:
    """Run all tests."""
    logger.info(f"Running comprehensive tests on {model_name}")
    logger.info(f"Running on device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    # Sample code snippets in different languages with similar functionality
    code_snippets = [
        # Python
        """def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1)
        """,
        
        # JavaScript
        """function factorial(n) {
    if (n <= 1) {
        return 1;
    }
    return n * factorial(n-1);
}
        """,
        
        # Python list comprehension (different approach)
        """def get_factorials(max_n):
    return [math.factorial(i) for i in range(1, max_n+1)]
        """
    ]
    
    # Test model loading
    test_model_loading(model_name)
    
    # Test embedding generation
    embeddings = test_embedding_generation(model_name, code_snippets)
    
    # Test similarity calculation with expected patterns
    expected_similarities = {
        (0, 1): "High similarity expected between Python and JavaScript factorial implementations",
        (0, 2): "Medium similarity expected between factorial and get_factorials",
        (1, 2): "Lower similarity expected between JavaScript factorial and Python get_factorials"
    }
    test_similarity_calculation(embeddings, expected_similarities)
    
    logger.info("All tests completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comprehensive test for CodeXEmbedModel2B implementation")
    parser.add_argument("--model", type=str, default="hf-internal-testing/tiny-random-GemmaForCausalLM",
                        help="Model name or path, use SFR/SFR-Embedding-Code-2B_R for actual model")
    args = parser.parse_args()
    
    run_comprehensive_test(args.model)

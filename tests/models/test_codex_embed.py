# SPDX-License-Identifier: Apache-2.0

"""Tests for the CodeXEmbedModel2B model."""

import pytest
import torch

from vllm.model_executor.models.codex_embed import CodeXEmbedModel2B
from vllm.config import VllmConfig, ModelConfig, PoolerConfig
from vllm.model_executor.layers.pooler import PoolingType
from vllm.model_executor.pooling_metadata import PoolingMetadata
from vllm.model_executor.models.gemma2 import Gemma2Config


@pytest.mark.parametrize("normalize", [True, False])
def test_codex_embed_pooler(normalize):
    """Test that the CodeXEmbedModel2B pooler produces the expected output."""
    # Create a simple model config for testing
    hidden_size = 32
    gemma2_config = Gemma2Config(
        hidden_size=hidden_size,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=hidden_size * 4,
        rms_norm_eps=1e-6,
    )
    
    # Create a pooler config with LAST pooling
    pooler_config = PoolerConfig(pooling_type=PoolingType.LAST.value)
    
    # Create a model config
    model_config = ModelConfig(hf_config=gemma2_config, pooler_config=pooler_config)
    
    # Create a VllmConfig
    vllm_config = VllmConfig(model_config=model_config)
    
    # Create the model
    model = CodeXEmbedModel2B(vllm_config=vllm_config)
    
    # Test the pooler
    batch_size = 2
    seq_len = 5
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    
    # Create pooling metadata for LAST token pooling
    pooling_metadata = PoolingMetadata(
        seq_groups=[[0], [1]],
        seq_data={0: (0, seq_len - 1), 1: (1, seq_len - 1)},
        prompt_lens=[seq_len, seq_len],
    )
    
    # Get pooler output
    output = model.pooler(hidden_states, pooling_metadata)
    
    # Check that output shape is correct
    assert output.hidden_states.shape == (batch_size, hidden_size)
    
    # Check that the last token is used for pooling
    expected_outputs = hidden_states[:, -1, :]
    if normalize:
        expected_outputs = torch.nn.functional.normalize(expected_outputs, p=2, dim=1)
    
    # For the test to pass with normalize=True, we need to actually configure the model's pooler
    # to use normalization, since the default in our implementation is True
    model._pooler.normalize = normalize
    
    torch.testing.assert_close(output.hidden_states, expected_outputs)

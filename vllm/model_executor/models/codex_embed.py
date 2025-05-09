# SPDX-License-Identifier: Apache-2.0

"""Inference-only CodeXEmbedModel2B model compatible with HuggingFace weights."""
from collections.abc import Iterable
from typing import Optional, Set, Tuple, Union

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.pooler import Pooler, PoolingType
from vllm.model_executor.models.gemma2 import Gemma2Model
from vllm.model_executor.models.utils import WeightsMapper, maybe_prefix
from vllm.model_executor.pooling_metadata import PoolingMetadata
from vllm.sequence import IntermediateTensors, PoolerOutput
from vllm.logger import init_logger

from .interfaces import SupportsLoRA, SupportsPP

logger = init_logger(__name__)


class CodeXEmbedModel2B(nn.Module, SupportsLoRA, SupportsPP):
    """
    Implementation of the SFR-Embedding-Code-2B_R embedding model.
    
    This model is based on Gemma2Model architecture and uses last token pooling.
    It's designed for code embeddings and is one of the best models for
    Code Information Retrieval according to the COIR benchmark.
    """

    hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={"model.": ""})

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config
        pooler_config = vllm_config.model_config.pooler_config

        self.config = config
        self.lora_config = lora_config
        self.quant_config = quant_config
        
        # Initialize the underlying Gemma2Model which is used by SFR-Embedding-Code-2B_R
        self.model = Gemma2Model(vllm_config=vllm_config,
                                prefix=maybe_prefix(prefix, "model"))

        # SFR-Embedding-Code-2B_R uses last token pooling
        self._pooler = Pooler.from_config_with_defaults(
            pooler_config,
            pooling_type=PoolingType.LAST,  # Use last token pooling
            normalize=True,                 # Normalize embeddings
            softmax=False)                  # Don't apply softmax

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> torch.Tensor:
        # Forward pass through the Gemma2 model
        return self.model(input_ids, positions, intermediate_tensors)

    def pooler(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> Optional[PoolerOutput]:
        # Apply last token pooling
        return self._pooler(hidden_states, pooling_metadata)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        # Apply weight mapping from HF to vLLM format
        weights = self.hf_to_vllm_mapper.apply(weights)
        # Filter out language model head weights which aren't needed for embeddings
        weights = ((name, data) for name, data in weights
                  if not name.startswith("lm_head."))
        # Load the weights into the underlying model
        return self.model.load_weights(weights)

from dataclasses import dataclass, field

from emperor.config import ConfigBase, optional_field
from emperor.embedding.absolute import AbsolutePositionalEmbeddingConfig
from emperor.embedding.contextual import ByteContextualEmbeddingConfig
from emperor.embedding.hierarchical import HierarchicalByteEmbeddingConfig
from emperor.layers import LayerStackConfig, NormalizationOptions, RecurrentLayerConfig
from models.gpt.expert_linear_adaptive._boundary_config_factory import (
    GptBoundaryConfig,
)


@dataclass
class ExperimentConfig(ConfigBase):
    contextual_embedding_config: ByteContextualEmbeddingConfig | None = field(
        default=None, kw_only=True
    )
    hierarchical_embedding_config: HierarchicalByteEmbeddingConfig | None = field(
        default=None, kw_only=True
    )
    decoder_output_normalization: NormalizationOptions = field(
        default=NormalizationOptions.LAYER_NORM, kw_only=True
    )
    positional_embedding_config: AbsolutePositionalEmbeddingConfig | None = (
        optional_field(
            "Config for the absolute positional embedding added to token embeddings."
        )
    )
    boundary_config: GptBoundaryConfig | None = optional_field(
        "Resolved configuration for GPT embeddings and the language-modeling head."
    )
    decoder_config: LayerStackConfig | RecurrentLayerConfig | None = optional_field(
        "Config for the transformer decoder block (a stack of decoder-layer blocks), "
        "optionally wrapped in a recurrent layer."
    )

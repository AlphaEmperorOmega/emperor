from dataclasses import dataclass, field

from emperor.config import ConfigBase, optional_field
from emperor.embedding.absolute import AbsolutePositionalEmbeddingConfig
from emperor.layers import LayerStackConfig, NormalizationOptions, RecurrentLayerConfig
from models.bert.linear_adaptive._boundary_config_factory import BertBoundaryConfig


@dataclass
class ExperimentConfig(ConfigBase):
    encoder_output_normalization: NormalizationOptions = field(
        default=NormalizationOptions.LAYER_NORM, kw_only=True
    )
    positional_embedding_config: AbsolutePositionalEmbeddingConfig | None = (
        optional_field(
            "Config for the absolute positional embedding added to token embeddings."
        )
    )
    boundary_config: BertBoundaryConfig | None = optional_field(
        "Resolved configuration for BERT token embeddings and MLM/NSP heads."
    )
    encoder_config: LayerStackConfig | RecurrentLayerConfig | None = optional_field(
        "Config for the transformer encoder block (a stack of encoder-layer blocks), "
        "optionally wrapped in a recurrent layer."
    )

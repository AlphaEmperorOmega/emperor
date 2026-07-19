from dataclasses import dataclass

from emperor.config import ConfigBase, optional_field
from emperor.embedding.absolute import AbsolutePositionalEmbeddingConfig
from emperor.layers import LayerStackConfig, RecurrentLayerConfig
from models.bert.expert_linear._boundary_config_factory import BertBoundaryConfig


@dataclass
class ExperimentConfig(ConfigBase):
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

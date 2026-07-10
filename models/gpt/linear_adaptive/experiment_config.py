from dataclasses import dataclass

from emperor.base.layer import LayerStackConfig, RecurrentLayerConfig
from emperor.base.utils import ConfigBase, optional_field
from emperor.embedding.absolute.core.config import AbsolutePositionalEmbeddingConfig

from models.gpt.linear_adaptive._boundary_config_factory import GptBoundaryConfig


@dataclass
class ExperimentConfig(ConfigBase):
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

from dataclasses import dataclass

from emperor.base.utils import ConfigBase, optional_field
from emperor.base.layer import LayerStackConfig, RecurrentLayerConfig
from emperor.embedding.absolute.core.config import AbsolutePositionalEmbeddingConfig


@dataclass
class ExperimentConfig(ConfigBase):
    positional_embedding_config: "AbsolutePositionalEmbeddingConfig | None" = (
        optional_field(
            "Config for the absolute positional embedding added to token embeddings."
        )
    )
    embedding_dropout_probability: float | None = optional_field(
        "Dropout applied to the summed token+positional+segment embedding, after the "
        "post-embedding LayerNorm and before the encoder."
    )
    encoder_config: "LayerStackConfig | RecurrentLayerConfig | None" = optional_field(
        "Config for the transformer encoder block (a stack of encoder-layer blocks), "
        "optionally wrapped in a recurrent layer."
    )

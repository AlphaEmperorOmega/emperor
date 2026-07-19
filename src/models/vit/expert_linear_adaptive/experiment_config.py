from dataclasses import dataclass

from emperor.config import ConfigBase, optional_field
from emperor.embedding.absolute import AbsolutePositionalEmbeddingConfig
from emperor.layers import LayerStackConfig, RecurrentLayerConfig
from emperor.linears import LinearLayerConfig
from emperor.patch import PatchConfig


@dataclass
class ExperimentConfig(ConfigBase):
    patch_config: PatchConfig | None = optional_field(
        "Config for image patch extraction and projection."
    )
    positional_embedding_config: AbsolutePositionalEmbeddingConfig | None = (
        optional_field("Config for image positional embeddings added to patches.")
    )
    encoder_config: LayerStackConfig | RecurrentLayerConfig | None = optional_field(
        "Config for the generic LayerStack transformer encoder block stack."
    )
    output_config: LinearLayerConfig | None = optional_field(
        "Config for the [CLS] classification head."
    )

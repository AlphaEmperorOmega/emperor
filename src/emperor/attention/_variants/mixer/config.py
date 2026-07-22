"""Private mixer-attention configuration implementation."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from emperor.config import ConfigBase, optional_field

if TYPE_CHECKING:
    from emperor.experts import MixtureOfExpertsModelConfig
    from emperor.layers import LayerStackConfig, RecurrentLayerConfig


@dataclass
class MixerAttentionConfig(ConfigBase):
    embedding_dim: int | None = optional_field(
        "Exact feature width of each token passed to the mixer."
    )
    sequence_length: int | None = optional_field(
        "Exact token sequence length accepted by the mixer."
    )
    batch_first_flag: bool | None = optional_field(
        "Explicit input layout. True selects [batch, sequence, embedding]; "
        "False selects [sequence, batch, embedding]."
    )
    causal_attention_mask_flag: bool | None = optional_field(
        "Whether causal processing is requested. MixerAttention requires False."
    )
    mixing_model_config: "LayerStackConfig | MixtureOfExpertsModelConfig | RecurrentLayerConfig | None" = (  # noqa: E501
        optional_field(
            "Model configuration built with sequence_length as both its input and "
            "output dimensions."
        )
    )

    def _registry_owner(self) -> type:
        from emperor.attention._variants.mixer.layer import MixerAttention

        return MixerAttention

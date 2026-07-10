from dataclasses import dataclass
from typing import TYPE_CHECKING

from emperor.base.utils import ConfigBase, optional_field

if TYPE_CHECKING:
    from emperor.base.layer import LayerStackConfig, RecurrentLayerConfig
    from emperor.experts.config import MixtureOfExpertsModelConfig


@dataclass
class FeedForwardConfig(ConfigBase):
    input_dim: int | None = optional_field("Feed-forward input feature dimension.")
    output_dim: int | None = optional_field("Feed-forward output feature dimension.")
    stack_config: "LayerStackConfig | MixtureOfExpertsModelConfig | RecurrentLayerConfig | None" = optional_field(  # noqa: E501
        "Either a LayerStackConfig (plain feed-forward stack), a "
        "MixtureOfExpertsModelConfig (mixture-of-experts model that manages its "
        "own routing state), or a RecurrentLayerConfig wrapping either form. "
        "Depth lives on the inner config."
    )

    def _registry_owner(self) -> type:
        from emperor.transformer.feed_forward.core.layers import FeedForward

        return FeedForward

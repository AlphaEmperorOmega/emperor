from dataclasses import dataclass
from emperor.base.config import ConfigBase, optional_field

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.augmentations.adaptive_parameters import (
        AdaptiveParameterAugmentationConfig,
    )


@dataclass
class LinearLayerConfig(ConfigBase):
    input_dim: int | None = optional_field(
        "Input feature dimension."
    )
    output_dim: int | None = optional_field(
        "Output feature dimension."
    )
    bias_flag: bool | None = optional_field(
        "Add a learnable bias to the output."
    )

    def _registry_owner(self) -> type:
        from emperor.linears.core.layers import LinearLayer

        return LinearLayer


@dataclass
class AdaptiveLinearLayerConfig(LinearLayerConfig):
    adaptive_augmentation_config: "AdaptiveParameterAugmentationConfig | None" = (
        optional_field(
            "Optional input-dependent parameter augmentation."
        )
    )

    def _registry_owner(self) -> type:
        from emperor.linears.core.layers import AdaptiveLinearLayer

        return AdaptiveLinearLayer

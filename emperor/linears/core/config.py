from dataclasses import dataclass
from emperor.base.utils import ConfigBase, optional_field

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.augmentations.adaptive_parameters import (
        AdaptiveParameterAugmentationConfig,
    )


@dataclass
class LinearLayerConfig(ConfigBase):
    input_dim: int | None = optional_field(
        "Number of input features for the linear transformation"
    )
    output_dim: int | None = optional_field(
        "Number of output features produced by the linear transformation"
    )
    bias_flag: bool | None = optional_field(
        "When true a learnable bias vector is added to the output after the linear transformation"
    )

    def _registry_owner(self) -> type:
        from emperor.linears.core.layers import LinearLayer

        return LinearLayer


@dataclass
class AdaptiveLinearLayerConfig(LinearLayerConfig):
    adaptive_augmentation_config: "AdaptiveParameterAugmentationConfig | None" = (
        optional_field(
            "Config for input-dependent parameter augmentations applied to the linear layer"
        )
    )

    def _registry_owner(self) -> type:
        from emperor.linears.core.layers import AdaptiveLinearLayer

        return AdaptiveLinearLayer

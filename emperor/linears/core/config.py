from dataclasses import dataclass, field
from emperor.base.utils import ConfigBase

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.base.utils import Module
    from emperor.augmentations.adaptive_parameters.config import (
        AdaptiveParameterAugmentationConfig,
    )


@dataclass
class LinearLayerConfig(ConfigBase):
    input_dim: int | None = field(
        default=None,
        metadata={"help": "Number of input features for the linear transformation"},
    )
    output_dim: int | None = field(
        default=None,
        metadata={
            "help": "Number of output features produced by the linear transformation"
        },
    )
    bias_flag: bool | None = field(
        default=None,
        metadata={
            "help": "When true a learnable bias vector is added to the output after the linear transformation"
        },
    )

    def build(
        self,
        overrides: "LinearLayerConfig | None" = None,
    ) -> "Module":
        from emperor.linears.core.layers import LinearLayer

        return LinearLayer(self, overrides)


@dataclass
class AdaptiveLinearLayerConfig(LinearLayerConfig):
    adaptive_augmentation_config: "AdaptiveParameterAugmentationConfig | None" = field(
        default=None,
        metadata={
            "help": "Config for input-dependent parameter augmentations applied to the linear layer"
        },
    )

    def build(self, overrides: "AdaptiveLinearLayerConfig | None" = None) -> "Module":
        from emperor.linears.core.layers import AdaptiveLinearLayer

        return AdaptiveLinearLayer(self, overrides)

from dataclasses import dataclass, field
from emperor.base.utils import ConfigBase
from emperor.augmentations.adaptive_parameters.config import (
    AdaptiveParameterAugmentationConfig,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.base.utils import Module


@dataclass
class LinearLayerConfig(ConfigBase):
    input_dim: int | None = field(
        default=None,
        metadata={"help": "Input dimension of the linear layer"},
    )
    output_dim: int | None = field(
        default=None,
        metadata={"help": "Output dimension of the linear layer"},
    )
    bias_flag: bool | None = field(
        default=None,
        metadata={
            "help": "When true bias will be added to after the matrix multiplication between, the input and output"
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

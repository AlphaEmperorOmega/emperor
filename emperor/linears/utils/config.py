from dataclasses import dataclass, field
from emperor.base.utils import ConfigBase
from emperor.linears.utils._monitors import TensorMonitor, StatisticsMonitor
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
    data_monitor: type[TensorMonitor] | None = field(
        default=None,
        metadata={
            "help": "Optional monitor class that tracks input/output statistics and logs to TensorBoard."
        },
    )
    parameter_monitor: type[StatisticsMonitor] | None = field(
        default=None,
        metadata={
            "help": "Optional monitor class that tracks parameter statistics (mean/var/norm) and logs to TensorBoard."
        },
    )

    def build(self, input_dim: int, output_dim: int) -> "Module":
        from emperor.linears.utils.layers import LinearLayer

        overrides = LinearLayerConfig(input_dim=input_dim, output_dim=output_dim)
        return LinearLayer(self, overrides)


@dataclass
class AdaptiveLinearLayerConfig(LinearLayerConfig):
    adaptive_augmentation_config: "AdaptiveParameterAugmentationConfig | None" = field(
        default=None,
        metadata={
            "help": "Config for input-dependent parameter augmentations applied to the linear layer"
        },
    )

    def build(self, input_dim: int, output_dim: int) -> "Module":
        from emperor.linears.utils.layers import AdaptiveLinearLayer

        overrides = AdaptiveLinearLayerConfig(
            input_dim=input_dim, output_dim=output_dim
        )
        return AdaptiveLinearLayer(self, overrides)

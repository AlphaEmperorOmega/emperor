from dataclasses import dataclass, field
from emperor.augmentations.adaptive_parameters.core.depth_mapper import (
    DepthMappingHandlerConfig,
)
from emperor.augmentations.adaptive_parameters.core.diagonal import (
    DynamicDiagonalConfig,
)
from emperor.base.layer.config import LayerStackConfig
from emperor.base.utils import ConfigBase
from emperor.augmentations.adaptive_parameters.core.bias import (
    DynamicBiasConfig,
)
from emperor.augmentations.adaptive_parameters.core.weight import (
    DynamicWeightConfig,
)
from emperor.augmentations.adaptive_parameters.core.mask import (
    AxisMaskConfig,
)


@dataclass
class AdaptiveParameterAugmentationConfig(ConfigBase):
    input_dim: int | None = field(
        default=None,
        metadata={"help": "Input feature dimension."},
    )
    output_dim: int | None = field(
        default=None,
        metadata={"help": "Output feature dimension."},
    )
    diagonal_config: DynamicDiagonalConfig | None = field(
        default=None,
        metadata={
            "help": "Optional dynamic diagonal adjustment."
        },
    )
    weight_config: DynamicWeightConfig | None = field(
        default=None,
        metadata={"help": "Optional dynamic weight adjustment."},
    )
    bias_config: DynamicBiasConfig | None = field(
        default=None,
        metadata={"help": "Optional dynamic bias adjustment."},
    )
    mask_config: AxisMaskConfig | None = field(
        default=None,
        metadata={"help": "Optional dynamic weight mask."},
    )
    model_config: LayerStackConfig | None = field(
        default=None,
        metadata={"help": "Internal generator network config."},
    )

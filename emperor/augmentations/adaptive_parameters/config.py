from dataclasses import dataclass
from emperor.augmentations.adaptive_parameters.core.depth_mapper import (
    DepthMappingHandlerConfig,
)
from emperor.augmentations.adaptive_parameters.core.diagonal import (
    DynamicDiagonalConfig,
)
from emperor.base.layer.config import LayerStackConfig
from emperor.base.utils import ConfigBase, optional_field
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
    input_dim: int | None = optional_field(
        "Input feature dimension."
    )
    output_dim: int | None = optional_field(
        "Output feature dimension."
    )
    diagonal_config: DynamicDiagonalConfig | None = optional_field(
        "Optional dynamic diagonal adjustment."
    )
    weight_config: DynamicWeightConfig | None = optional_field(
        "Optional dynamic weight adjustment."
    )
    bias_config: DynamicBiasConfig | None = optional_field(
        "Optional dynamic bias adjustment."
    )
    mask_config: AxisMaskConfig | None = optional_field(
        "Optional dynamic weight mask."
    )
    model_config: LayerStackConfig | None = optional_field(
        "Internal generator network config."
    )

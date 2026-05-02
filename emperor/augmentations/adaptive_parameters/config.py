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
        metadata={"help": "Input dimension of the linear layer"},
    )
    output_dim: int | None = field(
        default=None,
        metadata={"help": "Output dimension of the linear layer"},
    )
    diagonal_config: DynamicDiagonalConfig | None = field(
        default=None,
        metadata={
            "help": "Configuration for input-dependent diagonal weight adjustments."
        },
    )
    weight_config: DynamicWeightConfig | None = field(
        default=None,
        metadata={"help": "Configuration for input-dependent weight adjustments."},
    )
    bias_config: DynamicBiasConfig | None = field(
        default=None,
        metadata={"help": "Configuration for input-dependent bias adjustments."},
    )
    mask_config: AxisMaskConfig | None = field(
        default=None,
        metadata={"help": "Configuration for input-dependent weight matrix masking."},
    )
    model_config: LayerStackConfig | None = field(
        default=None,
        metadata={
            "help": "Layer stack configuration for the internal generator network."
        },
    )

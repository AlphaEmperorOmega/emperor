from dataclasses import dataclass, field
from emperor.base.layer.config import LayerStackConfig
from emperor.base.utils import ConfigBase
from emperor.augmentations.adaptive_parameters.options import DynamicDiagonalOptions
from emperor.augmentations.adaptive_parameters.utils.handlers.bias import BiasHandlerConfig
from emperor.augmentations.adaptive_parameters.utils.handlers.weight import WeightHandlerConfig
from emperor.augmentations.adaptive_parameters.utils.handlers.mask import MaskHandlerConfig
from emperor.augmentations.adaptive_parameters.utils.handlers.memory import MemoryHandlerConfig


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
    diagonal_option: DynamicDiagonalOptions | None = field(
        default=None,
        metadata={"help": "Input-dependent adjustment of the weight matrix diagonal."},
    )
    weight_config: WeightHandlerConfig | None = field(
        default=None,
        metadata={"help": "Configuration for input-dependent weight adjustments."},
    )
    bias_config: BiasHandlerConfig | None = field(
        default=None,
        metadata={"help": "Configuration for input-dependent bias adjustments."},
    )
    mask_config: MaskHandlerConfig | None = field(
        default=None,
        metadata={"help": "Configuration for input-dependent weight matrix masking."},
    )
    memory_config: MemoryHandlerConfig | None = field(
        default=None,
        metadata={"help": "Configuration for learned memory representation blending."},
    )
    model_config: LayerStackConfig | None = field(
        default=None,
        metadata={"help": ""},
    )

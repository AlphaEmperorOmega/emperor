from dataclasses import dataclass, field
from enum import Enum
from emperor.base.utils import ConfigBase
from emperor.augmentations.adaptive_parameters.config import AdaptiveParameterAugmentationConfig

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.sampler.utils.samplers import SamplerConfig
    from emperor.sampler.utils.routers import RouterConfig
    from emperor.parametric.utils.mixtures.options import (
        AdaptiveBiasOptions,
        AdaptiveWeightOptions,
    )


class AdaptiveRouterOptions(Enum):
    SHARED_ROUTER = 1
    INDEPENTENT_ROUTER = 2


@dataclass
class ParametricLayerConfig(ConfigBase):
    input_dim: int | None = field(
        default=None,
        metadata={"help": "Input dimensionality for weight parameters."},
    )
    output_dim: int | None = field(
        default=None,
        metadata={"help": "Output dimensionality for weight parameters."},
    )
    adaptive_weight_option: "AdaptiveWeightOptions | None" = field(
        default=None,
        metadata={
            "help": "Specifies options for generating weight parameters for individual input samples."
        },
    )
    adaptive_bias_option: "AdaptiveBiasOptions | None" = field(
        default=None,
        metadata={
            "help": "Specifies options for generating bias parameters for individual input samples."
        },
    )
    init_sampler_model_option: "AdaptiveRouterOptions | None" = field(
        default=None,
        metadata={
            "help": "When `True` the `RouterModel `and `SamplerModel` will be added to the current layer."
        },
    )
    time_tracker_flag: bool | None = field(
        default=None,
        metadata={
            "help": "When `True` it will generate bias parameters for each input sample."
        },
    )
    router_config: "RouterConfig | None" = field(
        default=None,
        metadata={"help": "Configuration for the router model."},
    )
    sampler_config: "SamplerConfig | None" = field(
        default=None,
        metadata={"help": "Configuration for the sampler model."},
    )
    adaptive_behaviour_config: "AdaptiveParameterAugmentationConfig | None" = field(
        default=None,
        metadata={"help": ""},
    )

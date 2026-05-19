from dataclasses import dataclass
from enum import Enum
from emperor.base.utils import ConfigBase, optional_field
from emperor.augmentations.adaptive_parameters.config import (
    AdaptiveParameterAugmentationConfig,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.sampler.core.samplers import SamplerConfig
    from emperor.sampler.core.routers import RouterConfig
    from emperor.parametric.core.mixtures.options import (
        AdaptiveBiasOptions,
        AdaptiveWeightOptions,
    )


class AdaptiveRouterOptions(Enum):
    SHARED_ROUTER = 1
    INDEPENTENT_ROUTER = 2


@dataclass
class ParametricLayerConfig(ConfigBase):
    input_dim: int | None = optional_field("Input feature dimension.")
    output_dim: int | None = optional_field("Output feature dimension.")
    adaptive_weight_option: "AdaptiveWeightOptions | None" = optional_field(
        "Weight parameter generation strategy."
    )
    adaptive_bias_option: "AdaptiveBiasOptions | None" = optional_field(
        "Bias parameter generation strategy."
    )
    routing_initialization_mode: "AdaptiveRouterOptions | None" = optional_field(
        "Router/sampler sharing mode."
    )
    router_config: "RouterConfig | None" = optional_field(
        "Configuration for the router model."
    )
    sampler_config: "SamplerConfig | None" = optional_field(
        "Configuration for the sampler model."
    )
    adaptive_augmentation_config: "AdaptiveParameterAugmentationConfig | None" = (
        optional_field("Input-dependent parameter application config.")
    )

    def _registry_owner(self) -> type:
        from emperor.parametric.core.layers import ParametricLayer

        return ParametricLayer

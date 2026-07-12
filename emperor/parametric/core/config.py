from dataclasses import dataclass
from enum import Enum
from emperor.base.layer import LayerConfig
from emperor.base.config import ConfigBase, optional_field
from emperor.augmentations.adaptive_parameters.config import (
    AdaptiveParameterAugmentationConfig,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.sampler.core.config import RouterConfig, SamplerConfig
    from emperor.parametric.core.mixtures.config import AdaptiveMixtureConfig


class AdaptiveRouterOptions(Enum):
    SHARED_ROUTER = 1
    INDEPENDENT_ROUTER = 2


@dataclass
class ParametricLayerConfig(ConfigBase):
    input_dim: int | None = optional_field("Input feature dimension.")
    output_dim: int | None = optional_field("Output feature dimension.")
    weight_mixture_config: "AdaptiveMixtureConfig | None" = optional_field(
        "Config for the input-dependent weight mixture model."
    )
    bias_mixture_config: "AdaptiveMixtureConfig | None" = optional_field(
        "Optional config for the input-dependent bias mixture model."
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


@dataclass
class ParametricLayerHandlerConfig(LayerConfig):
    def _registry_owner(self) -> type:
        from emperor.parametric.core.handlers import ParametricLayerHandler

        return ParametricLayerHandler

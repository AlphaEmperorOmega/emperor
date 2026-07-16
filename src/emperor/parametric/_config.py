from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from emperor.augmentations.adaptive_parameters import (
    AdaptiveParameterAugmentationConfig,
)
from emperor.config import ConfigBase, optional_field
from emperor.layers import LayerConfig

if TYPE_CHECKING:
    from emperor.parametric._mixtures.config import AdaptiveMixtureConfig
    from emperor.sampler import RouterConfig, SamplerConfig


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
        from emperor.parametric._layer import ParametricLayer

        return ParametricLayer


@dataclass
class ParametricLayerHandlerConfig(LayerConfig):
    def _registry_owner(self) -> type:
        from emperor.parametric._handlers import ParametricLayerHandler

        return ParametricLayerHandler

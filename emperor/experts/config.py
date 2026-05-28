from dataclasses import dataclass
from typing import TYPE_CHECKING

from emperor.base.utils import ConfigBase, optional_field

if TYPE_CHECKING:
    from emperor.base.layer import LayerStackConfig
    from emperor.experts.core.options import RoutingInitializationMode
    from emperor.sampler.core.config import SamplerConfig


@dataclass
class MixtureOfExpertsModelConfig(ConfigBase):
    input_dim: int | None = optional_field(
        "Input feature dimension for the mixture-of-experts model."
    )
    output_dim: int | None = optional_field(
        "Output feature dimension for the mixture-of-experts model."
    )
    top_k: int | None = optional_field(
        "Top-k probabilities and indices selected from the routing distribution."
    )
    routing_initialization_mode: "RoutingInitializationMode | None" = optional_field(
        "SHARED for a single sampler/router across all expert layers; "
        "LAYER for one per layer."
    )
    sampler_config: "SamplerConfig | None" = optional_field(
        "Sampler configuration used when routing is SHARED. The router lives "
        "at sampler_config.router_config."
    )
    stack_config: "LayerStackConfig | None" = optional_field(
        "Stack of expert layers built and orchestrated by MixtureOfExpertsModel."
    )

    def _registry_owner(self) -> type:
        from emperor.experts.model import MixtureOfExpertsModel

        return MixtureOfExpertsModel

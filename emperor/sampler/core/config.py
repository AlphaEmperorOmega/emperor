import copy

from dataclasses import dataclass
from emperor.base.config import ConfigBase, optional_field

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.sampler.model import SamplerModel


@dataclass
class RouterConfig(ConfigBase):
    input_dim: int | None = optional_field("Router input feature dimension.")
    num_experts: int | None = optional_field("Router output dimension.")
    noisy_topk_flag: bool | None = optional_field(
        "When True, the router emits an additional noise scale per expert."
    )
    model_config: "ConfigBase | None" = optional_field(
        "Internal router network config."
    )

    def _registry_owner(self) -> type:
        from emperor.sampler.core.routers import RouterModel

        return RouterModel


@dataclass
class SamplerConfig(ConfigBase):
    top_k: int | None = optional_field("Number of top probabilities to select.")
    threshold: float | None = optional_field(
        "Probability threshold for skip-mask updates."
    )
    filter_above_threshold: bool | None = optional_field(
        "Whether probabilities above threshold update the skip mask."
    )
    num_topk_samples: int | None = optional_field("Number of top-k entries to sample.")
    normalize_probabilities_flag: bool | None = optional_field(
        "Normalize selected probabilities to sum to one."
    )
    noisy_topk_flag: bool | None = optional_field(
        "Whether logits include noisy top-k scales."
    )
    num_experts: int | None = optional_field("Number of experts in the distribution.")
    coefficient_of_variation_loss_weight: float | None = optional_field(
        "CV loss weight."
    )
    switch_loss_weight: float | None = optional_field("Switch loss weight.")
    zero_centred_loss_weight: float | None = optional_field("Zero-centred loss weight.")
    mutual_information_loss_weight: float | None = optional_field(
        "Mutual information loss weight."
    )
    router_config: "RouterConfig | None" = optional_field(
        "Optional router configuration managed by SamplerModel."
    )

    def _registry_owner(self) -> type:
        from emperor.sampler.model import SamplerModel

        return SamplerModel

    def build_with_router_input_dim(self, input_dim: int) -> "SamplerModel":
        router_overrides = RouterConfig(input_dim=input_dim)
        router_config = self.__override_router_config(router_overrides)
        return self.build(SamplerConfig(router_config=router_config))

    def __override_router_config(self, overrides: "RouterConfig") -> "RouterConfig":
        if self.router_config is None:
            raise ValueError(
                "SamplerConfig.router_config must be set before overriding."
            )
        router_config = copy.deepcopy(self.router_config)
        for field_name in router_config.__dataclass_fields__:
            if field_name not in overrides.__dataclass_fields__:
                continue
            override_value = getattr(overrides, field_name)
            if override_value is None:
                continue
            setattr(router_config, field_name, override_value)
        return router_config

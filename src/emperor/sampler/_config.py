import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING

from emperor.config import ConfigBase, optional_field

if TYPE_CHECKING:
    from emperor.sampler._sampler import SamplerModel


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
        from emperor.sampler._router import RouterModel

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
        from emperor.sampler._sampler import SamplerModel

        return SamplerModel

    def build_with_router_input_dim(self, input_dim: int) -> "SamplerModel":
        router_input_dimension_override = RouterConfig(input_dim=input_dim)
        router_config_with_input_dimension = self.__override_router_config(
            router_input_dimension_override
        )
        return self.build(
            SamplerConfig(router_config=router_config_with_input_dimension)
        )

    def __override_router_config(
        self,
        router_config_overrides: "RouterConfig",
    ) -> "RouterConfig":
        if self.router_config is None:
            raise ValueError(
                "SamplerConfig.router_config must be set before overriding."
            )
        overridden_router_config = copy.deepcopy(self.router_config)
        for router_field_name in router_config_overrides.__dataclass_fields__:
            router_override_value = getattr(
                router_config_overrides,
                router_field_name,
            )
            if router_override_value is None:
                continue
            setattr(
                overridden_router_config,
                router_field_name,
                router_override_value,
            )
        return overridden_router_config

from dataclasses import dataclass

from emperor.base.layer import LayerStackConfig
from emperor.base.utils import ConfigBase, optional_field


@dataclass
class RouterConfig(ConfigBase):
    input_dim: int | None = optional_field("Router input feature dimension.")
    num_experts: int | None = optional_field("Router output dimension.")
    noisy_topk_flag: bool | None = optional_field(
        "When True, the router emits an additional noise scale per expert."
    )
    model_config: "LayerStackConfig | None" = optional_field(
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

"""Configuration for sampling positions within variable-length sequences."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from emperor.config import ConfigBase, optional_field
from emperor.sampler._config import RouterConfig

if TYPE_CHECKING:
    from emperor.sampler._token_sampler import TokenSamplerModel


@dataclass
class TokenSamplerConfig(ConfigBase):
    input_dim: int | None = optional_field("Token feature dimension.")
    selection_ratio: float | None = optional_field(
        "Select max(1, floor(ratio * token_count)) positions per sequence; (0, 1]."
    )
    router_config: RouterConfig | None = optional_field(
        "Configurable scoring model. Must emit one logit per token: num_experts=1 "
        "and noisy_topk_flag=False. The sampler uses sigmoid weights and Top-K "
        "across token positions, independently for each sequence."
    )

    def _registry_owner(self) -> type:
        from emperor.sampler._token_sampler import TokenSamplerModel

        return TokenSamplerModel

    def validate_for_input_dim(self, input_dim: int) -> None:
        self.registry_owner().VALIDATOR.validate_config(
            replace(self, input_dim=input_dim)
        )

    def build_with_router_input_dim(self, input_dim: int) -> TokenSamplerModel:
        return self.build(TokenSamplerConfig(input_dim=input_dim))

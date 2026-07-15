from dataclasses import dataclass
from typing import TYPE_CHECKING

from emperor.attention.core.config import MultiHeadAttentionConfig
from emperor.base.config import optional_field

if TYPE_CHECKING:
    from emperor.experts.core.config import MixtureOfExpertsConfig


@dataclass
class MixtureOfAttentionHeadsConfig(MultiHeadAttentionConfig):
    experts_config: "MixtureOfExpertsConfig | None" = optional_field(
        "Mixture-of-experts configuration used to build the projection experts."
    )
    use_kv_expert_models_flag: bool | None = optional_field(
        "If True, build the key and value projections as expert models as well."
    )

    def _registry_owner(self) -> type:
        from emperor.attention.core.variants.mixture_of_attention_heads.layer import (
            MixtureOfAttentionHeads,
        )

        return MixtureOfAttentionHeads

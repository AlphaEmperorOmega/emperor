from typing import TYPE_CHECKING

from emperor.attention.core.handlers.zero_attention import ZeroAttention

if TYPE_CHECKING:
    from emperor.attention.core.runtime import AttentionRuntimeShape
    from emperor.attention.core.variants.mixture_of_attention_heads.config import (
        MixtureOfAttentionHeadsConfig,
    )


class MixtureOfAttentionHeadsZeroAttention(ZeroAttention):
    def __init__(self, cfg: "MixtureOfAttentionHeadsConfig"):
        super().__init__(cfg)
        self.use_kv_expert_models_flag: bool = cfg.use_kv_expert_models_flag
        self.top_k: int = cfg.experts_config.top_k

    def _get_branch_count(
        self, runtime_shape: "AttentionRuntimeShape | None" = None
    ) -> int:
        if not self.use_kv_expert_models_flag:
            return super()._get_branch_count(runtime_shape)

        batch_size = (
            runtime_shape.batch_size if runtime_shape is not None else self.batch_size
        )
        return batch_size * self.top_k * self.num_heads

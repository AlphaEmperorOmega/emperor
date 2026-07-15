from typing import TYPE_CHECKING

from torch import Tensor

from emperor.attention.core.handlers.bias import KeyValueBias
from emperor.attention.core.variants.mixture_of_attention_heads.validator import (
    MixtureOfAttentionHeadsValidator,
)

if TYPE_CHECKING:
    from emperor.attention.core.runtime import AttentionRuntimeShape
    from emperor.attention.core.variants.mixture_of_attention_heads.config import (
        MixtureOfAttentionHeadsConfig,
    )


class MixtureOfAttentionHeadsKeyValueBias(KeyValueBias):
    VALIDATOR = MixtureOfAttentionHeadsValidator

    def __init__(self, cfg: "MixtureOfAttentionHeadsConfig"):
        super().__init__(cfg)
        self.use_kv_expert_models_flag: bool = cfg.use_kv_expert_models_flag
        self.top_k: int = cfg.experts_config.top_k

    def _expand_bias_vector(
        self,
        bias_vector: Tensor,
        projection: Tensor,
        runtime_shape: "AttentionRuntimeShape | None" = None,
    ) -> Tensor:
        if not self.use_kv_expert_models_flag:
            return super()._expand_bias_vector(
                bias_vector,
                projection,
                runtime_shape,
            )

        batch_size = (
            runtime_shape.batch_size if runtime_shape is not None else self.batch_size
        )
        branch_count = projection.size(0)
        expected_branch_count = batch_size * self.top_k * self.num_heads
        self.VALIDATOR.validate_expert_projection_branch_count(
            branch_count,
            expected_branch_count,
        )
        head_dim = projection.size(-1)
        bias_by_head = bias_vector.reshape(self.num_heads, head_dim)
        expanded_bias = bias_by_head.repeat(batch_size * self.top_k, 1)
        return expanded_bias.reshape(branch_count, 1, head_dim)

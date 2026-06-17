from torch import Tensor
from emperor.attention.core._validator import (
    AttentionValidatorBase,
    MultiHeadAttentionValidator,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.attention.core.layers import MultiHeadAttentionAbstract


class IndependentAttentionValidator(MultiHeadAttentionValidator):
    @staticmethod
    def validate_forward_inputs(
        model: "MultiHeadAttentionAbstract",
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> None:
        MultiHeadAttentionValidator.validate_forward_inputs(
            model, query, key, value, key_padding_mask, attention_mask
        )
        AttentionValidatorBase.validate_attention_weights_returned_for_self_attention_only(
            model
        )
        AttentionValidatorBase.validate_key_value_projection_shapes(key, value)

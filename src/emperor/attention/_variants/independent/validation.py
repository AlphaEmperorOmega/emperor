"""Private independent-attention validation implementation."""

from typing import TYPE_CHECKING

from emperor.attention._validation import MultiHeadAttentionValidator

if TYPE_CHECKING:
    from emperor.attention._base import MultiHeadAttentionAbstract
    from emperor.attention._runtime import QKV, AttentionMasks, MultiHeadAttentionInputs


class IndependentAttentionValidator(MultiHeadAttentionValidator):
    @classmethod
    def validate_forward_inputs(
        cls,
        model: "MultiHeadAttentionAbstract",
        attention_inputs: "MultiHeadAttentionInputs | QKV",
        masks: "AttentionMasks | None" = None,
    ) -> None:
        super().validate_forward_inputs(model, attention_inputs, masks)
        cls.validate_attention_weights_returned_for_self_attention_only(model)
        cls.validate_key_value_projection_shapes(
            attention_inputs.key,
            attention_inputs.value,
        )

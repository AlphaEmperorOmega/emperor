from typing import TYPE_CHECKING

from emperor.attention.core._validator import MultiHeadAttentionValidator

if TYPE_CHECKING:
    from emperor.attention.core.layers import MultiHeadAttentionAbstract
    from emperor.attention.core.runtime import QKV, AttentionMasks


class IndependentAttentionValidator(MultiHeadAttentionValidator):
    @classmethod
    def validate_forward_inputs(
        cls,
        model: "MultiHeadAttentionAbstract",
        qkv: "QKV",
        masks: "AttentionMasks",
    ) -> None:
        super().validate_forward_inputs(model, qkv, masks)
        cls.validate_attention_weights_returned_for_self_attention_only(model)
        cls.validate_key_value_projection_shapes(qkv.key, qkv.value)

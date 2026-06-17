from torch import Tensor
from emperor.attention.core._validator import MultiHeadAttentionValidator

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.attention.core.layers import MultiHeadAttentionAbstract


class SelfAttentionValidator(MultiHeadAttentionValidator):
    @staticmethod
    def validate(model: "MultiHeadAttentionAbstract") -> None:
        MultiHeadAttentionValidator.validate(model)
        SelfAttentionValidator.validate_self_attention_dimensions_equal(model)

    @staticmethod
    def validate_self_attention_dimensions_equal(
        model: "MultiHeadAttentionAbstract",
    ) -> None:
        query_key_projection_dim = (
            model.query_key_projection_dim or model.embedding_dim
        )
        value_projection_dim = model.value_projection_dim or model.embedding_dim
        if not (
            query_key_projection_dim == value_projection_dim == model.embedding_dim
        ):
            raise RuntimeError(
                "Self attention requires query_key_projection_dim, "
                "value_projection_dim, and embedding_dim to be equal, but got "
                f"query_key_projection_dim={model.query_key_projection_dim}, "
                f"value_projection_dim={model.value_projection_dim}, "
                f"embedding_dim={model.embedding_dim}."
            )

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
        SelfAttentionValidator.validate_query_key_value_are_same_tensor(
            query, key, value
        )

    @staticmethod
    def validate_query_key_value_are_same_tensor(
        query: Tensor, key: Tensor, value: Tensor
    ) -> None:
        if not (key is value and query is key):
            raise RuntimeError(
                "Self attention can only be computed when the query, key, and value "
                "are the same tensor."
            )

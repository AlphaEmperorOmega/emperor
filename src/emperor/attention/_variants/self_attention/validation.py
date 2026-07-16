"""Private self-attention validation implementation."""

from typing import TYPE_CHECKING

from torch import Tensor

from emperor.attention._validation import MultiHeadAttentionValidator
from emperor.attention._variants.self_attention.config import (
    SelfAttentionProjectionStrategy,
)
from emperor.layers import RecurrentLayerConfig

if TYPE_CHECKING:
    from emperor.attention._base import MultiHeadAttentionAbstract
    from emperor.attention._runtime import QKV, AttentionMasks


class SelfAttentionValidator(MultiHeadAttentionValidator):
    @classmethod
    def validate(cls, model: "MultiHeadAttentionAbstract") -> None:
        super().validate(model)
        cls.validate_self_attention_dimensions_equal(model)
        cls.validate_projection_strategy(model)

    @staticmethod
    def validate_projection_strategy(
        model: "MultiHeadAttentionAbstract",
    ) -> None:
        if (
            model.cfg.projection_strategy == SelfAttentionProjectionStrategy.FUSED
            and isinstance(model.cfg.projection_model_config, RecurrentLayerConfig)
        ):
            raise ValueError(
                "Self-attention with RecurrentLayerConfig requires "
                "projection_strategy=SelfAttentionProjectionStrategy.SEPARATE; "
                "the FUSED strategy changes embedding_dim to 3 * embedding_dim."
            )

    @staticmethod
    def validate_self_attention_dimensions_equal(
        model: "MultiHeadAttentionAbstract",
    ) -> None:
        query_key_projection_dim = model.query_key_projection_dim or model.embedding_dim
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

    @classmethod
    def validate_forward_inputs(
        cls,
        model: "MultiHeadAttentionAbstract",
        qkv: "QKV",
        masks: "AttentionMasks",
    ) -> None:
        super().validate_forward_inputs(model, qkv, masks)
        cls.validate_query_key_value_are_same_tensor(qkv.query, qkv.key, qkv.value)

    @staticmethod
    def validate_query_key_value_are_same_tensor(
        query: Tensor, key: Tensor, value: Tensor
    ) -> None:
        if not (key is value and query is key):
            raise RuntimeError(
                "Self attention can only be computed when the query, key, and value "
                "are the same tensor."
            )

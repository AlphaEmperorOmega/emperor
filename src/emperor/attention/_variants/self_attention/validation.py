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
        projection_strategy = model.cfg.projection_strategy
        if not isinstance(model.cfg.projection_model_config, RecurrentLayerConfig):
            return
        output_multipliers = {
            SelfAttentionProjectionStrategy.FUSED: 3,
            SelfAttentionProjectionStrategy.FUSED_KEY_VALUE: 2,
        }
        output_multiplier = output_multipliers.get(projection_strategy)
        if output_multiplier is not None:
            raise ValueError(
                "Self-attention with RecurrentLayerConfig requires "
                "projection_strategy=SelfAttentionProjectionStrategy.SEPARATE; "
                f"the {projection_strategy.name} strategy changes embedding_dim "
                f"to {output_multiplier} * embedding_dim."
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
        cls.validate_fused_projection_inputs(model, qkv)

    @staticmethod
    def validate_fused_projection_inputs(
        model: "MultiHeadAttentionAbstract",
        qkv: "QKV",
    ) -> None:
        if (
            model.cfg.projection_strategy == SelfAttentionProjectionStrategy.FUSED
            and qkv.query is not qkv.key
        ):
            raise RuntimeError(
                "SelfAttentionProjectionStrategy.FUSED requires query, key, and "
                "value to be the same tensor; use FUSED_KEY_VALUE when query and "
                "context differ."
            )

    @staticmethod
    def validate_query_key_value_are_same_tensor(
        query: Tensor, key: Tensor, value: Tensor
    ) -> None:
        del query
        if key is not value:
            raise RuntimeError(
                "Self attention requires the key and value to be the same tensor."
            )

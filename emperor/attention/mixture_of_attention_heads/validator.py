from torch import Tensor
from emperor.attention.core._validator import (
    AttentionValidatorBase,
    MultiHeadAttentionValidator,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.attention.core.layers import MultiHeadAttentionAbstract


class MixtureOfAttentionHeadsValidator(MultiHeadAttentionValidator):
    @staticmethod
    def validate(model: "MultiHeadAttentionAbstract") -> None:
        MultiHeadAttentionValidator.validate(model)
        MixtureOfAttentionHeadsValidator.validate_experts_configuration(model)

    @staticmethod
    def validate_experts_configuration(
        model: "MultiHeadAttentionAbstract",
    ) -> None:
        from emperor.experts.core.config import MixtureOfExpertsConfig

        experts_config = model.cfg.experts_config
        use_kv_expert_models_flag = model.cfg.use_kv_expert_models_flag
        if experts_config is None:
            raise ValueError("experts_config is required for mixture of attention heads.")
        if not isinstance(experts_config, MixtureOfExpertsConfig):
            raise TypeError(
                "experts_config must be a MixtureOfExpertsConfig, received "
                f"{type(experts_config).__name__}."
            )
        if use_kv_expert_models_flag is None:
            raise ValueError(
                "use_kv_expert_models_flag is required for mixture of attention heads."
            )
        if not isinstance(use_kv_expert_models_flag, bool):
            raise TypeError(
                "use_kv_expert_models_flag must be a bool, received "
                f"{type(use_kv_expert_models_flag).__name__}."
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
        AttentionValidatorBase.validate_attention_weights_returned_for_self_attention_only(
            model
        )
        AttentionValidatorBase.validate_key_value_projection_shapes(key, value)

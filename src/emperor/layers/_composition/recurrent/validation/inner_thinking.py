from __future__ import annotations

import math
from numbers import Real

from torch import Tensor, nn

from emperor.layers._composition.recurrent.config import InnerThinkingRecurrentConfig
from emperor.layers._composition.recurrent.validation.common import (
    _validate_variant_state,
)
from emperor.layers._composition.recurrent.validation.standard import (
    RecurrentLayerValidator,
)


class InnerThinkingRecurrentValidator(RecurrentLayerValidator):
    OPTIONAL_FIELDS = {*RecurrentLayerValidator.OPTIONAL_FIELDS, "thinking_step_scale"}

    @classmethod
    def validate(cls, model: object) -> None:
        from emperor.sampler import TokenSamplerConfig

        config = model.cfg
        if not isinstance(config, InnerThinkingRecurrentConfig):
            raise TypeError(
                "InnerThinkingRecurrent requires InnerThinkingRecurrentConfig."
            )
        super().validate(model)
        for name in ("input_dim", "output_dim", "max_steps"):
            if isinstance(getattr(config, name), bool):
                raise TypeError(f"{name} must be an integer, not bool.")
        if not isinstance(config.sampler_config, TokenSamplerConfig):
            raise TypeError("sampler_config must be a TokenSamplerConfig.")
        config.sampler_config.validate_for_input_dim(config.input_dim)
        scale = config.thinking_step_scale
        if scale is not None:
            if isinstance(scale, bool) or not isinstance(scale, Real):
                raise TypeError("thinking_step_scale must be a real number.")
            if not math.isfinite(scale) or scale < 0:
                raise ValueError("thinking_step_scale must be finite and non-negative.")

    @classmethod
    def validate_state(cls, state: object, expected_feature_dim: int) -> None:
        _validate_variant_state(
            state, expected_feature_dim, owner_name="InnerThinkingRecurrent"
        )

    @staticmethod
    def validate_block_layout(block: nn.Module, hidden: Tensor) -> None:
        if hidden.ndim > 2 and any(
            getattr(module, "batch_first_flag", None) is False
            for module in block.modules()
        ):
            raise ValueError(
                "Inner thinking requires batch-first blocks for batched token states."
            )

    @staticmethod
    def validate_attention_heads(head_counts: set[int]) -> None:
        if len(head_counts) > 1:
            raise ValueError(
                "Selected attention masks require a common head count across the block."
            )

    @staticmethod
    def validate_attention_mask(
        mask: Tensor,
        indices: Tensor,
        sequence_length: int,
        *,
        select_keys: bool,
        branch_count: int,
    ) -> None:
        if indices.ndim > 2:
            raise ValueError(
                "Attention masks require [tokens, features] or [batch, tokens, features]."
            )
        if not isinstance(mask, Tensor):
            raise TypeError("Attention mask must be a Tensor.")
        if mask.ndim not in (2, 3) or mask.shape[-2] != sequence_length:
            raise ValueError(
                "Attention mask must have rank 2 or 3 and match the token count."
            )
        if select_keys and mask.shape[-1] != sequence_length:
            raise ValueError(
                "Self-attention mask source length must match the token count."
            )
        if mask.device != indices.device:
            raise ValueError("Attention mask must be on the hidden device.")
        if mask.ndim == 3 and mask.shape[0] not in (1, branch_count):
            raise ValueError(
                "Attention mask leading dimension must be 1 or batch * heads."
            )

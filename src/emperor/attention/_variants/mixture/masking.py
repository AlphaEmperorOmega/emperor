"""Private mixture-of-attention-heads masking implementation."""

from typing import TYPE_CHECKING

from torch import Tensor

from emperor.attention._ops.masking import Mask
from emperor.attention._variants.mixture.validation import (
    MixtureOfAttentionHeadsValidator,
)

if TYPE_CHECKING:
    from emperor.attention._runtime import AttentionMasks, AttentionRuntimeShape
    from emperor.attention._variants.mixture.config import (
        MixtureOfAttentionHeadsConfig,
    )


class MixtureOfAttentionHeadsMask(Mask):
    VALIDATOR = MixtureOfAttentionHeadsValidator

    def __init__(self, cfg: "MixtureOfAttentionHeadsConfig"):
        super().__init__(cfg)
        self.top_k = self.cfg.experts_config.top_k

    def _validate_mask_shapes(
        self,
        key_padding_mask: Tensor | None,
        attention_mask: Tensor | None,
        runtime_shape: "AttentionRuntimeShape",
    ) -> None:
        batch_size = runtime_shape.batch_size
        standard_branch_count = runtime_shape.branch_count(self.num_heads)
        expert_branch_count = standard_branch_count * self.top_k
        self.VALIDATOR.validate_mixture_mask_shapes(
            key_padding_mask,
            attention_mask,
            expected_key_padding_shape=(
                batch_size,
                runtime_shape.source_sequence_length,
            ),
            expected_attention_sequence_shape=(
                runtime_shape.target_sequence_length,
                runtime_shape.source_sequence_length,
            ),
            standard_branch_count=standard_branch_count,
            expert_branch_count=expert_branch_count,
        )

    def merge_padding_and_attention_mask(
        self,
        key: Tensor,
        masks: "AttentionMasks",
        runtime_shape: "AttentionRuntimeShape",
    ) -> Tensor | None:
        batch_size = runtime_shape.batch_size
        target_sequence_length = runtime_shape.target_sequence_length
        source_sequence_length = key.size(-2)
        attention_mask = self.__normalize_attention_mask(
            masks.attention_mask,
            source_sequence_length,
            batch_size,
            target_sequence_length,
        )
        key_padding_mask = self.__normalize_key_padding_mask(
            masks.key_padding_mask,
            source_sequence_length,
            batch_size,
        )

        if key_padding_mask is None:
            return attention_mask
        if attention_mask is None:
            return key_padding_mask
        return attention_mask + key_padding_mask

    def __normalize_attention_mask(
        self,
        attention_mask: Tensor | None,
        source_sequence_length: int,
        batch_size: int,
        target_sequence_length: int,
    ) -> Tensor | None:
        if attention_mask is None:
            return None

        expected_sequence_shape = (
            target_sequence_length,
            source_sequence_length,
        )
        if tuple(attention_mask.shape[-2:]) != expected_sequence_shape:
            raise RuntimeError(
                "attention_mask must have target/source dimensions "
                f"{expected_sequence_shape}, got {tuple(attention_mask.shape[-2:])}."
            )

        branch_count = batch_size * self.top_k * self.num_heads
        if attention_mask.dim() == 2:
            return attention_mask.expand(
                batch_size,
                self.top_k,
                self.num_heads,
                *expected_sequence_shape,
            ).reshape(
                branch_count,
                *expected_sequence_shape,
            )

        if attention_mask.dim() != 3:
            raise RuntimeError(
                "attention_mask must be 2-D or 3-D for mixture of attention "
                f"heads, got {attention_mask.dim()}-D."
            )

        standard_branch_count = batch_size * self.num_heads
        leading_dimension = attention_mask.size(0)
        if leading_dimension == 1:
            return attention_mask.expand(branch_count, -1, -1).reshape(
                branch_count, *expected_sequence_shape
            )
        if leading_dimension == branch_count:
            return attention_mask.reshape(branch_count, *expected_sequence_shape)
        if leading_dimension == standard_branch_count:
            return (
                attention_mask.view(
                    batch_size,
                    self.num_heads,
                    *expected_sequence_shape,
                )
                .unsqueeze(1)
                .expand(
                    -1,
                    self.top_k,
                    -1,
                    -1,
                    -1,
                )
                .reshape(branch_count, *expected_sequence_shape)
            )

        raise RuntimeError(
            "3-D attention_mask leading dimension must be batch_size * num_heads "
            "or batch_size * top_k * num_heads "
            f"({standard_branch_count} or {branch_count}), got "
            f"{leading_dimension}."
        )

    def __normalize_key_padding_mask(
        self,
        key_padding_mask: Tensor | None,
        source_sequence_length: int,
        batch_size: int,
    ) -> Tensor | None:
        if key_padding_mask is None:
            return None

        expected_shape = (batch_size, source_sequence_length)
        if tuple(key_padding_mask.shape) != expected_shape:
            raise RuntimeError(
                f"key_padding_mask must have shape {expected_shape}, got "
                f"{tuple(key_padding_mask.shape)}."
            )

        branch_count = batch_size * self.top_k * self.num_heads
        return key_padding_mask.repeat_interleave(
            self.top_k * self.num_heads,
            dim=0,
        ).reshape(branch_count, 1, source_sequence_length)

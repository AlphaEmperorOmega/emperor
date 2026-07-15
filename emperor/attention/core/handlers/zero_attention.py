from dataclasses import replace
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor

if TYPE_CHECKING:
    from emperor.attention.core.config import MultiHeadAttentionConfig
    from emperor.attention.core.runtime import (
        QKV,
        AttentionMasks,
        AttentionRuntimeShape,
    )


class ZeroAttention:
    def __init__(self, cfg: "MultiHeadAttentionConfig"):
        self.cfg = cfg
        self.batch_size = self.cfg.batch_size
        self.num_heads = self.cfg.num_heads
        self.zero_attention_flag = self.cfg.zero_attention_flag

    def add_zero_attention(
        self,
        qkv: "QKV",
        masks: "AttentionMasks",
        runtime_shape: "AttentionRuntimeShape | None" = None,
    ) -> tuple["QKV", "AttentionMasks", "AttentionRuntimeShape | None"]:
        if not self.zero_attention_flag:
            return qkv, masks, runtime_shape

        padded_key = self._concatenate_zeros_tensor(qkv.key, runtime_shape)
        padded_value = self._concatenate_zeros_tensor(qkv.value, runtime_shape)
        key_padding_mask = masks.key_padding_mask
        attention_mask = masks.attention_mask
        if key_padding_mask is not None:
            key_padding_mask = F.pad(key_padding_mask, (0, 1))
        if attention_mask is not None:
            attention_mask = F.pad(attention_mask, (0, 1))

        updated_runtime_shape = (
            runtime_shape.with_source_extension() if runtime_shape is not None else None
        )
        return (
            replace(qkv, key=padded_key, value=padded_value),
            replace(
                masks,
                key_padding_mask=key_padding_mask,
                attention_mask=attention_mask,
            ),
            updated_runtime_shape,
        )

    def _concatenate_zeros_tensor(
        self,
        tensor: Tensor,
        runtime_shape: "AttentionRuntimeShape | None" = None,
    ) -> Tensor:
        head_dim = tensor.size(-1)
        zero_attention_shape = (self._get_branch_count(runtime_shape), 1, head_dim)
        zeros_tensor = tensor.new_zeros(zero_attention_shape)
        return torch.cat([tensor, zeros_tensor], dim=1)

    def _get_branch_count(
        self, runtime_shape: "AttentionRuntimeShape | None" = None
    ) -> int:
        batch_size = (
            runtime_shape.batch_size if runtime_shape is not None else self.batch_size
        )
        return batch_size * self.num_heads

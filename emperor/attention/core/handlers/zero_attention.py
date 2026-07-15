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

        updated_qkv = self.__append_zero_attention_to_key_value(qkv, runtime_shape)
        updated_masks = self.__pad_masks_for_zero_attention(masks)
        updated_runtime_shape = self.__extend_runtime_shape(runtime_shape)
        return updated_qkv, updated_masks, updated_runtime_shape

    def __append_zero_attention_to_key_value(
        self,
        qkv: "QKV",
        runtime_shape: "AttentionRuntimeShape | None",
    ) -> "QKV":
        padded_key = self.__concatenate_zeros_tensor(qkv.key, runtime_shape)
        padded_value = self.__concatenate_zeros_tensor(qkv.value, runtime_shape)
        updated_qkv = replace(qkv, key=padded_key, value=padded_value)
        return updated_qkv

    def __concatenate_zeros_tensor(
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

    def __pad_masks_for_zero_attention(
        self,
        masks: "AttentionMasks",
    ) -> "AttentionMasks":
        key_padding_mask = self.__pad_mask(masks.key_padding_mask)
        attention_mask = self.__pad_mask(masks.attention_mask)
        updated_masks = replace(
            masks, key_padding_mask=key_padding_mask, attention_mask=attention_mask
        )
        return updated_masks

    @staticmethod
    def __pad_mask(mask: Tensor | None) -> Tensor | None:
        if mask is None:
            return None
        zero_attention_padding = (0, 1)
        return F.pad(mask, zero_attention_padding)

    @staticmethod
    def __extend_runtime_shape(
        runtime_shape: "AttentionRuntimeShape | None",
    ) -> "AttentionRuntimeShape | None":
        if runtime_shape is None:
            return None
        return runtime_shape.with_source_extension()

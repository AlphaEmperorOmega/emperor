"""Private attention bias operations."""

from dataclasses import replace
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor

from emperor.attention._validation import AttentionValidatorBase
from emperor.nn import Module

if TYPE_CHECKING:
    from emperor.attention._config import MultiHeadAttentionConfig
    from emperor.attention._runtime import (
        QKV,
        AttentionMasks,
        AttentionRuntimeShape,
    )


class KeyValueBias(Module):
    VALIDATOR = AttentionValidatorBase

    def __init__(self, cfg: "MultiHeadAttentionConfig"):
        super().__init__()
        self.cfg = cfg
        self.batch_size = self.cfg.batch_size
        self.num_heads = self.cfg.num_heads
        self.embedding_dim = self.cfg.embedding_dim
        self.query_key_projection_dim = self.cfg.query_key_projection_dim
        self.value_projection_dim = self.cfg.value_projection_dim
        self.add_key_value_bias_flag = self.cfg.add_key_value_bias_flag
        self.__resolve_kv_dimensions()

        self.key_bias_vector, self.value_bias_vector = self.__build_kv_bias_vectors()

    def __resolve_kv_dimensions(self) -> None:
        if not self.query_key_projection_dim:
            self.query_key_projection_dim = self.embedding_dim
        if not self.value_projection_dim:
            self.value_projection_dim = self.embedding_dim

    def __build_kv_bias_vectors(self):
        if not self.add_key_value_bias_flag:
            return None, None
        bias_k = self._init_parameter_bank((1, 1, self.query_key_projection_dim))
        bias_v = self._init_parameter_bank((1, 1, self.value_projection_dim))
        return bias_k, bias_v

    def add_kv_learnable_bias_vectors(
        self,
        qkv: "QKV",
        masks: "AttentionMasks",
        runtime_shape: "AttentionRuntimeShape | None" = None,
    ) -> tuple["QKV", "AttentionMasks", "AttentionRuntimeShape | None"]:
        if not self.add_key_value_bias_flag:
            return qkv, masks, runtime_shape
        updated_qkv = self.__append_bias_vectors(qkv, runtime_shape)
        updated_masks = self.__pad_masks_for_bias_vector(masks)
        updated_runtime_shape = self.__extend_runtime_shape(runtime_shape)
        return updated_qkv, updated_masks, updated_runtime_shape

    def __append_bias_vectors(
        self,
        qkv: "QKV",
        runtime_shape: "AttentionRuntimeShape | None",
    ) -> "QKV":
        key_with_bias_vector = self.__append_bias_vector(
            self.key_bias_vector, qkv.key, runtime_shape
        )
        value_with_bias_vector = self.__append_bias_vector(
            self.value_bias_vector, qkv.value, runtime_shape
        )
        updated_qkv = replace(
            qkv, key=key_with_bias_vector, value=value_with_bias_vector
        )
        return updated_qkv

    def __append_bias_vector(
        self,
        bias_vector: Tensor,
        projection: Tensor,
        runtime_shape: "AttentionRuntimeShape | None",
    ) -> Tensor:
        expanded_bias = self._expand_bias_vector(bias_vector, projection, runtime_shape)
        projection_with_bias_vector = torch.cat([projection, expanded_bias], dim=1)
        return projection_with_bias_vector

    def _expand_bias_vector(
        self,
        bias_vector: Tensor,
        projection: Tensor,
        runtime_shape: "AttentionRuntimeShape | None" = None,
    ) -> Tensor:
        batch_size = self.__resolve_batch_size(runtime_shape)
        branch_count = projection.size(0)
        expected_branch_count = batch_size * self.num_heads
        self.VALIDATOR.validate_attention_ready_projection_branch_count(
            branch_count,
            expected_branch_count,
        )
        head_dim = projection.size(-1)
        bias_by_head = bias_vector.reshape(self.num_heads, head_dim)
        expanded_bias = bias_by_head.repeat(batch_size, 1)
        return expanded_bias.reshape(branch_count, 1, head_dim)

    def __resolve_batch_size(
        self,
        runtime_shape: "AttentionRuntimeShape | None",
    ) -> int:
        if runtime_shape is not None:
            return runtime_shape.batch_size
        return self.batch_size

    def __pad_masks_for_bias_vector(self, masks: "AttentionMasks") -> "AttentionMasks":
        key_padding_mask = self.__pad_mask(masks.key_padding_mask)
        attention_mask = self.__pad_mask(masks.attention_mask)
        updated_masks = replace(
            masks,
            key_padding_mask=key_padding_mask,
            attention_mask=attention_mask,
        )
        return updated_masks

    @staticmethod
    def __pad_mask(mask: Tensor | None) -> Tensor | None:
        if mask is None:
            return None
        return F.pad(mask, (0, 1))

    @staticmethod
    def __extend_runtime_shape(
        runtime_shape: "AttentionRuntimeShape | None",
    ) -> "AttentionRuntimeShape | None":
        if runtime_shape is not None:
            return runtime_shape.with_source_extension()
        return None

"""Private base attention layer implementation."""

from dataclasses import replace
from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor

from emperor.attention._ops.batching import BatchDimensionManager
from emperor.attention._ops.bias import KeyValueBias
from emperor.attention._ops.zero_attention import ZeroAttention
from emperor.attention._runtime import QKV, AttentionMasks, MultiHeadAttentionInputs
from emperor.attention._validation import MultiHeadAttentionValidator
from emperor.layers import RowLayout
from emperor.nn import Module

if TYPE_CHECKING:
    from emperor.attention._config import MultiHeadAttentionConfig
    from emperor.attention._runtime import AttentionRuntimeLayout
    from emperor.config import ModelConfig


class MultiHeadAttentionAbstract(Module):
    VALIDATOR = MultiHeadAttentionValidator
    BIAS_HANDLER = KeyValueBias
    ZERO_ATTENTION_HANDLER = ZeroAttention

    def __init__(
        self,
        cfg: "MultiHeadAttentionConfig | ModelConfig",
        overrides: "MultiHeadAttentionConfig | None" = None,
    ):
        super().__init__()
        config = getattr(cfg, "multi_head_attention_model_config", cfg)
        self.cfg: MultiHeadAttentionConfig = self._override_config(config, overrides)

        self.num_heads = self.cfg.num_heads
        self.batch_size = self.cfg.batch_size
        self.embedding_dim = self.cfg.embedding_dim
        self.target_dtype = self.cfg.target_dtype
        self.dropout_probability = self.cfg.dropout_probability
        self.query_key_projection_dim = self.cfg.query_key_projection_dim
        self.value_projection_dim = self.cfg.value_projection_dim
        self.target_sequence_length = self.cfg.target_sequence_length
        self.source_sequence_length = self.cfg.source_sequence_length
        self.zero_attention_flag = self.cfg.zero_attention_flag
        self.add_key_value_bias_flag = self.cfg.add_key_value_bias_flag
        self.causal_attention_mask_flag = self.cfg.causal_attention_mask_flag
        self.average_attention_weights_flag = self.cfg.average_attention_weights_flag
        self.return_attention_weights_flag = self.cfg.return_attention_weights_flag
        self.batch_first_flag = self.cfg.batch_first_flag

        self.VALIDATOR.validate(self)
        self.head_dim = self.embedding_dim // self.num_heads

        self.batch_manager = BatchDimensionManager(self.cfg)
        self.bias = self.BIAS_HANDLER(self.cfg)
        self.zero_attention = self.ZERO_ATTENTION_HANDLER(self.cfg)
        self._build_attention_components()
        self.to(dtype=self.target_dtype)

    def _build_attention_components(self) -> None:
        raise NotImplementedError(
            "_build_attention_components must be implemented by subclass."
        )

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        k_padding_mask: Tensor | None = None,
        attention_mask: Tensor | None = None,
        static_k: Tensor | None = None,
        static_v: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        self.projector.get_auxiliary_loss_and_clear()
        qkv = QKV(query=q, key=k, value=v)
        masks = AttentionMasks(
            key_padding_mask=k_padding_mask,
            attention_mask=attention_mask,
        )
        self.VALIDATOR.validate_forward_inputs(self, qkv, masks)
        attention_inputs = MultiHeadAttentionInputs(
            query=qkv.query,
            key=qkv.key,
            value=qkv.value,
            key_padding_mask=masks.key_padding_mask,
            attention_mask=masks.attention_mask,
            static_key=static_k,
            static_value=static_v,
        )
        attention_inputs = self.batch_manager.convert_inputs_to_internal_layout(
            attention_inputs
        )
        runtime_layout = cast(
            "AttentionRuntimeLayout",
            attention_inputs.runtime_layout,
        )
        qkv = QKV(
            query=attention_inputs.query,
            key=attention_inputs.key,
            value=attention_inputs.value,
        )
        masks = AttentionMasks(
            key_padding_mask=attention_inputs.key_padding_mask,
            attention_mask=attention_inputs.attention_mask,
        )
        self.VALIDATOR.validate_runtime_tensors(self, qkv)
        self.VALIDATOR.validate_static_key_value_inputs(
            self, qkv, static_k, static_v, runtime_layout
        )
        self.VALIDATOR.validate_runtime_layout(self, runtime_layout)
        attention_inputs = self.masks.prepare_attention_masks(attention_inputs)
        masks = AttentionMasks(
            key_padding_mask=attention_inputs.key_padding_mask,
            attention_mask=attention_inputs.attention_mask,
        )
        runtime_layout = self.__attach_projection_row_layout(
            qkv, masks, runtime_layout, static_k=static_k, static_v=static_v
        )
        qkv = self.projector.compute_qkv_projections(qkv, runtime_layout=runtime_layout)
        qkv = self.reshaper.reshape_qkv_for_attention(
            qkv, static_k, static_v, runtime_layout
        )
        qkv, masks, runtime_layout = self.bias.add_kv_learnable_bias_vectors(
            qkv, masks, runtime_layout
        )
        qkv, masks, runtime_layout = self.zero_attention.add_zero_attention(
            qkv, masks, runtime_layout
        )
        merged_attention_mask = self.masks.merge_padding_and_attention_mask(
            qkv.key, masks, runtime_layout
        )
        attention_output, attention_weights = self.processor.compute_attention(
            qkv, merged_attention_mask, runtime_layout
        )
        attention_inputs = replace(
            attention_inputs,
            runtime_layout=runtime_layout,
        )
        attention_output = self.batch_manager.restore_output_layout(
            attention_output, attention_inputs
        )
        auxiliary_loss = self.projector.get_auxiliary_loss_and_clear()
        return attention_output, attention_weights, auxiliary_loss

    def __attach_projection_row_layout(
        self,
        qkv: QKV,
        masks: AttentionMasks,
        runtime_layout: "AttentionRuntimeLayout",
        *,
        static_k: Tensor | None,
        static_v: Tensor | None,
    ) -> "AttentionRuntimeLayout":
        is_self_attention = qkv.query is qkv.key and qkv.key is qkv.value
        static_key_is_provided = static_k is not None
        static_value_is_provided = static_v is not None
        static_source_is_provided = static_key_is_provided or static_value_is_provided
        valid_projection_rows = None
        if is_self_attention and not static_source_is_provided:
            valid_projection_rows = self.__flatten_valid_self_attention_rows(
                masks.key_padding_mask,
                runtime_layout,
            )

        qkv_inputs_are_not_shared = not is_self_attention
        attention_mask_is_present = masks.attention_mask is not None
        context_sharing_restricted = (
            qkv_inputs_are_not_shared
            or static_key_is_provided
            or static_value_is_provided
            or attention_mask_is_present
        )
        projection_row_layout = RowLayout.sequence(
            leading_shape=(
                runtime_layout.target_sequence_length,
                runtime_layout.batch_size,
            ),
            batch_axis=1,
            sequence_axis=0,
            valid_rows=valid_projection_rows,
            context_sharing_restricted=context_sharing_restricted,
        )
        return runtime_layout.with_row_layout(projection_row_layout)

    @staticmethod
    def __flatten_valid_self_attention_rows(
        key_padding_mask: Tensor | None,
        runtime_layout,
    ) -> Tensor | None:
        if key_padding_mask is None:
            return None
        expected_shape = (
            runtime_layout.batch_size,
            runtime_layout.source_sequence_length,
        )
        if tuple(key_padding_mask.shape) != expected_shape:
            raise ValueError(
                "Prepared key padding mask must align with attention source rows, "
                f"expected {expected_shape}, received "
                f"{tuple(key_padding_mask.shape)}."
            )
        valid_batch_major_rows = ~torch.isneginf(key_padding_mask)
        return valid_batch_major_rows.transpose(0, 1).reshape(-1)

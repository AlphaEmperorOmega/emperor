"""Private base attention layer implementation."""

from dataclasses import replace
from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor

from emperor.attention._ops.batching import BatchDimensionManager
from emperor.attention._ops.bias import KeyValueBias
from emperor.attention._ops.zero_attention import ZeroAttention
from emperor.attention._runtime import MultiHeadAttentionInputs
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

    def _run_attention(
        self,
        attention_inputs: MultiHeadAttentionInputs,
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        self.projector._clear_transient_state()
        try:
            self.VALIDATOR.validate_forward_inputs(self, attention_inputs)
            attention_inputs = self.batch_manager.convert_inputs_to_internal_layout(
                attention_inputs
            )
            self.VALIDATOR.validate_runtime_tensors(self, attention_inputs)
            self.VALIDATOR.validate_static_key_value_inputs(self, attention_inputs)
            self.VALIDATOR.validate_runtime_layout(self, attention_inputs)
            attention_inputs = self.masks.prepare_attention_masks(attention_inputs)
            attention_inputs = self.__attach_projection_row_layout(attention_inputs)
            attention_inputs = self.projector.compute_qkv_projections(attention_inputs)
            attention_inputs = self.reshaper.reshape_qkv_for_attention(attention_inputs)
            attention_inputs = self.bias.add_kv_learnable_bias_vectors(attention_inputs)
            attention_inputs = self.zero_attention.add_zero_attention(attention_inputs)
            attention_inputs = self.masks.merge_padding_and_attention_mask(
                attention_inputs
            )
            attention_output, attention_weights = self.processor.compute_attention(
                attention_inputs
            )
            attention_output = self.batch_manager.restore_output_layout(
                attention_output, attention_inputs
            )
            auxiliary_loss = self.projector._get_auxiliary_loss()
            return attention_output, attention_weights, auxiliary_loss
        finally:
            self.projector._clear_transient_state()

    def __attach_projection_row_layout(
        self,
        attention_inputs: MultiHeadAttentionInputs,
    ) -> MultiHeadAttentionInputs:
        runtime_layout = attention_inputs.runtime_layout
        self.VALIDATOR.validate_projection_row_layout_runtime_layout(runtime_layout)
        runtime_layout = cast("AttentionRuntimeLayout", runtime_layout)
        is_self_attention = (
            attention_inputs.query is attention_inputs.key
            and attention_inputs.key is attention_inputs.value
        )
        static_key_is_provided = attention_inputs.static_key is not None
        static_value_is_provided = attention_inputs.static_value is not None
        static_source_is_provided = static_key_is_provided or static_value_is_provided
        valid_projection_rows = None
        if is_self_attention and not static_source_is_provided:
            valid_projection_rows = self.__flatten_valid_self_attention_rows(
                attention_inputs.key_padding_mask,
                runtime_layout,
            )

        query_key_value_inputs_are_not_shared = not is_self_attention
        attention_mask_is_present = attention_inputs.attention_mask is not None
        context_sharing_restricted = (
            query_key_value_inputs_are_not_shared
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
        return replace(
            attention_inputs,
            runtime_layout=runtime_layout.with_row_layout(projection_row_layout),
        )

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

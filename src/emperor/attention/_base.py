"""Private base attention layer implementation."""

from typing import TYPE_CHECKING

from torch import Tensor

from emperor.attention._ops.batching import BatchDimensionManager
from emperor.attention._ops.bias import KeyValueBias
from emperor.attention._ops.projection_layout import ProjectionRowLayoutManager
from emperor.attention._ops.zero_attention import ZeroAttention
from emperor.attention._runtime import MultiHeadAttentionInputs
from emperor.attention._validation import MultiHeadAttentionValidator
from emperor.nn import Module

if TYPE_CHECKING:
    from emperor.attention._config import MultiHeadAttentionConfig
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
        self.layout_manager = ProjectionRowLayoutManager(self.VALIDATOR)
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
            return self.__execute_attention_pipeline(attention_inputs)
        finally:
            self.projector._clear_transient_state()

    def __execute_attention_pipeline(
        self,
        attention_inputs: MultiHeadAttentionInputs,
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        self.VALIDATOR.validate_forward_inputs(self, attention_inputs)
        attention_inputs = self.batch_manager.convert_inputs_to_internal_layout(
            attention_inputs
        )
        self.VALIDATOR.validate_runtime_tensors(self, attention_inputs)
        self.VALIDATOR.validate_static_key_value_inputs(self, attention_inputs)
        self.VALIDATOR.validate_runtime_layout(self, attention_inputs)
        attention_inputs = self.masks.prepare_attention_masks(attention_inputs)
        attention_inputs = self.layout_manager.attach_projection_row_layout(
            attention_inputs
        )
        attention_inputs = self.projector.compute_qkv_projections(attention_inputs)
        attention_inputs = self.reshaper.reshape_qkv_for_attention(attention_inputs)
        attention_inputs = self.bias.add_kv_learnable_bias_vectors(attention_inputs)
        attention_inputs = self.zero_attention.add_zero_attention(attention_inputs)
        attention_inputs = self.masks.merge_padding_and_attention_mask(attention_inputs)
        attention_output, attention_weights = self.processor.compute_attention(
            attention_inputs
        )
        attention_output = self.batch_manager.restore_output_layout(
            attention_output, attention_inputs
        )
        auxiliary_loss = self.projector._get_auxiliary_loss()
        return attention_output, attention_weights, auxiliary_loss

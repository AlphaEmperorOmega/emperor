from typing import TYPE_CHECKING

from torch import Tensor

from emperor.attention.core._validator import MultiHeadAttentionValidator
from emperor.attention.core.handlers.batch import BatchDimensionManager
from emperor.attention.core.handlers.bias import KeyValueBias
from emperor.attention.core.handlers.zero_attention import ZeroAttention
from emperor.base.module import Module

if TYPE_CHECKING:
    from emperor.attention.core.config import MultiHeadAttentionConfig
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
        self.head_dim = self.embedding_dim // self.num_heads

        self.VALIDATOR.validate(self)

        self.batch_manager = BatchDimensionManager(self.cfg)
        self.bias = self.BIAS_HANDLER(self.cfg)
        self.zero_attention = self.ZERO_ATTENTION_HANDLER(self.cfg)
        self._build_attention_components()

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
        q, k, v = self.batch_manager.enforce_batch_as_second_dim(q, k, v)
        self.VALIDATOR.validate_forward_inputs(
            self, q, k, v, k_padding_mask, attention_mask
        )
        q, k, v, k_padding_mask, attention_mask = (
            self.batch_manager.add_batch_dimension_if_missing(
                q, k, v, k_padding_mask, attention_mask
            )
        )
        attention_mask = self.masks.resolve_causal_attention_mask(q, attention_mask)
        k_padding_mask, attention_mask = self.masks.process_attention_masks(
            k_padding_mask, attention_mask
        )
        q, k, v = self.projector.compute_qkv_projections(q, k, v)
        k, v, k_padding_mask, attention_mask = self.bias.add_kv_learnable_bias_vectors(
            k, v, k_padding_mask, attention_mask
        )
        q, k, v = self.reshaper.reshape_qkv_for_attention(q, k, v, static_k, static_v)
        k, v, attention_mask, k_padding_mask = self.zero_attention.add_zero_attention(
            k, v, attention_mask, k_padding_mask
        )
        merged_masks = self.masks.merge_padding_and_attention_mask(
            k, k_padding_mask, attention_mask
        )
        attention_output, attention_weights = self.processor.compute_attention(
            q, k, v, merged_masks
        )
        attention_output = self.batch_manager.reverse_enforced_batch_as_second_dim(
            attention_output
        )
        auxiliary_loss = self.projector.get_auxiliary_loss_and_clear()
        return attention_output, attention_weights, auxiliary_loss

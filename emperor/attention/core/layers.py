from typing import TYPE_CHECKING

from torch import Tensor

from emperor.attention.core._validator import MultiHeadAttentionValidator
from emperor.attention.core.handlers.batch import BatchDimensionManager
from emperor.attention.core.handlers.bias import KeyValueBias
from emperor.attention.core.handlers.zero_attention import ZeroAttention
from emperor.attention.core.runtime import QKV, AttentionMasks
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
        self.batch_first_flag = self.cfg.batch_first_flag
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
        self.projector.get_auxiliary_loss_and_clear()
        qkv = QKV(query=q, key=k, value=v)
        masks = AttentionMasks(
            key_padding_mask=k_padding_mask,
            attention_mask=attention_mask,
        )
        self.VALIDATOR.validate_forward_inputs(self, qkv, masks)
        qkv, masks, runtime_shape = (
            self.batch_manager.convert_inputs_to_internal_layout(
                qkv, masks, static_keys=static_k
            )
        )
        self.VALIDATOR.validate_runtime_tensors(self, qkv)
        self.VALIDATOR.validate_static_key_value_inputs(
            self, qkv, static_k, static_v, runtime_shape
        )
        self.VALIDATOR.validate_runtime_shape(self, runtime_shape)
        masks = self.masks.prepare_attention_masks(qkv.query, masks, runtime_shape)
        qkv = self.projector.compute_qkv_projections(qkv)
        qkv = self.reshaper.reshape_qkv_for_attention(
            qkv, static_k, static_v, runtime_shape
        )
        qkv, masks, runtime_shape = self.bias.add_kv_learnable_bias_vectors(
            qkv, masks, runtime_shape
        )
        qkv, masks, runtime_shape = self.zero_attention.add_zero_attention(
            qkv, masks, runtime_shape
        )
        merged_attention_mask = self.masks.merge_padding_and_attention_mask(
            qkv.key, masks, runtime_shape
        )
        attention_output, attention_weights = self.processor.compute_attention(
            qkv, merged_attention_mask, runtime_shape
        )
        attention_output = self.batch_manager.restore_output_layout(
            attention_output, runtime_shape
        )
        auxiliary_loss = self.projector.get_auxiliary_loss_and_clear()
        return attention_output, attention_weights, auxiliary_loss

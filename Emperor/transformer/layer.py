from torch import Tensor
from dataclasses import dataclass, field
from Emperor.attention.attention import MultiHeadAttention
from Emperor.base.enums import LayerNormPositionOptions
from Emperor.feedForward.feed_forward import FeedForward
from Emperor.base.utils import DataClassBase, Module
from Emperor.layers.utils.base import (
    FeedForwardLayerBlock,
    LayerBlock,
    MultiHeadAttentionCrossAttentionLayerBlock,
    MultiHeadAttentionSelfAttentionLayerBlock,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.config import ModelConfig


@dataclass
class TransformerLayerConfig(DataClassBase):
    layer_norm_position: "LayerNormPositionOptions | None" = field(
        default=None,
        metadata={"help": ""},
    )
    layer_norm_dim: int | None = field(
        default=None,
        metadata={"help": ""},
    )
    dropout_probability: float | None = field(
        default=None,
        metadata={"help": ""},
    )


class TransformerLayerBase(Module):
    def __init__(
        self,
        cfg: "TransformerLayerConfig | ModelConfig",
        overrides: "TransformerLayerConfig | None" = None,
    ):
        super().__init__()
        config = getattr(cfg, "transformer_layer_config", cfg)
        self.cfg: "TransformerLayerConfig" = self._overwrite_config(config, overrides)
        self.layer_norm_dim = self.cfg.layer_norm_dim
        self.layer_norm_position = self.cfg.layer_norm_position
        self.dropout_probability = self.cfg.dropout_probability

    def _create_self_attn_model(
        self, model: MultiHeadAttention | FeedForward
    ) -> LayerBlock:
        return MultiHeadAttentionSelfAttentionLayerBlock(
            model=model,
            residual_connection_flag=True,
            layer_norm_dim=self.layer_norm_dim,
            layer_norm_position=self.layer_norm_position,
            dropout_probability=self.dropout_probability,
        )

    def _create_cross_attn_model(
        self, model: MultiHeadAttention | FeedForward
    ) -> LayerBlock:
        return MultiHeadAttentionCrossAttentionLayerBlock(
            model=model,
            residual_connection_flag=True,
            layer_norm_dim=self.layer_norm_dim,
            layer_norm_position=self.layer_norm_position,
            dropout_probability=self.dropout_probability,
        )

    def _create_ff_model(self, model: MultiHeadAttention | FeedForward) -> LayerBlock:
        return FeedForwardLayerBlock(
            model=model,
            residual_connection_flag=True,
            layer_norm_dim=self.layer_norm_dim,
            layer_norm_position=self.layer_norm_position,
            dropout_probability=self.dropout_probability,
        )


class TransformerEncoderLayer(TransformerLayerBase):
    def __init__(
        self,
        cfg: "TransformerLayerConfig | ModelConfig",
        overrides: "TransformerLayerConfig | None" = None,
    ):
        super().__init__(cfg, overrides)

        attention = MultiHeadAttention(cfg)
        self.attention_model = self._create_self_attn_model(attention)
        feed_forward = FeedForward(cfg)
        self.feed_forward_model = self._create_ff_model(feed_forward)

    def forward(
        self,
        input_tensor: Tensor,
        key_padding_mask: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        additional_model_inputs = {
            "k_padding_mask": key_padding_mask,
            "attention_mask": attention_mask,
        }
        x, attn_loss = self.attention_model(input_tensor, additional_model_inputs)
        x, ff_loss = self.feed_forward_model(x)
        return x, attn_loss + ff_loss


class TransformerDecoderLayer(TransformerLayerBase):
    def __init__(
        self,
        cfg: "TransformerLayerConfig | ModelConfig",
        overrides: "TransformerLayerConfig | None" = None,
    ):
        super().__init__(cfg, overrides)

        # import torch.nn as nn
        # nn.TransformerDecoderLayer
        self.self_attention_model = self._create_self_attn_model(
            MultiHeadAttention(cfg)
        )
        self.cross_attention_model = self._create_cross_attn_model(
            MultiHeadAttention(cfg)
        )
        self.feed_forward_model = self._create_ff_model(FeedForward(cfg))

    def forward(
        self,
        input_tensor: Tensor,
        memory_tensor: Tensor,
        key_padding_mask: Tensor | None = None,
        memory_padding_mask: Tensor | None = None,
        attention_mask: Tensor | None = None,
        memory_attention_mask: Tensor | None = None,
    ) -> Tensor:
        additional_self_attention_model_inputs = {
            "k_padding_mask": key_padding_mask,
            "attention_mask": attention_mask,
        }
        x = self.self_attention_model(
            input_tensor, additional_self_attention_model_inputs
        )
        additional_model_inputs = {
            "k": memory_tensor,
            "v": memory_tensor,
            "k_padding_mask": memory_padding_mask,
            "attention_mask": memory_attention_mask,
        }
        x = self.cross_attention_model(x, additional_model_inputs)
        x = self.feed_forward_model(x)
        return x

import torch.nn as nn
# nn.TransformerEncoderLayer

from torch import Tensor
from dataclasses import dataclass, field
from Emperor.attention.attention import MultiHeadAttention
from Emperor.base.enums import LayerNormPositionOptions
from Emperor.feedForward.feed_forward import FeedForward
from Emperor.base.utils import DataClassBase, Module
from Emperor.layers.utils.base import LayerBlock
from Emperor.layers.utils.enums import (
    AttentionTypes,
    FeedForwardTypes,
    LayerTypes,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.config import ModelConfig


@dataclass
class TransformerLayerConfig(DataClassBase):
    model_type: LayerTypes | None = field(
        default=None,
        metadata={"help": ""},
    )
    attention_type: AttentionTypes | None = field(
        default=None,
        metadata={
            "help": "Type of attention model used to compute the transformer attention output."
        },
    )
    feed_forward_type: FeedForwardTypes | None = field(
        default=None,
        metadata={
            "help": "Type of feed forward model used to compute the transformer feed forward output"
        },
    )
    dropout_probability: float | None = field(
        default=None,
        metadata={"help": ""},
    )
    layer_norm_dim: int | None = field(
        default=None,
        metadata={"help": ""},
    )
    layer_norm_position: "LayerNormPositionOptions | None" = field(
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
        nn.TransformerDecoderLayer
        config = getattr(cfg, "multi_head_attention_model_config", cfg)
        self.cfg: "TransformerLayerConfig" = self._overwrite_config(config, overrides)
        self.layer_norm_dim = self.layer_norm_dim
        self.layer_norm_position = self.cfg.layer_norm_position
        self.dropout_probability = self.cfg.dropout_probability

    def _create_model(self, model: MultiHeadAttention | FeedForward) -> LayerBlock:
        return LayerBlock(
            model=model(cfg),
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
        super().__init__()
        nn.TransformerDecoderLayer
        config = getattr(cfg, "multi_head_attention_model_config", cfg)
        self.cfg: "TransformerLayerConfig" = self._overwrite_config(config, overrides)
        self.main_cfg = cfg
        self.attention_type = self.cfg.attention_type
        self.feed_forward_type = self.cfg.feed_forward_type
        self.dropout_probability = self.cfg.dropout_probability
        self.layer_norm_dim = self.cfg.layer
        self.layer_norm_position = self.cfg.layer_norm_position

        attention = MultiHeadAttention(cfg)
        self.attention_model = self._create_model(attention)
        feed_forward = FeedForward(cfg)
        self.feed_forward_model = self._create_model(feed_forward)

    def forward(
        self,
        input_tensor: Tensor,
        key_padding_mask: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        other_attention_inputs = (key_padding_mask, attention_mask)
        x = self.attention_model(input_tensor, other_attention_inputs)
        x = self.feed_forward_model(x)
        return x


class TransformerDecoderLayer(TransformerLayerBase):
    def __init__(
        self,
        cfg: "TransformerLayerConfig | ModelConfig",
        overrides: "TransformerLayerConfig | None" = None,
    ):
        super().__init__()
        config = getattr(cfg, "multi_head_attention_model_config", cfg)
        self.cfg: "TransformerLayerConfig" = self._overwrite_config(config, overrides)
        self.main_cfg = cfg
        self.attention_type = self.cfg.attention_type
        self.feed_forward_type = self.cfg.feed_forward_type

        self.self_attention_model = self._create_model(MultiHeadAttention(cfg))
        self.cross_attention_model = self._create_model(MultiHeadAttention(cfg))
        self.feed_forward_model = self._create_model(FeedForward(cfg))

    def forward(
        self,
        input_tensor: Tensor,
        key_padding_mask: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        other_attention_inputs = (key_padding_mask, attention_mask)
        x = self.self_attention_model(input_tensor, other_attention_inputs)
        x = self.cross_attention_model(x, other_attention_inputs)
        x = self.feed_forward_model(x)
        return x

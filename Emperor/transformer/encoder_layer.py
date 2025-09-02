# import torch.nn as nn
# nn.TransformerEncoderLayer

from torch import Tensor
from dataclasses import dataclass, field

from Emperor.layers.utils.base import LayerBlock
from Emperor.layers.utils.enums import (
    AttentionTypes,
    FeedForwardTypes,
    LayerTypes,
)
from Emperor.base.utils import DataClassBase, Module

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.config import ModelConfig


@dataclass
class TransformerEncoderLayerConfig(DataClassBase):
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
    qkv_projection_dim: int | None = field(
        default=None,
        metadata={
            "help": "Dimension of query, key, value hidden projections of the attention mechanism."
        },
    )
    feed_forward_projection_dim: int | None = field(
        default=None,
        metadata={
            "help": "Dimension applies to the feed forward module input hidden dimension."
        },
    )
    layer_norm_eps: float | None = field(
        default=None,
        metadata={"help": ""},
    )
    layer_norm_first_flag: bool | None = field(
        default=None,
        metadata={"help": ""},
    )
    dropout_probability: float | None = field(
        default=None,
        metadata={
            "help": "Dropout probability applied to attention weights (prevents overfitting)."
        },
    )


class TransformerEncoderLayer(Module):
    def __init__(
        self,
        cfg: "TransformerEncoderLayerConfig | ModelConfig",
        overrides: "TransformerEncoderLayerConfig | None" = None,
    ):
        super().__init__()
        config = getattr(cfg, "multi_head_attention_model_config", cfg)
        self.cfg: "TransformerEncoderLayerConfig" = self._overwrite_config(
            config, overrides
        )
        self.main_cfg = cfg
        self.attention_type = self.cfg.attention_type
        self.feed_forward_type = self.cfg.feed_forward_type
        self.dropout_probability = self.cfg.dropout_probability
        self.layer_norm_first_flag = self.cfg.layer_norm_first_flag

        self.attention_model = self.__create_model(self.attention_type)
        self.feed_forward_model = self.__create_model(self.feed_forward_type)

    def __create_model(
        self, model_type: AttentionTypes | FeedForwardTypes
    ) -> LayerBlock:
        return LayerBlock(
            model_type.value(self.cfg),
            residual_connection_flag=True,
            dropout_probability=self.dropout_probability,
            layer_form_first_flag=self.layer_norm_first_flag,
        )

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

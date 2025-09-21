import copy

from torch import Tensor
from torch.nn import ModuleList
from dataclasses import dataclass, field
from Emperor.base.enums import LayerNormPositionOptions
from Emperor.feedForward.feed_forward import FeedForward
from Emperor.attention.attention import MultiHeadAttention, MultiHeadAttentionConfig
from Emperor.base.utils import DataClassBase, Module
from Emperor.layers.utils.base import (
    LayerBlock,
    FeedForwardLayerBlock,
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
        source_token_embeddings: Tensor,
        source_key_padding_mask: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        additional_model_inputs = {
            "k_padding_mask": source_key_padding_mask,
            "attention_mask": attention_mask,
        }
        x, attn_loss = self.attention_model(
            source_token_embeddings, additional_model_inputs
        )
        x, ff_loss = self.feed_forward_model(x)
        return x, attn_loss + ff_loss


class TransformerDecoderLayer(TransformerLayerBase):
    def __init__(
        self,
        cfg: "TransformerLayerConfig | ModelConfig",
        overrides: "TransformerLayerConfig | None" = None,
    ):
        super().__init__(cfg, overrides)

        self.self_attention_model = self._create_self_attn_model(
            MultiHeadAttention(cfg)
        )
        corss_attention_overrides = MultiHeadAttentionConfig(
            use_separate_projection_weight_flag=True
        )
        self.cross_attention_model = self._create_cross_attn_model(
            MultiHeadAttention(cfg, corss_attention_overrides)
        )
        self.feed_forward_model = self._create_ff_model(FeedForward(cfg))

    def forward(
        self,
        target_token_embeddings: Tensor,
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
            target_token_embeddings, additional_self_attention_model_inputs
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


@dataclass
class TransformerConfig(DataClassBase):
    num_layers: int | None = field(
        default=None,
        metadata={"help": ""},
    )
    layer_norm_flag: bool | None = field(
        default=None,
        metadata={"help": ""},
    )
    enable_nested_tensor: bool | None = field(
        default=None,
        metadata={"help": ""},
    )
    source_sequence_length: bool | None = field(
        default=None,
        metadata={
            "help": "If True, use a causal mask to prevent attention to future positions (for decoding/generation)."
        },
    )
    causal_attention_mask_flag: bool | None = field(
        default=None,
        metadata={
            "help": "If True, use a causal mask to prevent attention to future positions (for decoding/generation)."
        },
    )


class TransformerEncoder(Module):
    def __init__(
        self,
        cfg: "TransformerConfig | ModelConfig",
        overrides: "TransformerConfig | None" = None,
    ):
        super().__init__()
        config = getattr(cfg, "transformer_config", cfg)
        self.cfg: "TransformerLayerConfig" = self._overwrite_config(config, overrides)

        self.num_layers = self.cfg.num_layers
        self.source_sequence_length = self.cfg.source_sequence_length
        self.causal_attention_mask_flag = self.cfg.causal_attention_mask_flag
        self.layer_norm_flag = self.cfg.layer_norm_flag

        self.layers = self._create_layers(cfg)

        # import torch.nn as nn
        # nn.TransformerEncoder

    def _create_layers(self, config: "ModelConfig") -> ModuleList:
        encoder_layer = TransformerEncoderLayer(cfg)
        return ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(self.num_layers)]
        )

    def forward(
        self,
        source_token_embeddings: Tensor,
        attention_mask: Tensor | None = None,
        key_padding_mask: bool | None = None,
    ) -> Tensor:
        is_causal = self.__detect_is_causal_mask(attention_mask, is_causal, seq_len)

        output = source_token_embeddings
        for mod in self.layers:
            output = mod(
                output,
                src_mask=mask,
                is_causal=is_causal,
                src_key_padding_mask=src_key_padding_mask_for_layers,
            )

    def __detect_is_causal_mask(
        self,
        attention_mask: Tensor,
        is_causal: bool,
    ):
        pass
        # make_causal =

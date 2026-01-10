import torch

from torch import Tensor
from dataclasses import dataclass, field
from Emperor.base.utils import ConfigBase, Module
from Emperor.transformer.utils.feed_forward import FeedForward, FeedForwardConfig
from Emperor.attention.utils.layer import MultiHeadAttention, MultiHeadAttentionConfig

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.config import ModelConfig


# TODO: Add the ability to freze old neurons or root neurons once
# new ones are added this should apply to

# TODO: When developing the transformer with recurence that is used
# for 10 times ensure that you add the ability to increate dimensionality
# across each iteration or decrease iteration across iterations just to make it clear
# - add the ability to increate the dimensionality of the vectors degenerated
# for each iterations
# - remove dimensionality as a constraint of requiering more computation for
# the current token


@dataclass
class TransformerConfig(ConfigBase):
    num_layers: int | None = field(
        default=None,
        metadata={"help": ""},
    )
    source_sequence_length: int | None = field(
        default=None,
        metadata={"help": ""},
    )
    target_sequence_length: int | None = field(
        default=None,
        metadata={"help": ""},
    )
    layer_norm_dim: int | None = field(
        default=None,
        metadata={"help": ""},
    )
    causal_attention_mask_flag: bool | None = field(
        default=None,
        metadata={"help": ""},
    )
    attention_config: "MultiHeadAttentionConfig | None" = field(
        default=None,
        metadata={"help": ""},
    )
    feed_forward_config: "FeedForwardConfig | None" = field(
        default=None,
        metadata={"help": ""},
    )


class TransformerLayerBase(Module):
    def __init__(
        self,
        cfg: "TransformerConfig | ModelConfig",
        overrides: "TransformerConfig | None" = None,
    ):
        super().__init__()
        config = getattr(cfg, "transformer_layer_config", cfg)
        self.cfg: "TransformerConfig" = self._overwrite_config(config, overrides)
        self.main_config = cfg
        self.attention_config: "MultiHeadAttentionConfig" = self.cfg.attention_config
        self.feed_forward_config: "FeedForwardConfig" = self.cfg.feed_forward_config

    def _create_self_attention_model(self) -> MultiHeadAttention:
        return MultiHeadAttention(self.attention_config)

    def _create_cross_attention_model(self) -> MultiHeadAttention:
        overrides = MultiHeadAttentionConfig(is_self_attention_projector_flag=False)
        return MultiHeadAttention(self.attention_config, overrides)

    def _create_feed_forward_model(self) -> FeedForward:
        return FeedForward(self.feed_forward_config)

    def _apply_residual_connection(self, input: Tensor, prev_input: Tensor):
        return input + prev_input


class TransformerEncoderLayer(TransformerLayerBase):
    def __init__(
        self,
        cfg: "TransformerConfig | ModelConfig",
        overrides: "TransformerConfig | None" = None,
    ):
        super().__init__(cfg, overrides)

        self.self_attention_model = self._create_self_attention_model()
        self.feed_forward_model = self._create_feed_forward_model()

    def forward(
        self,
        source_token_embeddings: Tensor,
        source_key_padding_mask: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        x_att, attn_loss = self.self_attention_model(
            q=source_token_embeddings,
            k=source_token_embeddings,
            v=source_token_embeddings,
            k_padding_mask=source_key_padding_mask,
            attention_mask=attention_mask,
        )
        x = self._apply_residual_connection(x_att, source_token_embeddings)
        x_ff, ff_loss = self.feed_forward_model(x)
        x = self._apply_residual_connection(x_ff, x)

        # FIXME: Ensure you get a tensor from the attention
        # and feed forward models, otherwise this returns an error
        # total_loss = attn_loss + ff_loss
        total_loss = torch.tensor(0.0)
        return x, total_loss


class TransformerDecoderLayer(TransformerLayerBase):
    def __init__(
        self,
        cfg: "TransformerConfig | ModelConfig",
        overrides: "TransformerConfig | None" = None,
    ):
        super().__init__(cfg, overrides)

        self.self_attention_model = self._create_self_attention_model()
        self.cross_attention_model = self._create_cross_attention_model()
        self.feed_forward_model = self._create_feed_forward_model()

    def forward(
        self,
        target_token_embeddings: Tensor,
        encoder_output: Tensor,
        key_padding_mask: Tensor | None = None,
        encoder_padding_mask: Tensor | None = None,
        attention_mask: Tensor | None = None,
        encoder_attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        # TODO: This will have to be replaced with an object for
        # a clearner implementation
        x_self, self_attn_loss = self.self_attention_model(
            q=target_token_embeddings,
            k=target_token_embeddings,
            v=target_token_embeddings,
            k_padding_mask=key_padding_mask,
            attention_mask=attention_mask,
        )
        x = self._apply_residual_connection(x_self, target_token_embeddings)
        x_corss, cross_attn_loss = self.cross_attention_model(
            q=x,
            k=encoder_output,
            v=encoder_output,
            k_padding_mask=encoder_padding_mask,
            attention_mask=encoder_attention_mask,
        )
        x = self._apply_residual_connection(x_corss, x)
        x_ff, ff_loss = self.feed_forward_model(x)
        x = self._apply_residual_connection(x_ff, x)

        # FIXME: Ensure you get a tensor from the attention and
        # feed forward models, otherwise this returns an error
        # total_loss = self_attn_loss + cross_attn_loss + ff_loss
        total_loss = torch.tensor(0.0)
        return x, total_loss

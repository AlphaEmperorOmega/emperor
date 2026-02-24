import torch

from torch import Tensor
from Emperor.attention.utils.enums import AttentionOptions
from Emperor.base.layer import Layer
from dataclasses import dataclass, field
from Emperor.base.utils import ConfigBase, Module
from Emperor.transformer.utils.feed_forward import FeedForward, FeedForwardConfig
from Emperor.embedding.options import AbsolutePositionalEmbeddingOptions
from Emperor.attention.utils.layer import MultiHeadAttention, MultiHeadAttentionConfig
from Emperor.transformer.utils.wrappers import (
    CrossAttentionLayer,
    FeedForwardLayer,
    SelfAttentionLayer,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.config import ModelConfig
    from Emperor.base.enums import LayerNormPositionOptions
    from Emperor.transformer.utils.patch.options.base import PatchConfig
    from Emperor.embedding.absolute.utils.config import (
        AbsolutePositionalEmbeddingConfig,
    )


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
    embedding_dim: int | None = field(
        default=None,
        metadata={"help": ""},
    )
    layer_norm_position: "LayerNormPositionOptions | None" = field(
        default=None,
        metadata={"help": ""},
    )
    dropout_probability: float | None = field(
        default=None,
        metadata={"help": ""},
    )
    causal_attention_mask_flag: bool | None = field(
        default=None,
        metadata={"help": ""},
    )
    positional_embedding_option: "AbsolutePositionalEmbeddingOptions | None" = field(
        default=None,
        metadata={"help": ""},
    )
    patch_config: "PatchConfig | None" = field(
        default=None,
        metadata={"help": ""},
    )
    positional_embedding_config: "AbsolutePositionalEmbeddingConfig | None" = field(
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
        self.embedding_dim = self.cfg.embedding_dim
        self.layer_norm_dim = self.embedding_dim
        self.layer_norm_position = self.cfg.layer_norm_position
        self.dropout_probability = self.cfg.dropout_probability
        self.attention_config: "MultiHeadAttentionConfig" = self.cfg.attention_config
        self.feed_forward_config: "FeedForwardConfig" = self.cfg.feed_forward_config

    def _create_self_attention_model(self) -> Layer:
        wrapper_class = SelfAttentionLayer
        overrides = MultiHeadAttentionConfig(
            attention_option=AttentionOptions.SELF_ATTENTION,
            query_key_projection_dim=self.embedding_dim,
            value_projection_dim=self.embedding_dim,
        )
        model = MultiHeadAttention(self.attention_config, overrides)
        return self.__create_layer_block(wrapper_class, model)

    def _create_cross_attention_model(self) -> Layer:
        wrapper_class = CrossAttentionLayer
        overrides = MultiHeadAttentionConfig(
            attention_option=AttentionOptions.INDEPENDENT
        )
        model = MultiHeadAttention(self.attention_config, overrides)
        return self.__create_layer_block(wrapper_class, model)

    def _create_feed_forward_model(self) -> Layer:
        wrapper_class = FeedForwardLayer
        model = FeedForward(self.feed_forward_config)
        return self.__create_layer_block(wrapper_class, model)

    def __create_layer_block(
        self,
        wrapper_class: type[Layer],
        model: MultiHeadAttention | FeedForward,
    ) -> Layer:
        return wrapper_class(
            model=model,
            residual_connection_flag=True,
            layer_norm_dim=self.layer_norm_dim,
            layer_norm_position=self.layer_norm_position,
            dropout_probability=self.dropout_probability,
        )

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
        X, attn_loss = self.self_attention_model(
            {
                "q": source_token_embeddings,
                "k": source_token_embeddings,
                "v": source_token_embeddings,
                "k_padding_mask": source_key_padding_mask,
                "attention_mask": attention_mask,
            }
        )
        X, ff_loss = self.feed_forward_model(X)

        # FIXME: Ensure you get a tensor from the attention
        # and feed forward models, otherwise this returns an error
        # total_loss = attn_loss + ff_loss
        total_loss = torch.tensor(0.0)
        return X, total_loss


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
        X, self_attn_loss = self.self_attention_model(
            {
                "q": target_token_embeddings,
                "k": target_token_embeddings,
                "v": target_token_embeddings,
                "k_padding_mask": key_padding_mask,
                "attention_mask": attention_mask,
            }
        )
        if self.__should_compute_cross_attention(encoder_output):
            X, cross_attn_loss = self.cross_attention_model(
                {
                    "q": X,
                    "k": encoder_output,
                    "v": encoder_output,
                    "k_padding_mask": encoder_padding_mask,
                    "attention_mask": encoder_attention_mask,
                }
            )
        X, ff_loss = self.feed_forward_model(X)

        # FIXME: Ensure you get a tensor from the attention and
        # feed forward models, otherwise this returns an error
        # total_loss = self_attn_loss + cross_attn_loss + ff_loss
        total_loss = torch.tensor(0.0)
        return X, total_loss

    def __should_compute_cross_attention(
        self,
        encoder_output: Tensor | None,
    ) -> bool:
        return self.cross_attention_model is None and encoder_output is None

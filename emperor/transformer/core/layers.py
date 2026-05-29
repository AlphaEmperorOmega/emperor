import torch.nn as nn

from torch import Tensor
from emperor.base.utils import Module
from emperor.base.options import LayerNormPositionOptions
from emperor.transformer.core._validator import TransformerValidator

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.transformer.core.config import (
        TransformerEncoderLayerConfig,
        TransformerDecoderLayerConfig,
    )


class TransformerEncoderLayer(Module):
    def __init__(
        self,
        cfg: "TransformerEncoderLayerConfig",
        overrides: "TransformerEncoderLayerConfig | None" = None,
    ):
        super().__init__()
        config = getattr(cfg, "transformer_encoder_layer_config", cfg)
        self.cfg: "TransformerEncoderLayerConfig" = self._override_config(
            config, overrides
        )

        self.embedding_dim: int = self.cfg.embedding_dim
        self.layer_norm_position: LayerNormPositionOptions = (
            self.cfg.layer_norm_position
        )
        self.dropout_probability: float = self.cfg.dropout_probability

        TransformerValidator.validate_encoder_layer(self)

        self.self_attention_model = self.cfg.attention_config.build()
        self.feed_forward_model = self.cfg.feed_forward_config.build()

        self.self_attention_layer_norm = nn.LayerNorm(self.embedding_dim)
        self.feed_forward_layer_norm = nn.LayerNorm(self.embedding_dim)
        self.self_attention_dropout = nn.Dropout(self.dropout_probability)
        self.feed_forward_dropout = nn.Dropout(self.dropout_probability)

    def forward(
        self,
        source_token_embeddings: Tensor,
        source_key_padding_mask: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        TransformerValidator.validate_encoder_layer_forward_inputs(
            self, source_token_embeddings
        )
        x = source_token_embeddings
        x, attention_loss = self.__apply_self_attention_sublayer(
            x,
            source_key_padding_mask=source_key_padding_mask,
            attention_mask=attention_mask,
        )
        x, feed_forward_loss = self.__apply_feed_forward_sublayer(x)
        total_loss = attention_loss + feed_forward_loss
        return x, total_loss

    def __apply_self_attention_sublayer(
        self,
        residual: Tensor,
        source_key_padding_mask: Tensor | None,
        attention_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        normed_input = self.__apply_pre_norm(
            residual, self.self_attention_layer_norm
        )
        attention_output, _attention_weights, auxiliary_loss = (
            self.self_attention_model(
                q=normed_input,
                k=normed_input,
                v=normed_input,
                k_padding_mask=source_key_padding_mask,
                attention_mask=attention_mask,
            )
        )
        attention_output = self.__apply_default_norm(
            attention_output, self.self_attention_layer_norm
        )
        attention_output = self.self_attention_dropout(attention_output)
        attention_output = attention_output + residual
        attention_output = self.__apply_post_norm(
            attention_output, self.self_attention_layer_norm
        )
        loss = self.__resolve_auxiliary_loss(residual, auxiliary_loss)
        return attention_output, loss

    def __apply_feed_forward_sublayer(
        self,
        residual: Tensor,
    ) -> tuple[Tensor, Tensor]:
        normed_input = self.__apply_pre_norm(residual, self.feed_forward_layer_norm)
        feed_forward_output, feed_forward_loss = self.feed_forward_model(normed_input)
        feed_forward_output = self.__apply_default_norm(
            feed_forward_output, self.feed_forward_layer_norm
        )
        feed_forward_output = self.feed_forward_dropout(feed_forward_output)
        feed_forward_output = feed_forward_output + residual
        feed_forward_output = self.__apply_post_norm(
            feed_forward_output, self.feed_forward_layer_norm
        )
        loss = self.__resolve_auxiliary_loss(residual, feed_forward_loss)
        return feed_forward_output, loss

    def __apply_pre_norm(self, x: Tensor, layer_norm: nn.LayerNorm) -> Tensor:
        if self.layer_norm_position == LayerNormPositionOptions.BEFORE:
            return layer_norm(x)
        return x

    def __apply_default_norm(self, x: Tensor, layer_norm: nn.LayerNorm) -> Tensor:
        if self.layer_norm_position == LayerNormPositionOptions.DEFAULT:
            return layer_norm(x)
        return x

    def __apply_post_norm(self, x: Tensor, layer_norm: nn.LayerNorm) -> Tensor:
        if self.layer_norm_position == LayerNormPositionOptions.AFTER:
            return layer_norm(x)
        return x

    def __resolve_auxiliary_loss(
        self, reference: Tensor, loss: Tensor | None
    ) -> Tensor:
        if loss is None:
            return reference.new_zeros(())
        return loss


class TransformerDecoderLayer(Module):
    def __init__(
        self,
        cfg: "TransformerDecoderLayerConfig",
        overrides: "TransformerDecoderLayerConfig | None" = None,
    ):
        super().__init__()
        config = getattr(cfg, "transformer_decoder_layer_config", cfg)
        self.cfg: "TransformerDecoderLayerConfig" = self._override_config(
            config, overrides
        )

        self.embedding_dim: int = self.cfg.embedding_dim
        self.layer_norm_position: LayerNormPositionOptions = (
            self.cfg.layer_norm_position
        )
        self.dropout_probability: float = self.cfg.dropout_probability

        TransformerValidator.validate_decoder_layer(self)

        self.self_attention_model = self.cfg.self_attention_config.build()
        self.cross_attention_model = self.__build_cross_attention_model()
        self.feed_forward_model = self.cfg.feed_forward_config.build()

        self.self_attention_layer_norm = nn.LayerNorm(self.embedding_dim)
        self.feed_forward_layer_norm = nn.LayerNorm(self.embedding_dim)
        self.self_attention_dropout = nn.Dropout(self.dropout_probability)
        self.feed_forward_dropout = nn.Dropout(self.dropout_probability)

        if self.cross_attention_model is not None:
            self.cross_attention_layer_norm = nn.LayerNorm(self.embedding_dim)
            self.cross_attention_dropout = nn.Dropout(self.dropout_probability)

    def __build_cross_attention_model(self):
        if self.cfg.cross_attention_config is None:
            return None
        return self.cfg.cross_attention_config.build()

    def forward(
        self,
        target_token_embeddings: Tensor,
        encoder_output: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
        encoder_padding_mask: Tensor | None = None,
        attention_mask: Tensor | None = None,
        encoder_attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        TransformerValidator.validate_decoder_layer_forward_inputs(
            self, target_token_embeddings, encoder_output
        )
        x = target_token_embeddings
        x, self_attention_loss = self.__apply_self_attention_sublayer(
            x,
            target_key_padding_mask=key_padding_mask,
            attention_mask=attention_mask,
        )
        x, cross_attention_loss = self.__apply_cross_attention_sublayer_if_present(
            x,
            encoder_output=encoder_output,
            encoder_padding_mask=encoder_padding_mask,
            encoder_attention_mask=encoder_attention_mask,
        )
        x, feed_forward_loss = self.__apply_feed_forward_sublayer(x)
        total_loss = self_attention_loss + cross_attention_loss + feed_forward_loss
        return x, total_loss

    def __apply_cross_attention_sublayer_if_present(
        self,
        residual: Tensor,
        encoder_output: Tensor | None,
        encoder_padding_mask: Tensor | None,
        encoder_attention_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        if self.cross_attention_model is None:
            return residual, residual.new_zeros(())
        return self.__apply_cross_attention_sublayer(
            residual,
            encoder_output=encoder_output,
            encoder_padding_mask=encoder_padding_mask,
            encoder_attention_mask=encoder_attention_mask,
        )

    def __apply_self_attention_sublayer(
        self,
        residual: Tensor,
        target_key_padding_mask: Tensor | None,
        attention_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        normed_input = self.__apply_pre_norm(
            residual, self.self_attention_layer_norm
        )
        attention_output, _attention_weights, auxiliary_loss = (
            self.self_attention_model(
                q=normed_input,
                k=normed_input,
                v=normed_input,
                k_padding_mask=target_key_padding_mask,
                attention_mask=attention_mask,
            )
        )
        attention_output = self.__apply_default_norm(
            attention_output, self.self_attention_layer_norm
        )
        attention_output = self.self_attention_dropout(attention_output)
        attention_output = attention_output + residual
        attention_output = self.__apply_post_norm(
            attention_output, self.self_attention_layer_norm
        )
        loss = self.__resolve_auxiliary_loss(residual, auxiliary_loss)
        return attention_output, loss

    def __apply_cross_attention_sublayer(
        self,
        residual: Tensor,
        encoder_output: Tensor,
        encoder_padding_mask: Tensor | None,
        encoder_attention_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        normed_input = self.__apply_pre_norm(
            residual, self.cross_attention_layer_norm
        )
        attention_output, _attention_weights, auxiliary_loss = (
            self.cross_attention_model(
                q=normed_input,
                k=encoder_output,
                v=encoder_output,
                k_padding_mask=encoder_padding_mask,
                attention_mask=encoder_attention_mask,
            )
        )
        attention_output = self.__apply_default_norm(
            attention_output, self.cross_attention_layer_norm
        )
        attention_output = self.cross_attention_dropout(attention_output)
        attention_output = attention_output + residual
        attention_output = self.__apply_post_norm(
            attention_output, self.cross_attention_layer_norm
        )
        loss = self.__resolve_auxiliary_loss(residual, auxiliary_loss)
        return attention_output, loss

    def __apply_feed_forward_sublayer(
        self,
        residual: Tensor,
    ) -> tuple[Tensor, Tensor]:
        normed_input = self.__apply_pre_norm(residual, self.feed_forward_layer_norm)
        feed_forward_output, feed_forward_loss = self.feed_forward_model(normed_input)
        feed_forward_output = self.__apply_default_norm(
            feed_forward_output, self.feed_forward_layer_norm
        )
        feed_forward_output = self.feed_forward_dropout(feed_forward_output)
        feed_forward_output = feed_forward_output + residual
        feed_forward_output = self.__apply_post_norm(
            feed_forward_output, self.feed_forward_layer_norm
        )
        loss = self.__resolve_auxiliary_loss(residual, feed_forward_loss)
        return feed_forward_output, loss

    def __apply_pre_norm(self, x: Tensor, layer_norm: nn.LayerNorm) -> Tensor:
        if self.layer_norm_position == LayerNormPositionOptions.BEFORE:
            return layer_norm(x)
        return x

    def __apply_default_norm(self, x: Tensor, layer_norm: nn.LayerNorm) -> Tensor:
        if self.layer_norm_position == LayerNormPositionOptions.DEFAULT:
            return layer_norm(x)
        return x

    def __apply_post_norm(self, x: Tensor, layer_norm: nn.LayerNorm) -> Tensor:
        if self.layer_norm_position == LayerNormPositionOptions.AFTER:
            return layer_norm(x)
        return x

    def __resolve_auxiliary_loss(
        self, reference: Tensor, loss: Tensor | None
    ) -> Tensor:
        if loss is None:
            return reference.new_zeros(())
        return loss

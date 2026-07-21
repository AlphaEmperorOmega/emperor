from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor

from emperor.attention import AttentionLayerState
from emperor.layers import (
    Layer,
    LayerNormPositionOptions,
    LayerState,
    ResidualConfig,
    ResidualConnection,
)
from emperor.nn import Module
from emperor.transformer._state import TransformerDecoderLayerState
from emperor.transformer._validation import TransformerValidator

if TYPE_CHECKING:
    from emperor.transformer._config import (
        TransformerDecoderLayerConfig,
        TransformerEncoderLayerConfig,
    )


class TransformerEncoderLayer(Module):
    VALIDATOR = TransformerValidator

    def __init__(
        self,
        cfg: "TransformerEncoderLayerConfig",
        overrides: "TransformerEncoderLayerConfig | None" = None,
    ):
        super().__init__()
        config = getattr(cfg, "transformer_encoder_layer_config", cfg)
        self.cfg: TransformerEncoderLayerConfig = self._override_config(
            config, overrides
        )

        self.embedding_dim: int = self.cfg.embedding_dim
        self.layer_norm_position: LayerNormPositionOptions = (
            self.cfg.layer_norm_position
        )
        self.dropout_probability: float = self.cfg.dropout_probability
        self.residual_config: ResidualConfig | None = self.cfg.residual_config

        self.VALIDATOR.validate_encoder_layer(self)

        self.self_attention_model = self.cfg.attention_config.build()
        self.feed_forward_model = self.cfg.feed_forward_config.build()
        self.self_attention_residual_connection = self.__build_residual_connection()
        self.feed_forward_residual_connection = self.__build_residual_connection()

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
        self.VALIDATOR.validate_encoder_layer_forward_inputs(
            self, source_token_embeddings
        )
        x = source_token_embeddings
        attention_mask = self.__resolve_attention_mask_for_padding_mask(
            source_token_embeddings,
            source_key_padding_mask,
            attention_mask,
        )
        x, attention_loss = self.__apply_self_attention_sublayer(
            x,
            source_key_padding_mask=source_key_padding_mask,
            attention_mask=attention_mask,
        )
        x, feed_forward_loss = self.__apply_feed_forward_sublayer(x)
        total_loss = attention_loss + feed_forward_loss
        return x, total_loss

    def __resolve_attention_mask_for_padding_mask(
        self,
        source_token_embeddings: Tensor,
        source_key_padding_mask: Tensor | None,
        attention_mask: Tensor | None,
    ) -> Tensor | None:
        if attention_mask is not None:
            return attention_mask
        if source_key_padding_mask is None:
            return None
        if not self.cfg.causal_attention_mask_flag:
            return None
        sequence_length = source_token_embeddings.size(1)
        return torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                dtype=torch.bool,
                device=source_token_embeddings.device,
            ),
            diagonal=1,
        )

    def __apply_self_attention_sublayer(
        self,
        residual: Tensor,
        source_key_padding_mask: Tensor | None,
        attention_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        normed_input = self.__apply_pre_norm(residual, self.self_attention_layer_norm)
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
        attention_output = self.__maybe_apply_residual_connection(
            self.self_attention_residual_connection,
            attention_output,
            residual,
        )
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
        feed_forward_output = self.__maybe_apply_residual_connection(
            self.feed_forward_residual_connection,
            feed_forward_output,
            residual,
        )
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

    def __build_residual_connection(self) -> ResidualConnection | None:
        if self.residual_config is None:
            return None
        return self._build_from_config(
            self.residual_config,
            residual_dim=self.embedding_dim,
        )

    def __maybe_apply_residual_connection(
        self,
        connection: ResidualConnection | None,
        current: Tensor,
        residual: Tensor,
    ) -> Tensor:
        if connection is None:
            return current
        return connection(current, residual)

    def __resolve_auxiliary_loss(
        self, reference: Tensor, loss: Tensor | None
    ) -> Tensor:
        if loss is None:
            return reference.new_zeros(())
        return loss


class TransformerEncoderBlockLayer(Layer):
    def _handle_model_processing(
        self,
        main_model_input: Tensor,
        state: "LayerState",
    ) -> Tensor:
        source_key_padding_mask = None
        attention_mask = None
        if isinstance(state, AttentionLayerState):
            source_key_padding_mask = state.key_padding_mask
            attention_mask = state.attention_mask
        output, loss = self.model(
            main_model_input,
            source_key_padding_mask=source_key_padding_mask,
            attention_mask=attention_mask,
        )
        state.loss = loss if state.loss is None else state.loss + loss
        return output


class TransformerDecoderBlockLayer(Layer):
    def _handle_model_processing(
        self,
        main_model_input: Tensor,
        state: "LayerState",
    ) -> Tensor:
        if not isinstance(state, TransformerDecoderLayerState):
            raise TypeError(
                "TransformerDecoderBlockLayer requires a TransformerDecoderLayerState."
            )
        output, loss = self.model(
            main_model_input,
            encoder_output=state.encoder_output,
            key_padding_mask=state.target_key_padding_mask,
            encoder_padding_mask=state.encoder_padding_mask,
            attention_mask=state.target_attention_mask,
            encoder_attention_mask=state.cross_attention_mask,
        )
        state.loss = loss if state.loss is None else state.loss + loss
        return output


class TransformerDecoderLayer(Module):
    VALIDATOR = TransformerValidator

    def __init__(
        self,
        cfg: "TransformerDecoderLayerConfig",
        overrides: "TransformerDecoderLayerConfig | None" = None,
    ):
        super().__init__()
        config = getattr(cfg, "transformer_decoder_layer_config", cfg)
        self.cfg: TransformerDecoderLayerConfig = self._override_config(
            config, overrides
        )

        self.embedding_dim: int = self.cfg.embedding_dim
        self.layer_norm_position: LayerNormPositionOptions = (
            self.cfg.layer_norm_position
        )
        self.dropout_probability: float = self.cfg.dropout_probability
        self.residual_config: ResidualConfig | None = self.cfg.residual_config

        self.VALIDATOR.validate_decoder_layer(self)

        self.self_attention_model = self.cfg.self_attention_config.build()
        self.cross_attention_model = self.__build_cross_attention_model()
        self.feed_forward_model = self.cfg.feed_forward_config.build()
        self.self_attention_residual_connection = self.__build_residual_connection()
        self.cross_attention_residual_connection = (
            self.__build_residual_connection()
            if self.cross_attention_model is not None
            else None
        )
        self.feed_forward_residual_connection = self.__build_residual_connection()

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
        self.VALIDATOR.validate_decoder_layer_forward_inputs(
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
        normed_input = self.__apply_pre_norm(residual, self.self_attention_layer_norm)
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
        attention_output = self.__maybe_apply_residual_connection(
            self.self_attention_residual_connection,
            attention_output,
            residual,
        )
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
        normed_input = self.__apply_pre_norm(residual, self.cross_attention_layer_norm)
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
        attention_output = self.__maybe_apply_residual_connection(
            self.cross_attention_residual_connection,
            attention_output,
            residual,
        )
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
        feed_forward_output = self.__maybe_apply_residual_connection(
            self.feed_forward_residual_connection,
            feed_forward_output,
            residual,
        )
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

    def __build_residual_connection(self) -> ResidualConnection | None:
        if self.residual_config is None:
            return None
        return self._build_from_config(
            self.residual_config,
            residual_dim=self.embedding_dim,
        )

    def __maybe_apply_residual_connection(
        self,
        connection: ResidualConnection | None,
        current: Tensor,
        residual: Tensor,
    ) -> Tensor:
        if connection is None:
            return current
        return connection(current, residual)

    def __resolve_auxiliary_loss(
        self, reference: Tensor, loss: Tensor | None
    ) -> Tensor:
        if loss is None:
            return reference.new_zeros(())
        return loss

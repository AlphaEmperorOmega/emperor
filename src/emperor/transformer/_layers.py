from typing import TYPE_CHECKING

from torch import Tensor

from emperor.attention import AttentionLayerState
from emperor.layers import ActivationOptions, Layer, LayerConfig, LayerState
from emperor.nn import Module
from emperor.transformer._state import TransformerDecoderLayerState
from emperor.transformer._validation import TransformerValidator

if TYPE_CHECKING:
    from emperor.config import ConfigBase
    from emperor.transformer._config import (
        TransformerDecoderLayerConfig,
        TransformerEncoderLayerConfig,
    )


class _TransformerSubLayer(Layer):
    def _accumulate_model_loss(
        self,
        state: LayerState,
        loss: Tensor | None,
    ) -> None:
        if loss is None:
            return
        state.loss = self._accumulate_auxiliary_loss(
            state.loss,
            self._reduce_auxiliary_loss(loss),
        )


class _EncoderSelfAttentionLayer(_TransformerSubLayer):
    def _handle_model_processing(
        self,
        main_model_input: Tensor,
        state: LayerState,
    ) -> Tensor:
        if not isinstance(state, AttentionLayerState):
            raise TypeError("Encoder self-attention requires an AttentionLayerState.")
        output, _attention_weights, loss = self.model(
            q=main_model_input,
            k=main_model_input,
            v=main_model_input,
            k_padding_mask=state.key_padding_mask,
            attention_mask=state.attention_mask,
        )
        self._accumulate_model_loss(state, loss)
        return output


class _DecoderSelfAttentionLayer(_TransformerSubLayer):
    def _handle_model_processing(
        self,
        main_model_input: Tensor,
        state: LayerState,
    ) -> Tensor:
        if not isinstance(state, TransformerDecoderLayerState):
            raise TypeError(
                "Decoder self-attention requires a TransformerDecoderLayerState."
            )
        output, _attention_weights, loss = self.model(
            q=main_model_input,
            k=main_model_input,
            v=main_model_input,
            k_padding_mask=state.target_key_padding_mask,
            attention_mask=state.target_attention_mask,
        )
        self._accumulate_model_loss(state, loss)
        return output


class _DecoderCrossAttentionLayer(_TransformerSubLayer):
    def _handle_model_processing(
        self,
        main_model_input: Tensor,
        state: LayerState,
    ) -> Tensor:
        if not isinstance(state, TransformerDecoderLayerState):
            raise TypeError(
                "Decoder cross-attention requires a TransformerDecoderLayerState."
            )
        output, _attention_weights, loss = self.model(
            q=main_model_input,
            k=state.encoder_output,
            v=state.encoder_output,
            k_padding_mask=state.encoder_padding_mask,
            attention_mask=state.cross_attention_mask,
        )
        self._accumulate_model_loss(state, loss)
        return output


class _FeedForwardLayer(_TransformerSubLayer):
    def _handle_model_processing(
        self,
        main_model_input: Tensor,
        state: LayerState,
    ) -> Tensor:
        output, loss = self.model(main_model_input)
        self._accumulate_model_loss(state, loss)
        return output


def _sub_layer_config(
    *,
    embedding_dim: int,
    owner_config: "TransformerEncoderLayerConfig | TransformerDecoderLayerConfig",
    model_config: "ConfigBase",
) -> LayerConfig:
    return LayerConfig(
        input_dim=embedding_dim,
        output_dim=embedding_dim,
        activation=ActivationOptions.DISABLED,
        residual_config=owner_config.residual_config,
        dropout_probability=owner_config.dropout_probability,
        layer_norm_position=owner_config.layer_norm_position,
        gate_config=None,
        halting_config=None,
        memory_config=None,
        layer_model_config=model_config,
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
        self.layer_norm_position = self.cfg.layer_norm_position
        self.dropout_probability: float = self.cfg.dropout_probability
        self.residual_config = self.cfg.residual_config

        self.VALIDATOR.validate_encoder_layer(self)

        self.self_attention_layer = _EncoderSelfAttentionLayer(
            _sub_layer_config(
                embedding_dim=self.embedding_dim,
                owner_config=self.cfg,
                model_config=self.cfg.attention_config,
            )
        )
        self.feed_forward_layer = _FeedForwardLayer(
            _sub_layer_config(
                embedding_dim=self.embedding_dim,
                owner_config=self.cfg,
                model_config=self.cfg.feed_forward_config,
            )
        )

    @property
    def self_attention_model(self):
        return self.self_attention_layer.model

    @property
    def feed_forward_model(self):
        return self.feed_forward_layer.model

    def forward(
        self,
        source_token_embeddings: Tensor,
        source_key_padding_mask: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        self.VALIDATOR.validate_encoder_layer_forward_inputs(
            self, source_token_embeddings
        )
        state = AttentionLayerState(
            hidden=source_token_embeddings,
            key_padding_mask=source_key_padding_mask,
            attention_mask=attention_mask,
        )
        state = self.self_attention_layer(state)
        state = self.feed_forward_layer(state)
        loss = (
            state.loss
            if state.loss is not None
            else source_token_embeddings.new_zeros(())
        )
        return state.hidden, loss


class TransformerEncoderBlockLayer(Layer):
    def _handle_model_processing(
        self,
        main_model_input: Tensor,
        state: LayerState,
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
        state.loss = self._accumulate_auxiliary_loss(
            state.loss,
            self._reduce_auxiliary_loss(loss),
        )
        return output


class TransformerDecoderBlockLayer(Layer):
    def _handle_model_processing(
        self,
        main_model_input: Tensor,
        state: LayerState,
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
        state.loss = self._accumulate_auxiliary_loss(
            state.loss,
            self._reduce_auxiliary_loss(loss),
        )
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
        self.layer_norm_position = self.cfg.layer_norm_position
        self.dropout_probability: float = self.cfg.dropout_probability
        self.residual_config = self.cfg.residual_config

        self.VALIDATOR.validate_decoder_layer(self)

        self.self_attention_layer = _DecoderSelfAttentionLayer(
            _sub_layer_config(
                embedding_dim=self.embedding_dim,
                owner_config=self.cfg,
                model_config=self.cfg.self_attention_config,
            )
        )
        self.cross_attention_layer = self.__build_cross_attention_layer()
        self.feed_forward_layer = _FeedForwardLayer(
            _sub_layer_config(
                embedding_dim=self.embedding_dim,
                owner_config=self.cfg,
                model_config=self.cfg.feed_forward_config,
            )
        )

    def __build_cross_attention_layer(self):
        if self.cfg.cross_attention_config is None:
            return None
        return _DecoderCrossAttentionLayer(
            _sub_layer_config(
                embedding_dim=self.embedding_dim,
                owner_config=self.cfg,
                model_config=self.cfg.cross_attention_config,
            )
        )

    @property
    def self_attention_model(self):
        return self.self_attention_layer.model

    @property
    def cross_attention_model(self):
        if self.cross_attention_layer is None:
            return None
        return self.cross_attention_layer.model

    @property
    def feed_forward_model(self):
        return self.feed_forward_layer.model

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
        state = TransformerDecoderLayerState(
            hidden=target_token_embeddings,
            target_key_padding_mask=key_padding_mask,
            target_attention_mask=attention_mask,
            encoder_output=encoder_output,
            encoder_padding_mask=encoder_padding_mask,
            cross_attention_mask=encoder_attention_mask,
        )
        state = self.self_attention_layer(state)
        if self.cross_attention_layer is not None:
            state = self.cross_attention_layer(state)
        state = self.feed_forward_layer(state)
        loss = (
            state.loss
            if state.loss is not None
            else target_token_embeddings.new_zeros(())
        )
        return state.hidden, loss

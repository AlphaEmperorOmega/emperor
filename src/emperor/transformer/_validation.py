from typing import TYPE_CHECKING

from emperor._validation import ValidatorBase
from emperor.config import ConfigBase

if TYPE_CHECKING:
    from torch import Tensor

    from emperor.transformer._feed_forward import FeedForward
    from emperor.transformer._layers import (
        TransformerDecoderLayer,
        TransformerEncoderLayer,
    )
    from emperor.transformer._model import Transformer


class TransformerValidator(ValidatorBase):
    OPTIONAL_FIELDS = {
        "override_config",
        "cross_attention_config",
        "residual_config",
    }

    # --- build-time structural validation ---

    @classmethod
    def validate_transformer(cls, model: "Transformer") -> None:
        encoder_stack_config = model.cfg.encoder_stack_config
        decoder_stack_config = model.cfg.decoder_stack_config
        if encoder_stack_config is None and decoder_stack_config is None:
            raise ValueError(
                "TransformerConfig requires at least one of "
                "encoder_stack_config or decoder_stack_config to be set; "
                "both are None."
            )
        cls._validate_stack_config_types(
            encoder_stack_config,
            decoder_stack_config,
        )
        cls._validate_decoder_cross_attention_has_encoder(
            encoder_stack_config, decoder_stack_config
        )

    @staticmethod
    def _validate_stack_config_types(*stack_configs) -> None:
        from emperor.layers import LayerStackConfig, RecurrentLayerConfig

        for stack_config in stack_configs:
            if stack_config is None:
                continue
            if not isinstance(stack_config, (LayerStackConfig, RecurrentLayerConfig)):
                raise TypeError(
                    "Transformer stack configurations must be LayerStackConfig or "
                    "RecurrentLayerConfig, got "
                    f"{type(stack_config).__name__}."
                )

    @classmethod
    def _validate_decoder_cross_attention_has_encoder(
        cls,
        encoder_stack_config,
        decoder_stack_config,
    ) -> None:
        if encoder_stack_config is not None or decoder_stack_config is None:
            return
        decoder_layer_config = cls._find_decoder_layer_config(decoder_stack_config)
        if (
            decoder_layer_config is not None
            and decoder_layer_config.cross_attention_config is not None
        ):
            raise ValueError(
                "A decoder-only Transformer (no encoder_stack_config) must "
                "configure the decoder layer with cross_attention_config=None; "
                "cross-attention requires encoder output."
            )

    @classmethod
    def _find_decoder_layer_config(cls, config):
        from emperor.transformer._config import TransformerDecoderLayerConfig

        if isinstance(config, TransformerDecoderLayerConfig):
            return config
        for field_name in ("block_config", "layer_config", "layer_model_config"):
            nested_config = getattr(config, field_name, None)
            if nested_config is None:
                continue
            match = cls._find_decoder_layer_config(nested_config)
            if match is not None:
                return match
        return None

    @classmethod
    def validate_encoder_layer(cls, model: "TransformerEncoderLayer") -> None:
        cls.validate_required_fields(model.cfg)
        cls.validate_field_types(model.cfg)
        cls.validate_dimensions(embedding_dim=model.embedding_dim)
        cls._validate_layer_norm_position(model.layer_norm_position)
        cls._validate_encoder_attention_config(model.cfg.attention_config)
        cls._validate_residual_history_bridge(
            model.cfg.residual_config,
            owner_name="TransformerEncoderLayerConfig",
        )

    @classmethod
    def validate_decoder_layer(cls, model: "TransformerDecoderLayer") -> None:
        cls.validate_required_fields(model.cfg)
        cls.validate_field_types(model.cfg)
        cls.validate_dimensions(embedding_dim=model.embedding_dim)
        cls._validate_layer_norm_position(model.layer_norm_position)
        cls._validate_decoder_self_attention_config(model.cfg.self_attention_config)
        cls._validate_decoder_cross_attention_config(model.cfg.cross_attention_config)
        cls._validate_residual_history_bridge(
            model.cfg.residual_config,
            owner_name="TransformerDecoderLayerConfig",
        )

    @staticmethod
    def _validate_encoder_attention_config(attention_config) -> None:
        from emperor.attention import (
            MixerAttentionConfig,
            MixtureOfAttentionHeadsConfig,
            SelfAttentionConfig,
        )

        if not isinstance(
            attention_config,
            (
                SelfAttentionConfig,
                MixtureOfAttentionHeadsConfig,
                MixerAttentionConfig,
            ),
        ):
            raise TypeError(
                "attention_config must be a SelfAttentionConfig, "
                "MixtureOfAttentionHeadsConfig, or MixerAttentionConfig, got "
                f"{type(attention_config).__name__}."
            )

    @staticmethod
    def _validate_decoder_self_attention_config(attention_config) -> None:
        from emperor.attention import (
            MixerAttentionConfig,
            MixtureOfAttentionHeadsConfig,
            SelfAttentionConfig,
        )

        if not isinstance(
            attention_config,
            (
                SelfAttentionConfig,
                MixtureOfAttentionHeadsConfig,
                MixerAttentionConfig,
            ),
        ):
            raise TypeError(
                "self_attention_config must be a SelfAttentionConfig, "
                "MixtureOfAttentionHeadsConfig, or MixerAttentionConfig, got "
                f"{type(attention_config).__name__}."
            )

    @staticmethod
    def _validate_decoder_cross_attention_config(attention_config) -> None:
        from emperor.attention import (
            IndependentAttentionConfig,
            MixtureOfAttentionHeadsConfig,
        )

        if attention_config is None:
            return
        if not isinstance(
            attention_config,
            (IndependentAttentionConfig, MixtureOfAttentionHeadsConfig),
        ):
            raise TypeError(
                "cross_attention_config must be an IndependentAttentionConfig, "
                "MixtureOfAttentionHeadsConfig, or None; MixerAttentionConfig "
                "is self-processing only. Got "
                f"{type(attention_config).__name__}."
            )

    @staticmethod
    def _validate_layer_norm_position(layer_norm_position) -> None:
        from emperor.layers import LayerNormPositionOptions

        if not isinstance(layer_norm_position, LayerNormPositionOptions):
            raise TypeError(
                "layer_norm_position must be a LayerNormPositionOptions value, "
                f"got {type(layer_norm_position).__name__}"
            )

    @staticmethod
    def _validate_residual_history_bridge(residual_config, *, owner_name: str) -> None:
        from emperor.layers import ResidualConfig, ResidualConnectionOptions

        if not isinstance(residual_config, ResidualConfig):
            return
        if residual_config.option != ResidualConnectionOptions.ATTENTION_RESIDUAL:
            return
        raise ValueError(
            f"ATTENTION_RESIDUAL is not supported for {owner_name} until "
            "Transformer sublayers share an explicit forward-local history bridge."
        )

    # --- forward-boundary validation ---

    @classmethod
    def validate_encoder_layer_forward_inputs(
        cls,
        model: "TransformerEncoderLayer",
        source_token_embeddings: "Tensor",
    ) -> None:
        cls._validate_last_dim(
            source_token_embeddings, model.embedding_dim, "source_token_embeddings"
        )

    @classmethod
    def validate_decoder_layer_forward_inputs(
        cls,
        model: "TransformerDecoderLayer",
        target_token_embeddings: "Tensor",
        encoder_output: "Tensor | None",
    ) -> None:
        cls._validate_last_dim(
            target_token_embeddings, model.embedding_dim, "target_token_embeddings"
        )
        if model.cross_attention_model is None:
            return
        if encoder_output is None:
            raise ValueError(
                "TransformerDecoderLayer with cross-attention requires "
                "encoder_output, received None."
            )
        cls._validate_last_dim(encoder_output, model.embedding_dim, "encoder_output")

    @staticmethod
    def validate_transformer_forward_inputs(
        model: "Transformer",
        source_token_embeddings: "Tensor | None",
        target_token_embeddings: "Tensor | None",
    ) -> None:
        if model.encoder_model is not None and source_token_embeddings is None:
            raise ValueError(
                "Transformer with an encoder requires source_token_embeddings, "
                "received None."
            )
        if model.decoder_model is not None and target_token_embeddings is None:
            raise ValueError(
                "Transformer with a decoder requires target_token_embeddings, "
                "received None."
            )

    @staticmethod
    def _validate_last_dim(
        tensor: "Tensor | None",
        expected_dim: int,
        name: str,
    ) -> None:
        if tensor is None:
            raise ValueError(f"{name} is required, received None.")
        if tensor.size(-1) != expected_dim:
            raise ValueError(
                f"{name} last dimension must be {expected_dim}, received "
                f"{tensor.size(-1)}."
            )


class FeedForwardValidator(ValidatorBase):
    @classmethod
    def validate(cls, model: "FeedForward") -> None:
        cls.validate_required_fields(model.cfg)
        cls.validate_dimensions(input_dim=model.input_dim, output_dim=model.output_dim)
        cls._validate_stack_config_type(model.stack_config)

    @staticmethod
    def _validate_stack_config_type(stack_config: ConfigBase) -> None:
        from emperor.experts import MixtureOfExpertsModelConfig
        from emperor.layers import LayerStackConfig, RecurrentLayerConfig

        if not isinstance(
            stack_config,
            (LayerStackConfig, MixtureOfExpertsModelConfig, RecurrentLayerConfig),
        ):
            raise TypeError(
                "FeedForward.stack_config must be a LayerStackConfig, "
                "MixtureOfExpertsModelConfig, or RecurrentLayerConfig, got "
                f"{type(stack_config).__name__}"
            )

from typing import TYPE_CHECKING

from emperor._validation import ValidatorBase
from emperor.config import ConfigBase
from emperor.layers._validation import ResidualConnectionValidator

if TYPE_CHECKING:
    from torch import Tensor

    from emperor.transformer._feed_forward import FeedForward
    from emperor.transformer._layers import (
        TransformerDecoderLayer,
        TransformerEncoderLayer,
    )
    from emperor.transformer._model import Transformer
    from emperor.transformer._stacks import (
        TransformerDecoderStack,
        TransformerEncoderStack,
    )


class TransformerValidator(ValidatorBase):
    RESIDUAL_VALIDATOR = ResidualConnectionValidator

    OPTIONAL_FIELDS = {
        "override_config",
        "cross_attention_config",
        "causal_attention_mask_flag",
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
        cls._validate_decoder_cross_attention_has_encoder(
            encoder_stack_config, decoder_stack_config
        )

    @staticmethod
    def _validate_decoder_cross_attention_has_encoder(
        encoder_stack_config,
        decoder_stack_config,
    ) -> None:
        if encoder_stack_config is not None or decoder_stack_config is None:
            return
        layer_config = decoder_stack_config.layer_config
        if layer_config is not None and layer_config.cross_attention_config is not None:
            raise ValueError(
                "A decoder-only Transformer (no encoder_stack_config) must "
                "configure the decoder layer with cross_attention_config=None; "
                "cross-attention requires encoder output."
            )

    @classmethod
    def validate_encoder_layer(cls, model: "TransformerEncoderLayer") -> None:
        cls.validate_required_fields(model.cfg)
        cls.validate_field_types(model.cfg)
        cls.validate_dimensions(embedding_dim=model.embedding_dim)
        cls._validate_layer_norm_position(model.layer_norm_position)
        cls._validate_residual_config(
            model.residual_config,
            owner_name="TransformerEncoderLayerConfig",
        )

    @classmethod
    def validate_decoder_layer(cls, model: "TransformerDecoderLayer") -> None:
        cls.validate_required_fields(model.cfg)
        cls.validate_field_types(model.cfg)
        cls.validate_dimensions(embedding_dim=model.embedding_dim)
        cls._validate_layer_norm_position(model.layer_norm_position)
        cls._validate_residual_config(
            model.residual_config,
            owner_name="TransformerDecoderLayerConfig",
        )

    @staticmethod
    def _validate_layer_norm_position(layer_norm_position) -> None:
        from emperor.layers import LayerNormPositionOptions

        if not isinstance(layer_norm_position, LayerNormPositionOptions):
            raise TypeError(
                "layer_norm_position must be a LayerNormPositionOptions value, "
                f"got {type(layer_norm_position).__name__}"
            )

    @classmethod
    def _validate_residual_config(cls, residual_config, owner_name: str) -> None:
        cls.RESIDUAL_VALIDATOR.validate_residual_config(
            residual_config,
            owner_name=owner_name,
        )

    @classmethod
    def validate_encoder_stack(cls, model: "TransformerEncoderStack") -> None:
        cls.validate_required_fields(model.cfg)
        cls.validate_dimensions(
            num_layers=model.num_layers,
            embedding_dim=model.embedding_dim,
            source_sequence_length=model.source_sequence_length,
            target_sequence_length=model.target_sequence_length,
        )
        if model.source_sequence_length != model.target_sequence_length:
            raise ValueError(
                "TransformerEncoderStack requires source_sequence_length == "
                f"target_sequence_length, got {model.source_sequence_length} "
                f"and {model.target_sequence_length}"
            )

    @classmethod
    def validate_decoder_stack(cls, model: "TransformerDecoderStack") -> None:
        cls.validate_required_fields(model.cfg)
        cls.validate_dimensions(
            num_layers=model.num_layers,
            embedding_dim=model.embedding_dim,
            source_sequence_length=model.source_sequence_length,
            target_sequence_length=model.target_sequence_length,
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

    @classmethod
    def validate_encoder_stack_forward_inputs(
        cls,
        model: "TransformerEncoderStack",
        source_token_embeddings: "Tensor",
    ) -> None:
        cls._validate_last_dim(
            source_token_embeddings, model.embedding_dim, "source_token_embeddings"
        )

    @classmethod
    def validate_decoder_stack_forward_inputs(
        cls,
        model: "TransformerDecoderStack",
        target_token_embeddings: "Tensor",
        encoder_output: "Tensor | None",
    ) -> None:
        cls._validate_last_dim(
            target_token_embeddings, model.embedding_dim, "target_token_embeddings"
        )

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

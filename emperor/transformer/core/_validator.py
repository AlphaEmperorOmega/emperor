from typing import TYPE_CHECKING

from emperor.base.validator import ValidatorBase

if TYPE_CHECKING:
    from torch import Tensor

    from emperor.transformer.core.layers import (
        TransformerDecoderLayer,
        TransformerEncoderLayer,
    )
    from emperor.transformer.core.stack import (
        TransformerDecoderStack,
        TransformerEncoderStack,
    )
    from emperor.transformer.model import Transformer


class TransformerValidator(ValidatorBase):
    OPTIONAL_FIELDS = {
        "override_config",
        "cross_attention_config",
        "causal_attention_mask_flag",
    }

    # --- build-time structural validation ---

    @staticmethod
    def validate_transformer(model: "Transformer") -> None:
        encoder_stack_config = model.cfg.encoder_stack_config
        decoder_stack_config = model.cfg.decoder_stack_config
        if encoder_stack_config is None and decoder_stack_config is None:
            raise ValueError(
                "TransformerConfig requires at least one of "
                "encoder_stack_config or decoder_stack_config to be set; "
                "both are None."
            )
        TransformerValidator.__validate_decoder_cross_attention_has_encoder(
            encoder_stack_config, decoder_stack_config
        )

    @staticmethod
    def __validate_decoder_cross_attention_has_encoder(
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

    @staticmethod
    def validate_encoder_layer(model: "TransformerEncoderLayer") -> None:
        TransformerValidator.validate_required_fields(model.cfg)
        TransformerValidator.validate_field_types(model.cfg)
        TransformerValidator.validate_dimensions(embedding_dim=model.embedding_dim)

    @staticmethod
    def validate_decoder_layer(model: "TransformerDecoderLayer") -> None:
        TransformerValidator.validate_required_fields(model.cfg)
        TransformerValidator.validate_field_types(model.cfg)
        TransformerValidator.validate_dimensions(embedding_dim=model.embedding_dim)

    @staticmethod
    def validate_encoder_stack(model: "TransformerEncoderStack") -> None:
        TransformerValidator.validate_required_fields(model.cfg)
        TransformerValidator.validate_dimensions(
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

    @staticmethod
    def validate_decoder_stack(model: "TransformerDecoderStack") -> None:
        TransformerValidator.validate_required_fields(model.cfg)
        TransformerValidator.validate_dimensions(
            num_layers=model.num_layers,
            embedding_dim=model.embedding_dim,
            source_sequence_length=model.source_sequence_length,
            target_sequence_length=model.target_sequence_length,
        )

    # --- forward-boundary validation ---

    @staticmethod
    def validate_encoder_layer_forward_inputs(
        model: "TransformerEncoderLayer",
        source_token_embeddings: "Tensor",
    ) -> None:
        TransformerValidator.__validate_last_dim(
            source_token_embeddings, model.embedding_dim, "source_token_embeddings"
        )

    @staticmethod
    def validate_decoder_layer_forward_inputs(
        model: "TransformerDecoderLayer",
        target_token_embeddings: "Tensor",
        encoder_output: "Tensor | None",
    ) -> None:
        TransformerValidator.__validate_last_dim(
            target_token_embeddings, model.embedding_dim, "target_token_embeddings"
        )
        if model.cross_attention_model is None:
            return
        if encoder_output is None:
            raise ValueError(
                "TransformerDecoderLayer with cross-attention requires "
                "encoder_output, received None."
            )
        TransformerValidator.__validate_last_dim(
            encoder_output, model.embedding_dim, "encoder_output"
        )

    @staticmethod
    def validate_encoder_stack_forward_inputs(
        model: "TransformerEncoderStack",
        source_token_embeddings: "Tensor",
    ) -> None:
        TransformerValidator.__validate_last_dim(
            source_token_embeddings, model.embedding_dim, "source_token_embeddings"
        )

    @staticmethod
    def validate_decoder_stack_forward_inputs(
        model: "TransformerDecoderStack",
        target_token_embeddings: "Tensor",
        encoder_output: "Tensor | None",
    ) -> None:
        TransformerValidator.__validate_last_dim(
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
    def __validate_last_dim(
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

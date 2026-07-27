from dataclasses import fields
from typing import TYPE_CHECKING

from emperor._validation import ValidatorBase
from emperor.config import ConfigBase

if TYPE_CHECKING:
    from torch import Tensor

    from emperor.layers import RowLayout
    from emperor.transformer._feed_forward import FeedForward
    from emperor.transformer._layers import (
        TransformerDecoderLayer,
        TransformerEncoderLayer,
    )
    from emperor.transformer._model import Transformer


def _find_enabled_adaptive_grouping_path(
    value: object,
    path: str,
    visited: set[int] | None = None,
    *,
    grouping_scope: object | None = None,
) -> str | None:
    """Find the first grouped adaptive leaf in a Transformer-owned config tree."""

    if visited is None:
        visited = set()

    if isinstance(value, ConfigBase):
        identity = id(value)
        if identity in visited:
            return None
        visited.add(identity)

        try:
            config_validator = value.registry_owner().VALIDATOR
        except (AttributeError, NotImplementedError):
            config_validator = None
        grouping_is_enabled = getattr(
            config_validator,
            "grouping_is_enabled",
            None,
        )
        if (
            callable(grouping_is_enabled)
            and grouping_is_enabled(value)
            and (
                grouping_scope is None
                or getattr(value, "grouping_scope", None) is grouping_scope
            )
        ):
            return path

        for config_field in fields(value):
            match = _find_enabled_adaptive_grouping_path(
                getattr(value, config_field.name),
                f"{path}.{config_field.name}",
                visited,
                grouping_scope=grouping_scope,
            )
            if match is not None:
                return match
        return None

    if isinstance(value, dict):
        identity = id(value)
        if identity in visited:
            return None
        visited.add(identity)
        for key, item in value.items():
            match = _find_enabled_adaptive_grouping_path(
                item,
                f"{path}[{key!r}]",
                visited,
                grouping_scope=grouping_scope,
            )
            if match is not None:
                return match
        return None

    if isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in visited:
            return None
        visited.add(identity)
        for index, item in enumerate(value):
            match = _find_enabled_adaptive_grouping_path(
                item,
                f"{path}[{index}]",
                visited,
                grouping_scope=grouping_scope,
            )
            if match is not None:
                return match
    return None


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
        cls._validate_outer_grouped_gates(
            encoder_stack_config,
            "TransformerConfig.encoder_stack_config",
        )
        cls._validate_outer_grouped_gates(
            decoder_stack_config,
            "TransformerConfig.decoder_stack_config",
        )
        cls._validate_decoder_cross_attention_has_encoder(
            encoder_stack_config, decoder_stack_config
        )

    @classmethod
    def _validate_outer_grouped_gates(cls, config, path: str) -> None:
        if config is None:
            return
        from emperor.layers import (
            LayerConfig,
            LayerStackConfig,
            RecurrentCompositionConfig,
        )
        from emperor.transformer._config import (
            TransformerDecoderLayerConfig,
            TransformerEncoderLayerConfig,
        )

        gate_configs: tuple[tuple[str, object | None], ...]
        nested_configs: tuple[tuple[str, object], ...] = ()
        if isinstance(config, RecurrentCompositionConfig):
            gate_configs = (
                (f"{path}.gate_config", getattr(config, "gate_config", None)),
            )
            nested_configs = tuple(
                (f"{path}.{field_name}", transition_config)
                for field_name, transition_config in config._transition_config_items()
            )
        elif isinstance(config, LayerStackConfig):
            layer_config = config.layer_config
            gate_configs = ((f"{path}.shared_gate_config", config.shared_gate_config),)
            if isinstance(layer_config, LayerConfig):
                gate_configs += (
                    (f"{path}.layer_config.gate_config", layer_config.gate_config),
                )
                if layer_config.layer_model_config is not None:
                    nested_configs = (
                        (
                            f"{path}.layer_config.layer_model_config",
                            layer_config.layer_model_config,
                        ),
                    )
        elif isinstance(config, LayerConfig):
            gate_configs = ((f"{path}.gate_config", config.gate_config),)
            if config.layer_model_config is not None:
                nested_configs = (
                    (f"{path}.layer_model_config", config.layer_model_config),
                )
        else:
            return

        for gate_path, gate_config in gate_configs:
            if gate_config is None:
                continue
            grouping_path = _find_enabled_adaptive_grouping_path(
                gate_config,
                gate_path,
            )
            if grouping_path is not None:
                raise ValueError(
                    "Enabled adaptive parameter grouping is not supported in an "
                    "outer Transformer gate because that gate receives rank-three "
                    f"token tensors. Found grouping at {grouping_path}."
                )

        for nested_path, nested_config in nested_configs:
            if isinstance(
                nested_config,
                (TransformerEncoderLayerConfig, TransformerDecoderLayerConfig),
            ):
                continue
            cls._validate_outer_grouped_gates(nested_config, nested_path)

    @staticmethod
    def _validate_stack_config_types(*stack_configs) -> None:
        from emperor.layers import LayerStackConfig, RecurrentCompositionConfig

        for stack_config in stack_configs:
            if stack_config is None:
                continue
            if not isinstance(
                stack_config,
                (LayerStackConfig, RecurrentCompositionConfig),
            ):
                raise TypeError(
                    "Transformer stack configurations must be LayerStackConfig or "
                    "RecurrentCompositionConfig, got "
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
        cls._validate_feed_forward_grouping(
            model.cfg.feed_forward_config,
            owner_name="TransformerEncoderLayerConfig",
        )
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
        cls._validate_feed_forward_grouping(
            model.cfg.feed_forward_config,
            owner_name="TransformerDecoderLayerConfig",
        )
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
        from emperor.layers import AttentionResidualConfig

        if not isinstance(residual_config, AttentionResidualConfig):
            return
        raise ValueError(
            f"AttentionResidualConfig is not supported for {owner_name} until "
            "Transformer sublayers share an explicit forward-local history bridge."
        )

    @staticmethod
    def _validate_feed_forward_grouping(
        feed_forward_config,
        *,
        owner_name: str,
    ) -> None:
        from emperor.augmentations.adaptive_parameters import (
            AdaptiveParameterGroupingScopeOptions,
        )

        grouping_path = _find_enabled_adaptive_grouping_path(
            feed_forward_config,
            f"{owner_name}.feed_forward_config",
            grouping_scope=AdaptiveParameterGroupingScopeOptions.ROWS,
        )
        if grouping_path is None:
            return
        raise ValueError(
            f"{owner_name} feed-forward does not support ROWS adaptive parameter "
            "grouping; use SEQUENCE for token inputs or DISABLED. Found grouping "
            f"at {grouping_path}."
        )

    # --- forward-boundary validation ---

    @classmethod
    def validate_encoder_layer_forward_inputs(
        cls,
        model: "TransformerEncoderLayer",
        source_token_embeddings: "Tensor",
        source_key_padding_mask: "Tensor | None" = None,
    ) -> None:
        cls._validate_last_dim(
            source_token_embeddings, model.embedding_dim, "source_token_embeddings"
        )
        cls._validate_transformer_key_padding_mask_shape(
            source_token_embeddings,
            source_key_padding_mask,
            model.self_attention_model,
        )

    @classmethod
    def validate_decoder_layer_forward_inputs(
        cls,
        model: "TransformerDecoderLayer",
        target_token_embeddings: "Tensor",
        encoder_output: "Tensor | None",
        target_key_padding_mask: "Tensor | None" = None,
    ) -> None:
        cls._validate_last_dim(
            target_token_embeddings, model.embedding_dim, "target_token_embeddings"
        )
        cls._validate_transformer_key_padding_mask_shape(
            target_token_embeddings,
            target_key_padding_mask,
            model.self_attention_model,
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

    @staticmethod
    def _validate_transformer_key_padding_mask_shape(
        hidden: "Tensor",
        key_padding_mask: "Tensor | None",
        attention_model,
    ) -> None:
        if key_padding_mask is None:
            return
        if hidden.dim() != 3:
            expected_shape = (hidden.size(0),)
        else:
            batch_first_flag = getattr(attention_model, "batch_first_flag", None)
            if batch_first_flag is None:
                configured_batch_size = getattr(attention_model, "batch_size", None)
                batch_first_flag = hidden.size(1) != configured_batch_size
            batch_axis = 0 if batch_first_flag else 1
            sequence_axis = 1 if batch_first_flag else 0
            expected_shape = (
                hidden.size(batch_axis),
                hidden.size(sequence_axis),
            )
        if tuple(key_padding_mask.shape) != expected_shape:
            raise RuntimeError(
                f"key_padding_mask must have shape {expected_shape}, got "
                f"{tuple(key_padding_mask.shape)}."
            )


class FeedForwardValidator(ValidatorBase):
    @classmethod
    def validate(cls, model: "FeedForward") -> None:
        cls.validate_required_fields(model.cfg)
        cls.validate_dimensions(input_dim=model.input_dim, output_dim=model.output_dim)
        cls._validate_stack_config_type(model.stack_config)
        cls._validate_mirrorable_stack_topology(model.stack_config)

    @staticmethod
    def _validate_stack_config_type(stack_config: ConfigBase) -> None:
        from emperor.experts import MixtureOfExpertsModelConfig
        from emperor.layers import LayerStackConfig, RecurrentCompositionConfig

        if not isinstance(
            stack_config,
            (
                LayerStackConfig,
                MixtureOfExpertsModelConfig,
                RecurrentCompositionConfig,
            ),
        ):
            raise TypeError(
                "FeedForward.stack_config must be a LayerStackConfig, "
                "MixtureOfExpertsModelConfig, or RecurrentCompositionConfig, got "
                f"{type(stack_config).__name__}"
            )

    @classmethod
    def _validate_mirrorable_stack_topology(cls, stack_config: ConfigBase) -> None:
        from emperor.experts import MixtureOfExpertsModelConfig
        from emperor.layers import LayerStackConfig, RecurrentCompositionConfig

        if isinstance(stack_config, LayerStackConfig):
            return
        if isinstance(stack_config, MixtureOfExpertsModelConfig):
            cls._validate_mirrorable_stack_topology(stack_config.stack_config)
            return
        elif isinstance(stack_config, RecurrentCompositionConfig):
            if stack_config._missing_transition_config_fields():
                raise TypeError(
                    "FeedForward cannot mirror stack_config of type NoneType."
                )
            for transition_config in stack_config._transition_configs():
                cls._validate_mirrorable_stack_topology(transition_config)
            return
        else:
            raise TypeError(
                "FeedForward cannot mirror stack_config of type "
                f"{type(stack_config).__name__}."
            )

    @staticmethod
    def validate_forward_inputs(
        flattened_input: "Tensor",
        row_layout: "RowLayout | None",
    ) -> None:
        if row_layout is None or row_layout.row_count == flattened_input.size(0):
            return
        raise ValueError(
            f"row_layout row_count={row_layout.row_count} does not match "
            f"feed-forward row count {flattened_input.size(0)}."
        )

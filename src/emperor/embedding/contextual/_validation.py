from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Real
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from emperor.embedding.contextual._component import ByteContextualEmbedding
    from emperor.embedding.contextual._config import (
        ByteContextualEmbeddingConfig,
        CausalPrefixKernelConfig,
    )
    from emperor.embedding.contextual._kernel import CausalPrefixKernel


class CausalPrefixKernelValidator:
    @classmethod
    def validate(cls, model: CausalPrefixKernel) -> None:
        cls.validate_config(model.cfg)

    @classmethod
    def validate_config(cls, cfg: CausalPrefixKernelConfig) -> None:
        from emperor.embedding.contextual import CausalPrefixKernelConfig
        from emperor.embedding.relative import DynamicPositionalBiasConfig

        if not isinstance(cfg, CausalPrefixKernelConfig):
            raise TypeError(
                f"cfg must be CausalPrefixKernelConfig, got {type(cfg).__name__}"
            )
        cls._validate_positive_integer("hidden_dim", cfg.hidden_dim)
        cls._validate_positive_integer("kernel_dim", cfg.kernel_dim)
        if not isinstance(
            cfg.relative_position_config,
            DynamicPositionalBiasConfig,
        ):
            raise TypeError(
                "relative_position_config must be DynamicPositionalBiasConfig, "
                f"got {type(cfg.relative_position_config).__name__}"
            )
        relative_config = cfg.relative_position_config
        cls._validate_positive_integer(
            "relative_position_config.num_heads",
            relative_config.num_heads,
        )
        cls._validate_positive_integer(
            "relative_position_config.embedding_dim",
            relative_config.embedding_dim,
        )
        cls._validate_positive_integer(
            "relative_position_config.max_positions",
            relative_config.max_positions,
        )
        if relative_config.num_heads != 1:
            raise ValueError(
                "relative_position_config.num_heads must equal 1, "
                f"received {relative_config.num_heads}"
            )
        if relative_config.embedding_dim != cfg.kernel_dim:
            raise ValueError(
                "relative_position_config.embedding_dim must equal kernel_dim, "
                f"received {relative_config.embedding_dim} and {cfg.kernel_dim}"
            )

    @staticmethod
    def _validate_positive_integer(name: str, value: object) -> None:
        if type(value) is not int:
            raise TypeError(f"{name} must be int, got {type(value).__name__}")
        if value <= 0:
            raise ValueError(f"{name} must be greater than 0, received {value}")

    @staticmethod
    def validate_config_type(cfg: object) -> None:
        from emperor.embedding.contextual import CausalPrefixKernelConfig

        if not isinstance(cfg, CausalPrefixKernelConfig):
            raise TypeError(
                f"cfg must be CausalPrefixKernelConfig, got {type(cfg).__name__}"
            )

    @staticmethod
    def validate_overrides_type(overrides: object) -> None:
        from emperor.embedding.contextual import CausalPrefixKernelConfig

        if overrides is not None and not isinstance(
            overrides,
            CausalPrefixKernelConfig,
        ):
            raise TypeError(
                "overrides must be CausalPrefixKernelConfig or None, "
                f"got {type(overrides).__name__}"
            )

    @staticmethod
    def validate_forward_inputs(
        model: CausalPrefixKernel,
        inputs: object,
        attention_mask: object,
    ) -> None:
        if not isinstance(inputs, Tensor):
            raise TypeError(f"inputs must be a Tensor, got {type(inputs).__name__}")
        if inputs.dim() != 3:
            raise ValueError(
                "inputs must have shape (batch, sequence, hidden_dim), "
                f"got {tuple(inputs.shape)}"
            )
        if inputs.size(0) == 0 or inputs.size(1) == 0:
            raise ValueError(
                "inputs batch and sequence dimensions must both be greater than 0, "
                f"got {tuple(inputs.shape[:2])}"
            )
        if not torch.is_floating_point(inputs):
            raise TypeError(f"inputs must be floating point, got {inputs.dtype}")
        if inputs.size(-1) != model.hidden_dim:
            raise ValueError(
                f"inputs final dimension must be {model.hidden_dim}, "
                f"got {inputs.size(-1)}"
            )
        if not isinstance(attention_mask, Tensor):
            raise TypeError(
                f"attention_mask must be a Tensor, got {type(attention_mask).__name__}"
            )
        expected_shape = inputs.shape[:2]
        if tuple(attention_mask.shape) != tuple(expected_shape):
            raise ValueError(
                f"attention_mask must have shape {tuple(expected_shape)}, "
                f"got {tuple(attention_mask.shape)}"
            )
        if attention_mask.dtype is not torch.bool:
            raise TypeError(
                f"attention_mask must use torch.bool, got {attention_mask.dtype}"
            )
        if attention_mask.device != inputs.device:
            raise ValueError(
                "attention_mask and inputs must be on the same device, received "
                f"{attention_mask.device} and {inputs.device}"
            )


class ByteContextualEmbeddingValidator:
    @classmethod
    def validate(cls, model: ByteContextualEmbedding) -> None:
        cfg = model.cfg
        cls._validate_positive_integer("max_token_bytes", cfg.max_token_bytes)
        cls._validate_positive_integer("hidden_dim", cfg.hidden_dim)
        cls._validate_residual_scale(cfg.residual_scale_initial_value)
        cls._validate_nested_config_types(cfg)
        cls._validate_moe(
            "byte_moe_config",
            cfg.byte_moe_config,
            input_dim=cfg.max_token_bytes * 9,
            output_dim=cfg.hidden_dim,
        )
        cls._validate_position_config(cfg)
        cls._validate_prefix_config(cfg)
        cls._validate_moe(
            "context_moe_config",
            cfg.context_moe_config,
            input_dim=cfg.hidden_dim * 2,
            output_dim=cfg.hidden_dim,
        )

    @staticmethod
    def validate_config_type(cfg: object) -> None:
        from emperor.embedding.contextual import ByteContextualEmbeddingConfig

        if not isinstance(cfg, ByteContextualEmbeddingConfig):
            raise TypeError(
                f"cfg must be ByteContextualEmbeddingConfig, got {type(cfg).__name__}"
            )

    @staticmethod
    def validate_overrides_type(overrides: object) -> None:
        from emperor.embedding.contextual import ByteContextualEmbeddingConfig

        if overrides is not None and not isinstance(
            overrides,
            ByteContextualEmbeddingConfig,
        ):
            raise TypeError(
                "overrides must be ByteContextualEmbeddingConfig or None, "
                f"got {type(overrides).__name__}"
            )

    @staticmethod
    def _validate_positive_integer(name: str, value: object) -> None:
        if type(value) is not int:
            raise TypeError(f"{name} must be int, got {type(value).__name__}")
        if value <= 0:
            raise ValueError(f"{name} must be greater than 0, received {value}")

    @staticmethod
    def _validate_residual_scale(value: object) -> None:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(
                "residual_scale_initial_value must be a real scalar, "
                f"got {type(value).__name__}"
            )
        if not math.isfinite(value):
            raise ValueError(
                f"residual_scale_initial_value must be finite, received {value}"
            )

    @staticmethod
    def _validate_nested_config_types(cfg: ByteContextualEmbeddingConfig) -> None:
        from emperor.embedding.absolute import TextLearnedPositionalEmbeddingConfig
        from emperor.embedding.contextual import CausalPrefixKernelConfig
        from emperor.experts import MixtureOfExpertsConfig

        expected_types = (
            ("byte_moe_config", cfg.byte_moe_config, MixtureOfExpertsConfig),
            (
                "positional_embedding_config",
                cfg.positional_embedding_config,
                TextLearnedPositionalEmbeddingConfig,
            ),
            (
                "prefix_kernel_config",
                cfg.prefix_kernel_config,
                CausalPrefixKernelConfig,
            ),
            ("context_moe_config", cfg.context_moe_config, MixtureOfExpertsConfig),
        )
        for name, value, expected_type in expected_types:
            if not isinstance(value, expected_type):
                raise TypeError(
                    f"{name} must be {expected_type.__name__}, "
                    f"got {type(value).__name__}"
                )

    @classmethod
    def _validate_moe(
        cls,
        name: str,
        cfg,
        *,
        input_dim: int,
        output_dim: int,
    ) -> None:
        from emperor.experts import (
            ExpertWeightingPositionOptions,
            RoutingInitializationMode,
        )
        from emperor.sampler import SamplerConfig

        if cfg.input_dim != input_dim:
            raise ValueError(
                f"{name}.input_dim must equal {input_dim}, received {cfg.input_dim}"
            )
        if cfg.output_dim != output_dim:
            raise ValueError(
                f"{name}.output_dim must equal {output_dim}, received {cfg.output_dim}"
            )
        if cfg.capacity_factor != 0.0:
            raise ValueError(
                f"{name}.capacity_factor must equal 0.0, received {cfg.capacity_factor}"
            )
        if cfg.routing_initialization_mode is not RoutingInitializationMode.LAYER:
            raise ValueError(
                f"{name}.routing_initialization_mode must be "
                "RoutingInitializationMode.LAYER"
            )
        if cfg.compute_expert_mixture_flag is not True:
            raise ValueError(f"{name}.compute_expert_mixture_flag must be True")
        if cfg.weighted_parameters_flag is not True:
            raise ValueError(f"{name}.weighted_parameters_flag must be True")
        if (
            cfg.weighting_position_option
            is not ExpertWeightingPositionOptions.AFTER_EXPERTS
        ):
            raise ValueError(
                f"{name}.weighting_position_option must be "
                "ExpertWeightingPositionOptions.AFTER_EXPERTS"
            )
        if not isinstance(cfg.sampler_config, SamplerConfig):
            raise TypeError(
                f"{name}.sampler_config must be SamplerConfig, "
                f"got {type(cfg.sampler_config).__name__}"
            )
        if cfg.sampler_config.normalize_probabilities_flag is not True:
            raise ValueError(
                f"{name}.sampler_config.normalize_probabilities_flag must be True"
            )
        cls._validate_expert_stack(name, cfg)
        cfg.registry_owner().VALIDATOR.validate_config(
            cfg,
            input_dim=input_dim,
            output_dim=output_dim,
        )

    @staticmethod
    def _validate_expert_stack(name: str, cfg) -> None:
        from emperor.augmentations.adaptive_parameters import (
            AdaptiveLinearLayerConfig,
            AdaptiveParameterAugmentationConfig,
            AdaptiveParameterGroupingScopeOptions,
        )
        from emperor.layers import LayerConfig, LayerStackConfig

        stack_config = cfg.expert_model_config
        if not isinstance(stack_config, LayerStackConfig):
            raise TypeError(
                f"{name}.expert_model_config must be LayerStackConfig, "
                f"got {type(stack_config).__name__}"
            )
        if type(stack_config.num_layers) is not int:
            raise TypeError(
                f"{name}.expert_model_config.num_layers must be int, "
                f"got {type(stack_config.num_layers).__name__}"
            )
        if stack_config.num_layers != 2:
            raise ValueError(
                f"{name}.expert_model_config.num_layers must equal 2, "
                f"received {stack_config.num_layers}"
            )
        layer_config = stack_config.layer_config
        if not isinstance(layer_config, LayerConfig):
            raise TypeError(
                f"{name}.expert_model_config.layer_config must be LayerConfig, "
                f"got {type(layer_config).__name__}"
            )
        layer_model_config = layer_config.layer_model_config
        if not isinstance(layer_model_config, AdaptiveLinearLayerConfig):
            raise TypeError(
                f"{name}.expert layer model must be AdaptiveLinearLayerConfig, "
                f"got {type(layer_model_config).__name__}"
            )
        augmentation_config = layer_model_config.adaptive_augmentation_config
        if not isinstance(
            augmentation_config,
            AdaptiveParameterAugmentationConfig,
        ):
            raise TypeError(
                f"{name}.expert adaptive augmentation must be "
                "AdaptiveParameterAugmentationConfig, "
                f"got {type(augmentation_config).__name__}"
            )
        if (
            augmentation_config.grouping_scope
            is not AdaptiveParameterGroupingScopeOptions.DISABLED
        ):
            raise ValueError(
                f"{name}.expert adaptive grouping must be "
                "AdaptiveParameterGroupingScopeOptions.DISABLED"
            )
        adaptive_components = (
            augmentation_config.diagonal_config,
            augmentation_config.weight_config,
            augmentation_config.bias_config,
            augmentation_config.mask_config,
        )
        if not any(component is not None for component in adaptive_components):
            raise ValueError(
                f"{name}.expert adaptive augmentation must configure at least one "
                "adaptive parameter component"
            )

    @classmethod
    def _validate_position_config(
        cls,
        cfg: ByteContextualEmbeddingConfig,
    ) -> None:
        position_config = cfg.positional_embedding_config
        cls._validate_positive_integer(
            "positional_embedding_config.num_embeddings",
            position_config.num_embeddings,
        )
        cls._validate_positive_integer(
            "positional_embedding_config.embedding_dim",
            position_config.embedding_dim,
        )
        if position_config.embedding_dim != cfg.hidden_dim:
            raise ValueError(
                "positional_embedding_config.embedding_dim must equal hidden_dim, "
                f"received {position_config.embedding_dim} and {cfg.hidden_dim}"
            )
        if position_config.padding_idx != 0:
            raise ValueError(
                "positional_embedding_config.padding_idx must equal 0, "
                f"received {position_config.padding_idx}"
            )

    @staticmethod
    def _validate_prefix_config(cfg: ByteContextualEmbeddingConfig) -> None:
        prefix_config = cfg.prefix_kernel_config
        if prefix_config.hidden_dim != cfg.hidden_dim:
            raise ValueError(
                "prefix_kernel_config.hidden_dim must equal hidden_dim, "
                f"received {prefix_config.hidden_dim} and {cfg.hidden_dim}"
            )
        prefix_config.registry_owner().VALIDATOR.validate_config(prefix_config)

    @classmethod
    def validate_forward_inputs(
        cls,
        token_texts: object,
        attention_mask: object,
        *,
        maximum_sequence_length: int,
    ) -> tuple[int, int]:
        if not cls._is_non_string_sequence(token_texts):
            raise TypeError("token_texts must be a sequence of token-text sequences")
        batch_size = len(token_texts)
        if batch_size == 0:
            raise ValueError("token_texts must contain at least one batch row")
        first_row = token_texts[0]
        if not cls._is_non_string_sequence(first_row):
            raise TypeError("each token_texts row must be a sequence")
        sequence_length = len(first_row)
        if sequence_length == 0:
            raise ValueError("token_texts rows must contain at least one token")
        for batch_index, row in enumerate(token_texts):
            if not cls._is_non_string_sequence(row):
                raise TypeError(f"token_texts[{batch_index}] must be a sequence")
            if len(row) != sequence_length:
                raise ValueError("token_texts must be rectangular")
            for token_index, token_text in enumerate(row):
                if type(token_text) is not str:
                    raise TypeError(
                        f"token_texts[{batch_index}][{token_index}] must be exact str, "
                        f"got {type(token_text).__name__}"
                    )
        if sequence_length > maximum_sequence_length:
            raise ValueError(
                "token_texts sequence length exceeds the configured maximum "
                f"sequence length {maximum_sequence_length}"
            )
        cls._validate_attention_mask(
            attention_mask,
            expected_shape=(batch_size, sequence_length),
        )
        return batch_size, sequence_length

    @staticmethod
    def _is_non_string_sequence(value: object) -> bool:
        return isinstance(value, Sequence) and not isinstance(
            value,
            (str, bytes, bytearray),
        )

    @staticmethod
    def _validate_attention_mask(
        attention_mask: object,
        *,
        expected_shape: tuple[int, int],
    ) -> None:
        if attention_mask is None:
            return
        if not isinstance(attention_mask, Tensor):
            raise TypeError(
                "attention_mask must be a Tensor or None, "
                f"got {type(attention_mask).__name__}"
            )
        if attention_mask.dtype is not torch.bool:
            raise TypeError(
                f"attention_mask must use torch.bool, got {attention_mask.dtype}"
            )
        if tuple(attention_mask.shape) != expected_shape:
            raise ValueError(
                f"attention_mask must have shape {expected_shape}, "
                f"got {tuple(attention_mask.shape)}"
            )

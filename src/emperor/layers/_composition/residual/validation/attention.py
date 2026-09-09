from __future__ import annotations

import math
from numbers import Real
from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor

from emperor.layers._composition.residual.validation.common import (
    ResidualConnectionValidator,
)

if TYPE_CHECKING:
    from emperor.layers._composition.residual.base import (
        ResidualConnectionAbstract,
        ResidualState,
    )
    from emperor.layers._composition.residual.config import ResidualConfig
    from emperor.layers._composition.residual.variants.attention import (
        AttentionResidual,
        AttentionResidualState,
    )
    from emperor.layers._config import LayerStackConfig


class AttentionResidualValidator(ResidualConnectionValidator):
    OPTIONAL_FIELDS = {"model_config"}

    @staticmethod
    def validate_state_lifecycle(connection: ResidualConnectionAbstract) -> None:
        if connection.residual_state_lifecycle is None:
            raise RuntimeError(
                f"{type(connection).__name__} requires forward-local residual "
                "state but does not provide a ResidualStateLifecycle."
            )

    @staticmethod
    def validate_stateless_execution(
        config: ResidualConfig,
        *,
        owner_name: str,
    ) -> None:
        raise ValueError(
            f"{type(config).__name__} is not supported for {owner_name}; "
            "the execution owner must support forward-local residual state "
            "and depth-specific residual connections."
        )

    @classmethod
    def _validate_config(cls, config: object) -> None:
        cls.validate_required_fields(config)
        cls.validate_field_types(config)
        cls.validate_positive_integer(config.residual_dim, name="residual_dim")
        cls.validate_positive_integer(config.block_size, name="block_size")
        cls.validate_finite_positive_number(
            config.rms_norm_epsilon,
            name="rms_norm_epsilon",
        )
        cls.__validate_query_model_config(config)

    @classmethod
    def __validate_query_model_config(cls, config: object) -> None:
        model_config = config.model_config
        if model_config is None:
            return
        from emperor.layers import LayerConfig, LayerStackConfig
        from emperor.linears import LinearLayerConfig

        if isinstance(model_config, LinearLayerConfig):
            return
        if not isinstance(model_config, LayerStackConfig):
            raise TypeError(
                "AttentionResidualConfig.model_config must be a LayerStackConfig "
                "or LinearLayerConfig when provided, "
                f"got {type(model_config).__name__}."
            )
        layer_config = model_config.layer_config
        if type(layer_config) is not LayerConfig:
            raise TypeError(
                "AttentionResidualConfig.model_config.layer_config must be "
                f"exactly LayerConfig, got {type(layer_config).__name__}."
            )
        if not isinstance(layer_config.layer_model_config, LinearLayerConfig):
            raise TypeError(
                "AttentionResidualConfig.model_config.layer_config."
                "layer_model_config must be LinearLayerConfig, "
                f"got {type(layer_config.layer_model_config).__name__}."
            )
        cls.__validate_query_model_controllers(model_config, layer_config)

    @staticmethod
    def __validate_query_model_controllers(
        model_config: object, layer_config: object
    ) -> None:
        nested_configs = {
            "layer_config.gate_config": layer_config.gate_config,
            "layer_config.halting_config": layer_config.halting_config,
            "layer_config.memory_config": layer_config.memory_config,
            "shared_gate_config": model_config.shared_gate_config,
            "shared_halting_config": model_config.shared_halting_config,
            "shared_memory_config": model_config.shared_memory_config,
        }
        for path, nested_config in nested_configs.items():
            if nested_config is not None:
                raise ValueError(
                    f"AttentionResidualConfig.model_config.{path} must be None "
                    "for a residual query model."
                )

    @staticmethod
    def validate_query_model_output(query: object, current: Tensor) -> None:
        if not isinstance(query, Tensor) or not torch.is_floating_point(query):
            raise TypeError(
                "attention residual query model must return a floating-point tensor."
            )
        if query.shape != current.shape:
            raise ValueError(
                "attention residual query model output must have shape "
                f"{tuple(current.shape)}, got {tuple(query.shape)}."
            )
        if query.device != current.device:
            raise ValueError(
                "attention residual query model output must be on device "
                f"{current.device}, got {query.device}."
            )

    @staticmethod
    def validate_stack_config(config: LayerStackConfig) -> None:
        config_name = type(config.layer_config.residual_config).__name__
        if (
            config.input_dim != config.hidden_dim
            or config.hidden_dim != config.output_dim
        ):
            raise ValueError(
                "input_dim, hidden_dim, and output_dim must all be equal when "
                f"{config_name} is enabled, "
                f"got input_dim={config.input_dim}, hidden_dim={config.hidden_dim}, "
                f"output_dim={config.output_dim}."
            )
        if not config.apply_output_postprocessing_flag:
            raise ValueError(
                "apply_output_postprocessing_flag must be True when "
                f"{config_name} is enabled so the final layer performs the "
                "required final aggregation."
            )
        if (
            config.shared_halting_config is not None
            or config.layer_config.halting_config is not None
        ):
            raise ValueError(
                f"halting cannot be combined with {config_name} until residual "
                "history masking and finalization semantics are defined."
            )

    @staticmethod
    def validate_positive_integer(value: object, *, name: str) -> None:
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")

    @staticmethod
    def validate_finite_positive_number(value: object, *, name: str) -> None:
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{name} must be a finite positive number.")

    @staticmethod
    def validate_source(source: object, *, residual_dim: int) -> None:
        if not isinstance(source, Tensor) or not torch.is_floating_point(source):
            raise TypeError(
                "attention residual sources must be floating-point tensors."
            )
        if source.ndim == 0 or source.shape[-1] != residual_dim:
            raise ValueError(
                "attention residual source last dimension must equal "
                f"residual_dim {residual_dim}."
            )

    @classmethod
    def validate_created_attention_state(
        cls,
        connection: AttentionResidual,
        state: ResidualState | None,
    ) -> None:
        if state is None:
            raise RuntimeError(
                "AttentionResidual failed to create forward-local residual state."
            )
        cls.validate_attention_state(connection, state)

    @classmethod
    def validate_attention_forward_inputs(
        cls,
        connection: AttentionResidual,
        current: Tensor,
        state: object,
    ) -> None:
        cls.validate_attention_state(connection, state)
        cls.validate_compatible_sources(
            cast("AttentionResidualState", state),
            current,
            residual_dim=connection.residual_dim,
        )

    @staticmethod
    def validate_attention_state(
        connection: AttentionResidual,
        state: object,
    ) -> None:
        from emperor.layers._composition.residual.variants.attention import (
            AttentionResidualState,
        )

        if not isinstance(state, AttentionResidualState):
            raise TypeError(
                "residual_state must be an AttentionResidualState, "
                f"got {type(state).__name__}."
            )
        if state.block_size != connection.block_size:
            raise ValueError(
                f"residual_state block_size {state.block_size} does not match "
                f"configured block_size {connection.block_size}."
            )

    @classmethod
    def validate_compatible_sources(
        cls,
        state: AttentionResidualState,
        current: Tensor,
        *,
        residual_dim: int,
    ) -> None:
        expected_shape = state.initial_source.shape
        expected_device = state.initial_source.device
        for source in (*state.sources, current):
            cls.validate_source(source, residual_dim=residual_dim)
            if source.shape != expected_shape:
                raise ValueError(
                    "all attention residual sources must have shape "
                    f"{tuple(expected_shape)}."
                )
            if source.device != expected_device:
                raise ValueError(
                    "all attention residual sources must be on device "
                    f"{expected_device}."
                )

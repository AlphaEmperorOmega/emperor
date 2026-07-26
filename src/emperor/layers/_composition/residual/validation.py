from __future__ import annotations

import math
from numbers import Real
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from emperor._validation import ValidatorBase

if TYPE_CHECKING:
    from emperor.layers._composition.residual.base import (
        ResidualConnectionAbstract,
    )
    from emperor.layers._composition.residual.config import ResidualConfig
    from emperor.layers._composition.residual.variants.attention import (
        AttentionResidualState,
    )


class ResidualConnectionValidator(ValidatorBase):
    @classmethod
    def validate(cls, model: ResidualConnectionAbstract) -> None:
        from emperor.layers._composition.residual.config import (
            AttentionResidualConfig,
            ResidualConfig,
            WeightedBlendResidualConfig,
            WeightedResidualConfig,
        )

        config = model.cfg
        if not isinstance(config, ResidualConfig):
            raise TypeError(
                "residual connection cfg must be a ResidualConfig, "
                f"got {type(config).__name__}."
            )
        cls._validate_concrete_config(config, owner_name=type(model).__name__)
        expected_owner = config.registry_owner()
        if not isinstance(model, expected_owner):
            raise TypeError(
                f"{type(config).__name__} builds {expected_owner.__name__}, not "
                f"{type(model).__name__}."
            )
        if isinstance(config, AttentionResidualConfig):
            cls._validate_attention_config(config)
        else:
            cls._validate_optional_residual_dim(config.residual_dim)
        if isinstance(
            config,
            (WeightedResidualConfig, WeightedBlendResidualConfig),
        ):
            cls._validate_weighted_config(config)

    @classmethod
    def validate_residual_config(
        cls,
        residual_config: ResidualConfig | None,
        owner_name: str,
    ) -> None:
        if residual_config is None:
            return
        from emperor.layers._composition.residual.config import ResidualConfig

        if not isinstance(residual_config, ResidualConfig):
            raise TypeError(
                "residual_config must be an instance of ResidualConfig for "
                f"{owner_name}, got {type(residual_config).__name__}"
            )
        cls._validate_concrete_config(residual_config, owner_name=owner_name)

    @staticmethod
    def _validate_concrete_config(
        residual_config: ResidualConfig,
        *,
        owner_name: str,
    ) -> None:
        try:
            residual_config.registry_owner()
        except (NotImplementedError, ValueError) as exc:
            raise ValueError(
                f"residual_config must be a concrete residual config for {owner_name}"
            ) from exc

    @staticmethod
    def _validate_optional_residual_dim(residual_dim: int | None) -> None:
        if residual_dim is None:
            return
        if isinstance(residual_dim, bool) or not isinstance(residual_dim, int):
            raise TypeError(
                "residual_dim must be int for a residual config, "
                f"got {type(residual_dim).__name__}."
            )
        if residual_dim <= 0:
            raise ValueError(
                f"residual_dim must be greater than 0, received {residual_dim}"
            )

    @classmethod
    def _validate_weighted_config(cls, config: object) -> None:
        model_config = config.model_config
        if model_config is None:
            return
        from emperor.linears import LinearLayerConfig

        if not isinstance(model_config, LinearLayerConfig):
            raise TypeError(
                f"{type(config).__name__}.model_config must be a "
                "LinearLayerConfig when provided, "
                f"got {type(model_config).__name__}."
            )
        if model_config.bias_flag is not True:
            raise ValueError(
                f"{type(config).__name__}.model_config.bias_flag must be True so "
                "the initial mixing coefficient can be represented."
            )
        if config.residual_dim is None:
            raise TypeError(
                f"residual_dim must be int for {type(config).__name__} when "
                "model_config is provided, got NoneType."
            )

    @classmethod
    def _validate_attention_config(cls, config: object) -> None:
        cls.validate_positive_integer(config.residual_dim, name="residual_dim")
        if config.block_size is not None:
            cls.validate_positive_integer(config.block_size, name="block_size")
        if config.rms_norm_epsilon is not None:
            cls.validate_finite_positive_number(
                config.rms_norm_epsilon,
                name="rms_norm_epsilon",
            )

    @staticmethod
    def validate_raw_mix_coefficient(raw_mix_coefficient: Tensor | None) -> None:
        if raw_mix_coefficient is None:
            raise RuntimeError(
                "weighted residual requires either raw_weight or a coefficient model."
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
            or not math.isfinite(float(value))
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
    def validate_attention_forward_inputs(
        cls,
        current: Tensor,
        state: object,
        *,
        residual_dim: int,
        block_size: int,
    ) -> None:
        cls.validate_attention_state(state, block_size=block_size)
        cls.validate_compatible_sources(
            state,
            current,
            residual_dim=residual_dim,
        )

    @staticmethod
    def validate_attention_state(state: object, *, block_size: int) -> None:
        from emperor.layers._composition.residual.variants.attention import (
            AttentionResidualState,
        )

        if not isinstance(state, AttentionResidualState):
            raise TypeError(
                "residual_state must be an AttentionResidualState, "
                f"got {type(state).__name__}."
            )
        if state.block_size != block_size:
            raise ValueError(
                f"residual_state block_size {state.block_size} does not match "
                f"configured block_size {block_size}."
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

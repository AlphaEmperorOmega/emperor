from __future__ import annotations

import math
from numbers import Real
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from emperor._validation import ValidatorBase

if TYPE_CHECKING:
    from emperor.layers._composition.attention_residual import AttentionResidualState


class AttentionResidualValidator(ValidatorBase):
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
    def validate_forward_inputs(
        cls,
        current: Tensor,
        state: AttentionResidualState,
        *,
        residual_dim: int,
        block_size: int,
    ) -> None:
        cls.validate_state(state, block_size=block_size)
        cls.validate_compatible_sources(
            state,
            current,
            residual_dim=residual_dim,
        )

    @staticmethod
    def validate_state(
        state: object,
        *,
        block_size: int,
    ) -> None:
        from emperor.layers._composition.attention_residual import (
            AttentionResidualState,
        )

        if not isinstance(state, AttentionResidualState):
            raise TypeError(
                f"state must be an AttentionResidualState, got {type(state).__name__}."
            )
        if state.block_size != block_size:
            raise ValueError(
                f"state block_size {state.block_size} does not match configured "
                f"block_size {block_size}."
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
            cls.validate_source(
                source,
                residual_dim=residual_dim,
            )
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

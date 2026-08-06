from __future__ import annotations

import math
from numbers import Real
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from emperor._validation import ValidatorBase
from emperor.layers._composition.residual.base import ResidualRuntimeRequirement
from emperor.layers._composition.residual.validation import (
    ResidualConnectionValidator,
)
from emperor.layers._options import LayerNormPositionOptions
from emperor.layers._validation.common import (
    _HALTING_CONFIG_FIELDS,
    _MEMORY_CONFIG_FIELDS,
    _matches_config_contract,
    _validate_halting_lifecycle_owner,
    _validate_no_grouping_with_context_controllers,
)
from emperor.layers._validation.gate import LayerGateValidator

_GRADIENT_WINDOW_FIELDS = {"no_gradient_transition_count"}
_RECURRENT_CONTROLLER_OPTIONAL_FIELDS = {
    "recurrent_layer_norm_position",
    "gate_config",
    "residual_config",
    "halting_config",
    "memory_config",
}

if TYPE_CHECKING:
    from emperor.layers._composition.recurrent.schedule import (
        DepthwiseRecurrentResidualSchedule,
        RecurrentResidualSchedule,
    )


class RecurrentResidualScheduleValidator(ValidatorBase):
    @classmethod
    def validate(cls, schedule: RecurrentResidualSchedule) -> None:
        transition_count = schedule.transition_count
        if (
            isinstance(transition_count, bool)
            or not isinstance(transition_count, int)
            or transition_count <= 0
        ):
            raise ValueError("transition_count must be a positive integer.")

    @staticmethod
    def validate_transition_index(
        schedule: RecurrentResidualSchedule,
        transition_index: object,
    ) -> None:
        if (
            isinstance(transition_index, bool)
            or not isinstance(transition_index, int)
            or not 0 <= transition_index < schedule.transition_count
        ):
            raise IndexError(
                "transition_index must identify a configured recurrent transition."
            )

    @staticmethod
    def validate_subsequent_connections(
        schedule: DepthwiseRecurrentResidualSchedule,
    ) -> None:
        expected_connection_count = schedule.transition_count - 1
        if len(schedule.subsequent_connections) != expected_connection_count:
            raise ValueError(
                "subsequent_connections must contain one connection for every "
                "transition after transition zero."
            )


def _validate_initialization_standard_deviation(value: object) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or value < 0
    ):
        raise ValueError(
            "initialization_standard_deviation must be a finite non-negative number."
        )


def _validate_variant_hidden(
    hidden: object,
    expected_feature_dim: int,
    *,
    field_name: str,
    owner_name: str,
) -> None:
    if not isinstance(hidden, Tensor) or not torch.is_floating_point(hidden):
        raise TypeError(f"{field_name} must be a floating-point Tensor.")
    if hidden.ndim < 2:
        raise ValueError(f"{field_name} must have rank >= 2 with feature-last layout.")
    if any(dimension == 0 for dimension in hidden.shape[:-1]):
        raise ValueError(f"{field_name} must have non-empty leading dimensions.")
    if hidden.shape[-1] != expected_feature_dim:
        raise ValueError(
            f"{field_name} last dimension must be {expected_feature_dim} for "
            f"{owner_name}, got {hidden.shape[-1]}."
        )


def _validate_variant_state(
    state: object,
    expected_feature_dim: int,
    *,
    owner_name: str,
) -> None:
    from emperor.layers._state import LayerState

    if not isinstance(state, LayerState):
        raise TypeError(
            f"state must be an instance of LayerState for {owner_name}, "
            f"got {type(state).__name__}"
        )
    _validate_variant_hidden(
        state.hidden,
        expected_feature_dim,
        field_name="state.hidden",
        owner_name=owner_name,
    )


def _validate_variant_transition_output(
    output_state: object,
    transition_input: Tensor,
    expected_row_layout: object,
    *,
    expected_feature_dim: int,
    owner_name: str,
    transition_name: str,
) -> None:
    from emperor.layers._state import LayerState

    if not isinstance(output_state, LayerState):
        raise TypeError(
            f"{transition_name} transition block must return LayerState, got "
            f"{type(output_state).__name__}."
        )
    _validate_variant_hidden(
        output_state.hidden,
        expected_feature_dim,
        field_name="transition output hidden",
        owner_name=owner_name,
    )
    if output_state.hidden.shape != transition_input.shape:
        raise ValueError(
            f"{transition_name} transition block must preserve hidden shape."
        )
    if output_state.hidden.dtype != transition_input.dtype:
        raise ValueError(
            f"{transition_name} transition block must preserve hidden dtype."
        )
    if output_state.hidden.device != transition_input.device:
        raise ValueError(
            f"{transition_name} transition block must preserve hidden device."
        )
    if output_state.row_layout is not expected_row_layout:
        raise ValueError(
            f"{transition_name} transition block must preserve the exact "
            "row_layout object."
        )


def _validate_transition_gradient_window(
    config: object,
    *,
    total_transition_count: int,
) -> None:
    no_gradient_count = config.no_gradient_transition_count
    if no_gradient_count is None:
        return
    if isinstance(no_gradient_count, bool) or not isinstance(no_gradient_count, int):
        raise TypeError(
            "no_gradient_transition_count must be int, "
            f"got {type(no_gradient_count).__name__}."
        )
    if no_gradient_count < 0:
        raise ValueError(
            "no_gradient_transition_count must be greater than or equal to 0."
        )
    if no_gradient_count >= total_transition_count:
        raise ValueError(
            "no_gradient_transition_count must be less than the variant's "
            f"{total_transition_count} scheduled transitions so at least one "
            "transition uses gradients."
        )


def _validate_recurrent_controller_config(
    config: object,
    *,
    supported_residual_requirements: frozenset[
        ResidualRuntimeRequirement
    ] = frozenset(),
) -> None:
    owner_name = type(config).__name__
    recurrent_layer_norm_position = config.recurrent_layer_norm_position
    if recurrent_layer_norm_position is not None and not isinstance(
        recurrent_layer_norm_position,
        LayerNormPositionOptions,
    ):
        raise TypeError(
            "recurrent_layer_norm_position must be None or a "
            "LayerNormPositionOptions value for "
            f"{owner_name}, got {type(recurrent_layer_norm_position).__name__}."
        )

    LayerGateValidator.validate_recurrent_gate_config(
        config.gate_config,
        owner_name=f"{owner_name}.gate_config",
    )
    ResidualConnectionValidator.validate_residual_config(
        config.residual_config,
        owner_name=owner_name,
    )
    residual_config = config.residual_config
    if residual_config is not None:
        residual_owner = residual_config.registry_owner()
        residual_requirements = getattr(
            residual_owner,
            "RUNTIME_REQUIREMENTS",
            frozenset(),
        )
        unsupported_requirements = residual_requirements.difference(
            supported_residual_requirements
        )
    else:
        unsupported_requirements = frozenset()
    if unsupported_requirements:
        requirement_names = ", ".join(
            sorted(requirement.value for requirement in unsupported_requirements)
        )
        raise ValueError(
            f"{type(residual_config).__name__} is not supported for {owner_name}; "
            "the recurrent owner does not satisfy residual runtime requirements: "
            f"{requirement_names}."
        )

    halting_config = config.halting_config
    if halting_config is not None:
        if not _matches_config_contract(halting_config, _HALTING_CONFIG_FIELDS):
            raise TypeError(
                "halting_config must be an instance of HaltingConfig for "
                f"{owner_name}, got {type(halting_config).__name__}."
            )
        _validate_halting_lifecycle_owner(
            halting_config,
            field_name="halting_config",
            owner_name=owner_name,
        )

    memory_config = config.memory_config
    if memory_config is not None and not _matches_config_contract(
        memory_config,
        _MEMORY_CONFIG_FIELDS,
    ):
        raise TypeError(
            "memory_config must be an instance of DynamicMemoryConfig for "
            f"{owner_name}, got {type(memory_config).__name__}"
        )

    _validate_no_grouping_with_context_controllers(
        config,
        owner_name=owner_name,
        controllers=(
            ("halting_config", halting_config),
            ("memory_config", memory_config),
        ),
    )


class _RecurrentCompositionValidator(ValidatorBase):
    @staticmethod
    def validate_halting_output(
        output_hidden: Tensor,
        candidate_hidden: Tensor,
    ) -> None:
        if output_hidden.shape != candidate_hidden.shape:
            raise ValueError(
                "recurrent halting must preserve the candidate hidden shape."
            )
        if output_hidden.dtype != candidate_hidden.dtype:
            raise ValueError(
                "recurrent halting must preserve the candidate hidden dtype."
            )
        if output_hidden.device != candidate_hidden.device:
            raise ValueError(
                "recurrent halting must preserve the candidate hidden device."
            )

    @staticmethod
    def validate_initial_buffer(
        buffer: Tensor,
        hidden: Tensor,
        *,
        name: str,
        expected_feature_dim: int,
    ) -> None:
        if tuple(buffer.shape) != (expected_feature_dim,):
            raise ValueError(
                f"{name} must have shape ({expected_feature_dim},), got "
                f"{tuple(buffer.shape)}."
            )
        if buffer.dtype != hidden.dtype or buffer.device != hidden.device:
            raise ValueError(
                f"{name} dtype/device must match state.hidden; move the recurrent "
                "composition to the input dtype and device before forward."
            )

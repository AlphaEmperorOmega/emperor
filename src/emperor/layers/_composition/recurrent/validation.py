from __future__ import annotations

import math
from dataclasses import fields
from numbers import Real
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from emperor._validation import ValidatorBase
from emperor.config import ConfigBase
from emperor.layers._composition.residual.config import AttentionResidualConfig
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

if TYPE_CHECKING:
    from emperor.layers._composition.recurrent.variants.standard import RecurrentLayer
    from emperor.layers._state import LayerState


_GRADIENT_WINDOW_FIELDS = {"no_gradient_transition_count"}
_RECURRENT_CONTROLLER_OPTIONAL_FIELDS = {
    "recurrent_layer_norm_position",
    "gate_config",
    "residual_config",
    "halting_config",
    "memory_config",
}


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


def _validate_recurrent_controller_config(config: object) -> None:
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
    if isinstance(config.residual_config, AttentionResidualConfig):
        raise ValueError(
            f"AttentionResidualConfig is not supported for {owner_name} until "
            "recurrent depth owns a distinct learned query and an explicit "
            "forward-local history bridge."
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


class RecurrentLayerValidator(_RecurrentCompositionValidator):
    OPTIONAL_FIELDS = {
        *_RECURRENT_CONTROLLER_OPTIONAL_FIELDS,
        "reinject_original_hidden_flag",
        "override_config",
        *_GRADIENT_WINDOW_FIELDS,
    }

    @classmethod
    def validate(cls, model: RecurrentLayer) -> None:
        from emperor.layers._composition.recurrent.config import RecurrentLayerConfig

        cfg = model.cfg
        if not isinstance(cfg, RecurrentLayerConfig):
            raise TypeError(
                "RecurrentLayer cfg must be a RecurrentLayerConfig, "
                f"got {type(cfg).__name__}."
            )
        cls.validate_required_fields(cfg)
        cls._validate_integer_field(
            "input_dim",
            cfg.input_dim,
        )
        cls._validate_integer_field(
            "output_dim",
            cfg.output_dim,
        )
        cls._validate_integer_field(
            "max_steps",
            cfg.max_steps,
        )
        cls.validate_dimensions(
            input_dim=cfg.input_dim,
            output_dim=cfg.output_dim,
            max_steps=cfg.max_steps,
        )
        _validate_transition_gradient_window(
            cfg,
            total_transition_count=cfg.max_steps,
        )
        cls.__validate_reinject_original_hidden_flag(cfg.reinject_original_hidden_flag)
        cls.__validate_stable_dimensions(
            cfg.input_dim,
            cfg.output_dim,
        )
        cls.__validate_block_config(cfg.block_config)
        _validate_recurrent_controller_config(cfg)
        expected_owner = cfg.registry_owner()
        if not isinstance(model, expected_owner):
            raise TypeError(
                f"{type(cfg).__name__} builds {expected_owner.__name__}, not "
                f"{type(model).__name__}."
            )

    @classmethod
    def validate_state(cls, state: LayerState, expected_feature_dim: int) -> None:
        from emperor.layers._state import LayerState

        if not isinstance(state, LayerState):
            raise TypeError(
                f"state must be an instance of LayerState for RecurrentLayer, "
                f"got {type(state).__name__}"
            )
        cls.validate_hidden(
            state.hidden,
            expected_feature_dim,
            "state.hidden",
        )

    @classmethod
    def validate_transition_output(
        cls,
        output_state: object,
        transition_input: Tensor,
        expected_row_layout: object,
        *,
        expected_feature_dim: int,
    ) -> None:
        from emperor.layers._state import LayerState

        if not isinstance(output_state, LayerState):
            raise TypeError(
                "recurrent transition block must return LayerState, got "
                f"{type(output_state).__name__}."
            )
        cls.validate_candidate(
            output_state.hidden,
            transition_input,
            expected_feature_dim,
        )
        if output_state.hidden.dtype != transition_input.dtype:
            raise ValueError("recurrent transition block must preserve hidden dtype.")
        if output_state.hidden.device != transition_input.device:
            raise ValueError("recurrent transition block must preserve hidden device.")
        cls.validate_row_layout_preserved(
            output_state.row_layout,
            expected_row_layout,
        )

    @staticmethod
    def validate_hidden(
        hidden: Tensor,
        expected_feature_dim: int,
        field_name: str = "hidden",
    ) -> None:
        if hidden.dim() < 2:
            raise ValueError(
                f"{field_name} must have rank >= 2 with feature-last layout, "
                f"got {hidden.dim()}D tensor with shape {tuple(hidden.shape)}"
            )
        actual_feature_dim = hidden.shape[-1]
        if actual_feature_dim != expected_feature_dim:
            raise ValueError(
                f"{field_name} last dimension must be {expected_feature_dim} "
                f"for RecurrentLayer, got {actual_feature_dim} with shape "
                f"{tuple(hidden.shape)}"
            )

    @classmethod
    def validate_candidate(
        cls,
        candidate: Tensor,
        previous_hidden: Tensor,
        expected_feature_dim: int,
    ) -> None:
        cls.validate_hidden(candidate, expected_feature_dim)
        if candidate.shape != previous_hidden.shape:
            raise ValueError(
                f"recurrent block must preserve hidden shape, got candidate "
                f"shape {tuple(candidate.shape)} and previous shape "
                f"{tuple(previous_hidden.shape)}"
            )

    @staticmethod
    def validate_row_layout_preserved(candidate_layout, expected_layout) -> None:
        if candidate_layout is expected_layout:
            return
        raise ValueError(
            "recurrent block must preserve the exact row_layout object; row "
            "selection, reordering, or replacement requires an explicit layout "
            "transformation contract."
        )

    @staticmethod
    def _validate_integer_field(field_name: str, value: int) -> None:
        if not isinstance(value, int):
            raise TypeError(
                f"{field_name} must be int for RecurrentLayerConfig, "
                f"got {type(value).__name__}"
            )

    @staticmethod
    def __validate_stable_dimensions(input_dim: int, output_dim: int) -> None:
        if input_dim != output_dim:
            raise ValueError(
                f"input_dim and output_dim must be equal for RecurrentLayerConfig, "
                f"got input_dim={input_dim} and output_dim={output_dim}."
            )

    @staticmethod
    def __validate_reinject_original_hidden_flag(value: bool | None) -> None:
        if value is not None and not isinstance(value, bool):
            raise TypeError(
                "reinject_original_hidden_flag must be bool or None for "
                "RecurrentLayerConfig, "
                f"got {type(value).__name__}."
            )

    @staticmethod
    def __validate_block_config(block_config: ConfigBase) -> None:
        if not isinstance(block_config, ConfigBase):
            raise TypeError(
                f"block_config must be an instance of ConfigBase for "
                f"RecurrentLayerConfig, "
                f"got {type(block_config).__name__}"
            )

        field_names = {field.name for field in fields(block_config)}
        missing_fields = {"input_dim", "output_dim"} - field_names
        if missing_fields:
            missing_field_list = ", ".join(sorted(missing_fields))
            raise TypeError(
                f"block_config must declare dataclass fields input_dim and "
                f"output_dim for RecurrentLayerConfig; "
                f"{type(block_config).__name__} is missing {missing_field_list}"
            )


class TinyRecursiveModelRecurrentValidator(_RecurrentCompositionValidator):
    OPTIONAL_FIELDS = {
        *_RECURRENT_CONTROLLER_OPTIONAL_FIELDS,
        *_GRADIENT_WINDOW_FIELDS,
    }

    @classmethod
    def validate(cls, model: object) -> None:
        from emperor.layers._composition.recurrent.config import (
            TinyRecursiveModelRecurrentConfig,
        )

        config = model.cfg
        if not isinstance(config, TinyRecursiveModelRecurrentConfig):
            raise TypeError(
                "TinyRecursiveModelRecurrent cfg must be a TinyRecursiveModelRecurrentConfig, "
                f"got {type(config).__name__}."
            )
        cls.validate_required_fields(config)
        for field_name in (
            "input_dim",
            "output_dim",
            "latent_updates_per_answer_update",
            "answer_update_count",
        ):
            cls.__validate_integer_field(field_name, getattr(config, field_name))
        cls.validate_dimensions(
            input_dim=config.input_dim,
            output_dim=config.output_dim,
            latent_updates_per_answer_update=(config.latent_updates_per_answer_update),
            answer_update_count=config.answer_update_count,
        )
        _validate_transition_gradient_window(
            config,
            total_transition_count=(
                config.answer_update_count
                * (config.latent_updates_per_answer_update + 1)
            ),
        )
        if config.input_dim != config.output_dim:
            raise ValueError(
                "input_dim and output_dim must be equal for TinyRecursiveModelRecurrentConfig, "
                f"got input_dim={config.input_dim} and "
                f"output_dim={config.output_dim}."
            )
        _validate_recurrent_controller_config(config)
        cls.__validate_block_config(config.block_config)
        _validate_initialization_standard_deviation(
            config.initialization_standard_deviation
        )
        expected_owner = config.registry_owner()
        if not isinstance(model, expected_owner):
            raise TypeError(
                f"{type(config).__name__} builds {expected_owner.__name__}, not "
                f"{type(model).__name__}."
            )

    @staticmethod
    def __validate_integer_field(field_name: str, value: object) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(
                f"{field_name} must be int for TinyRecursiveModelRecurrentConfig, "
                f"got {type(value).__name__}"
            )

    @staticmethod
    def __validate_block_config(block_config: object) -> None:
        from emperor.layers._composition.recurrent.config import (
            RecurrentCompositionConfig,
        )

        if not isinstance(block_config, ConfigBase):
            raise TypeError(
                "block_config must be an instance of ConfigBase for "
                f"TinyRecursiveModelRecurrentConfig, got {type(block_config).__name__}"
            )
        if isinstance(block_config, RecurrentCompositionConfig):
            raise ValueError(
                "TinyRecursiveModelRecurrentConfig.block_config cannot contain another recurrent "
                "composition."
            )
        field_names = {field.name for field in fields(block_config)}
        missing_fields = {"input_dim", "output_dim"} - field_names
        if missing_fields:
            missing_field_list = ", ".join(sorted(missing_fields))
            raise TypeError(
                "block_config must declare dataclass fields input_dim and output_dim "
                "for TinyRecursiveModelRecurrentConfig; "
                f"{type(block_config).__name__} is missing {missing_field_list}"
            )

    @classmethod
    def validate_state(cls, state: object, expected_feature_dim: int) -> None:
        _validate_variant_state(
            state,
            expected_feature_dim,
            owner_name="TinyRecursiveModelRecurrent",
        )

    @staticmethod
    def validate_hidden(
        hidden: object,
        expected_feature_dim: int,
        *,
        field_name: str,
    ) -> None:
        _validate_variant_hidden(
            hidden,
            expected_feature_dim,
            field_name=field_name,
            owner_name="TinyRecursiveModelRecurrent",
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

    @classmethod
    def validate_initial_buffers(
        cls,
        hidden: Tensor,
        *,
        answer_initial: Tensor,
        latent_initial: Tensor,
        expected_feature_dim: int,
    ) -> None:
        for buffer_name, initial_buffer in (
            ("answer_initial", answer_initial),
            ("latent_initial", latent_initial),
        ):
            cls.validate_initial_buffer(
                initial_buffer,
                hidden,
                name=buffer_name,
                expected_feature_dim=expected_feature_dim,
            )

    @classmethod
    def validate_transition_output(
        cls,
        output_state: object,
        transition_input: Tensor,
        expected_row_layout: object,
        *,
        expected_feature_dim: int,
    ) -> None:
        _validate_variant_transition_output(
            output_state,
            transition_input,
            expected_row_layout,
            expected_feature_dim=expected_feature_dim,
            owner_name="TinyRecursiveModelRecurrent",
            transition_name="Tiny Recursive Model",
        )


class HierarchicalReasoningModelRecurrentValidator(
    TinyRecursiveModelRecurrentValidator
):
    @classmethod
    def validate(cls, model: object) -> None:
        from emperor.layers._composition.recurrent.config import (
            HierarchicalReasoningModelRecurrentConfig,
            RecurrentCompositionConfig,
        )

        config = model.cfg
        if not isinstance(config, HierarchicalReasoningModelRecurrentConfig):
            raise TypeError(
                "HierarchicalReasoningModelRecurrent cfg must be a "
                "HierarchicalReasoningModelRecurrentConfig, "
                f"got {type(config).__name__}."
            )
        cls.validate_required_fields(config)
        for field_name in ("input_dim", "output_dim", "high_cycles", "low_cycles"):
            value = getattr(config, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"{field_name} must be int for HierarchicalReasoningModelRecurrentConfig, "
                    f"got {type(value).__name__}"
                )
        cls.validate_dimensions(
            input_dim=config.input_dim,
            output_dim=config.output_dim,
            high_cycles=config.high_cycles,
            low_cycles=config.low_cycles,
        )
        _validate_transition_gradient_window(
            config,
            total_transition_count=(config.high_cycles * (config.low_cycles + 1)),
        )
        if config.input_dim != config.output_dim:
            raise ValueError(
                "input_dim and output_dim must be equal for HierarchicalReasoningModelRecurrentConfig, "
                f"got input_dim={config.input_dim} and "
                f"output_dim={config.output_dim}."
            )
        _validate_recurrent_controller_config(config)
        for field_name in ("high_block_config", "low_block_config"):
            block_config = getattr(config, field_name)
            if not isinstance(block_config, ConfigBase):
                raise TypeError(
                    f"{field_name} must be an instance of ConfigBase for "
                    f"HierarchicalReasoningModelRecurrentConfig, got {type(block_config).__name__}"
                )
            if isinstance(block_config, RecurrentCompositionConfig):
                raise ValueError(
                    f"{field_name} cannot contain another recurrent composition."
                )
            block_fields = {field.name for field in fields(block_config)}
            if not {"input_dim", "output_dim"}.issubset(block_fields):
                raise TypeError(
                    f"{field_name} must declare dataclass fields input_dim and "
                    "output_dim for HierarchicalReasoningModelRecurrentConfig."
                )
        _validate_initialization_standard_deviation(
            config.initialization_standard_deviation
        )
        expected_owner = config.registry_owner()
        if not isinstance(model, expected_owner):
            raise TypeError(
                f"{type(config).__name__} builds {expected_owner.__name__}, not "
                f"{type(model).__name__}."
            )

    @classmethod
    def validate_state(cls, state: object, expected_feature_dim: int) -> None:
        _validate_variant_state(
            state,
            expected_feature_dim,
            owner_name="HierarchicalReasoningModelRecurrent",
        )

    @staticmethod
    def validate_hidden(
        hidden: object,
        expected_feature_dim: int,
        *,
        field_name: str,
    ) -> None:
        _validate_variant_hidden(
            hidden,
            expected_feature_dim,
            field_name=field_name,
            owner_name="HierarchicalReasoningModelRecurrent",
        )

    @classmethod
    def validate_initial_buffers(
        cls,
        hidden: Tensor,
        *,
        high_initial: Tensor,
        low_initial: Tensor,
        expected_feature_dim: int,
    ) -> None:
        for buffer_name, initial_buffer in (
            ("high_initial", high_initial),
            ("low_initial", low_initial),
        ):
            cls.validate_initial_buffer(
                initial_buffer,
                hidden,
                name=buffer_name,
                expected_feature_dim=expected_feature_dim,
            )

    @classmethod
    def validate_transition_output(
        cls,
        output_state: object,
        transition_input: Tensor,
        expected_row_layout: object,
        *,
        expected_feature_dim: int,
    ) -> None:
        _validate_variant_transition_output(
            output_state,
            transition_input,
            expected_row_layout,
            expected_feature_dim=expected_feature_dim,
            owner_name="HierarchicalReasoningModelRecurrent",
            transition_name="Hierarchical Reasoning Model",
        )

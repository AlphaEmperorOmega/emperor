from __future__ import annotations

from dataclasses import fields
from typing import TYPE_CHECKING

from torch import Tensor

from emperor.config import ConfigBase
from emperor.layers._composition.recurrent.validation.common import (
    _GRADIENT_WINDOW_FIELDS,
    _RECURRENT_CONTROLLER_OPTIONAL_FIELDS,
    _RecurrentCompositionValidator,
    _validate_recurrent_controller_config,
    _validate_transition_gradient_window,
)
from emperor.layers._composition.residual.base import ResidualRuntimeRequirement

if TYPE_CHECKING:
    from emperor.layers._composition.recurrent.variants.standard import RecurrentLayer
    from emperor.layers._state import LayerState


_SUPPORTED_RESIDUAL_REQUIREMENTS = frozenset(
    {
        ResidualRuntimeRequirement.FORWARD_LOCAL_STATE,
        ResidualRuntimeRequirement.DEPTH_SPECIFIC_CONNECTIONS,
    }
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
        _validate_recurrent_controller_config(
            cfg,
            supported_residual_requirements=_SUPPORTED_RESIDUAL_REQUIREMENTS,
        )
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

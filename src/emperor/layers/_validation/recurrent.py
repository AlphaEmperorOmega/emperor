from __future__ import annotations

from dataclasses import fields
from typing import TYPE_CHECKING

from emperor._validation import ValidatorBase
from emperor.config import ConfigBase
from emperor.layers._options import LayerNormPositionOptions
from emperor.layers._validation.common import (
    _HALTING_CONFIG_FIELDS,
    _MEMORY_CONFIG_FIELDS,
    _matches_config_contract,
    _validate_halting_lifecycle_owner,
)
from emperor.layers._validation.gate import LayerGateValidator
from emperor.layers._validation.residual import ResidualConnectionValidator

if TYPE_CHECKING:
    from torch import Tensor

    from emperor.halting import HaltingConfig
    from emperor.layers._config import GateConfig, ResidualConfig
    from emperor.layers._recurrent import RecurrentLayer
    from emperor.layers._state import LayerState


class RecurrentLayerValidator(ValidatorBase):
    GATE_VALIDATOR = LayerGateValidator
    RESIDUAL_VALIDATOR = ResidualConnectionValidator

    OPTIONAL_FIELDS = {
        "gate_config",
        "halting_config",
        "memory_config",
        "residual_config",
        "override_config",
    }

    @classmethod
    def validate(cls, model: RecurrentLayer) -> None:
        cfg = model.cfg
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
        cls._validate_stable_dimensions(
            cfg.input_dim,
            cfg.output_dim,
        )
        cls._validate_recurrent_layer_norm_position(cfg.recurrent_layer_norm_position)
        cls._validate_block_config(cfg.block_config)
        cls._validate_residual_config(cfg.residual_config)
        cls._validate_gate_config(cfg.gate_config)
        cls._validate_halting_config(cfg.halting_config)
        cls._validate_memory_config(cfg.memory_config)

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
    def _validate_integer_field(field_name: str, value: int) -> None:
        if not isinstance(value, int):
            raise TypeError(
                f"{field_name} must be int for RecurrentLayerConfig, "
                f"got {type(value).__name__}"
            )

    @staticmethod
    def _validate_stable_dimensions(input_dim: int, output_dim: int) -> None:
        if input_dim != output_dim:
            raise ValueError(
                f"input_dim and output_dim must be equal for RecurrentLayerConfig, "
                f"got input_dim={input_dim} and output_dim={output_dim}."
            )

    @staticmethod
    def _validate_block_config(block_config: ConfigBase) -> None:
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

    @staticmethod
    def _validate_recurrent_layer_norm_position(
        recurrent_layer_norm_position: LayerNormPositionOptions | None,
    ) -> None:
        if not isinstance(recurrent_layer_norm_position, LayerNormPositionOptions):
            raise TypeError(
                "recurrent_layer_norm_position must be a "
                "LayerNormPositionOptions value for RecurrentLayerConfig, "
                f"got {type(recurrent_layer_norm_position).__name__}"
            )

    @classmethod
    def _validate_gate_config(cls, gate_config: GateConfig | None) -> None:
        cls.GATE_VALIDATOR.validate_recurrent_gate_config(
            gate_config,
            owner_name="RecurrentLayerConfig.gate_config",
        )

    @classmethod
    def _validate_residual_config(
        cls,
        residual_config: ResidualConfig | None,
    ) -> None:
        cls.RESIDUAL_VALIDATOR.validate_residual_config(
            residual_config,
            owner_name="RecurrentLayerConfig",
        )

    @staticmethod
    def _validate_halting_config(
        halting_config: HaltingConfig | None,
    ) -> None:
        if halting_config is None:
            return
        if not _matches_config_contract(
            halting_config,
            _HALTING_CONFIG_FIELDS,
        ):
            raise TypeError(
                f"halting_config must be an instance of HaltingConfig for "
                f"RecurrentLayerConfig, got {type(halting_config).__name__}"
            )

        _validate_halting_lifecycle_owner(
            halting_config,
            field_name="halting_config",
            owner_name="RecurrentLayerConfig",
        )

    @staticmethod
    def _validate_memory_config(memory_config) -> None:
        if memory_config is None:
            return
        if not _matches_config_contract(memory_config, _MEMORY_CONFIG_FIELDS):
            raise TypeError(
                f"memory_config must be an instance of DynamicMemoryConfig for "
                f"RecurrentLayerConfig, got {type(memory_config).__name__}"
            )

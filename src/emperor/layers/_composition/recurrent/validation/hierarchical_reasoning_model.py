from __future__ import annotations

from dataclasses import fields

from torch import Tensor

from emperor.config import ConfigBase
from emperor.layers._composition.recurrent.validation.common import (
    _GRADIENT_WINDOW_FIELDS,
    _RECURRENT_CONTROLLER_OPTIONAL_FIELDS,
    _RecurrentCompositionValidator,
    _validate_initialization_standard_deviation,
    _validate_recurrent_controller_config,
    _validate_transition_gradient_window,
    _validate_variant_hidden,
    _validate_variant_state,
    _validate_variant_transition_output,
)


class HierarchicalReasoningModelRecurrentValidator(_RecurrentCompositionValidator):
    OPTIONAL_FIELDS = {
        *_RECURRENT_CONTROLLER_OPTIONAL_FIELDS,
        *_GRADIENT_WINDOW_FIELDS,
    }

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

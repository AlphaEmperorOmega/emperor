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

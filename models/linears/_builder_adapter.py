from __future__ import annotations

from dataclasses import replace
from importlib import import_module
from types import ModuleType
from typing import Any

from models.linears import _builder_options as linears_options
from models.linears._controller_stack import (
    SubmoduleStackOptions,
    SubmoduleStackSource,
)


_TOP_LEVEL_KEYS = ("batch_size", "learning_rate", "input_dim", "output_dim")

_CONTROLLER_STACK_FIELD_MAP = {
    "independent_flag": "independent_flag",
    "hidden_dim": "hidden_dim",
    "num_layers": "num_layers",
    "last_layer_bias_option": "last_layer_bias_option",
    "apply_output_pipeline_flag": "apply_output_pipeline_flag",
    "activation": "activation",
    "layer_norm_position": "layer_norm_position",
    "residual_connection_option": "residual_connection_option",
    "dropout_probability": "dropout_probability",
    "bias_flag": "bias_flag",
}

_ADAPTIVE_GENERATOR_SOURCE_FIELD_MAP = {
    "independent_flag": "independent_flag",
    "hidden_dim": "hidden_dim",
    "layer_norm_position": "layer_norm_position",
    "num_layers": "num_layers",
    "activation": "activation",
    "residual_connection_option": "residual_connection_option",
    "dropout_probability": "dropout_probability",
    "last_layer_bias_option": "last_layer_bias_option",
    "apply_output_pipeline_flag": "apply_output_pipeline_flag",
    "bias_flag": "bias_flag",
}

_HIDDEN_ADAPTIVE_WEIGHT_FIELD_MAP = {
    "generator_depth": "generator_depth",
    "weight_option_flag": "option_flag",
    "weight_option": "option",
    "weight_normalization_option": "normalization_option",
    "weight_normalization_position_option": "normalization_position_option",
    "weight_decay_schedule": "decay_schedule",
    "weight_decay_rate": "decay_rate",
    "weight_decay_warmup_batches": "decay_warmup_batches",
    "weight_bank_expansion_factor": "bank_expansion_factor",
}

_HIDDEN_ADAPTIVE_BIAS_FIELD_MAP = {
    "bias_option_flag": "option_flag",
    "bias_option": "option",
    "bias_decay_schedule": "decay_schedule",
    "bias_decay_rate": "decay_rate",
    "bias_decay_warmup_batches": "decay_warmup_batches",
    "bias_bank_expansion_factor": "bank_expansion_factor",
}

_HIDDEN_ADAPTIVE_DIAGONAL_FIELD_MAP = {
    "diagonal_option_flag": "option_flag",
    "diagonal_option": "option",
}

_HIDDEN_ADAPTIVE_MASK_FIELD_MAP = {
    "mask_option_flag": "option_flag",
    "row_mask_option": "row_mask_option",
    "mask_dimension_option": "mask_dimension_option",
    "mask_threshold": "mask_threshold",
    "mask_surrogate_scale": "mask_surrogate_scale",
    "mask_floor": "mask_floor",
    "mask_transition_width": "mask_transition_width",
}


def linear_builder_kwargs_from_flat(
    flat_kwargs: dict[str, Any],
    config_module: ModuleType,
) -> dict[str, Any]:
    kwargs = dict(flat_kwargs)
    builder_kwargs = _pop_top_level_kwargs(kwargs)
    builder_kwargs.update(
        {
            "stack_options": _linear_stack_options_from_kwargs(
                kwargs,
                config_module,
                provided=kwargs.pop("stack_options", None),
            ),
            "submodule_stack_options": _submodule_stack_options_from_kwargs(
                kwargs,
                config_module,
                provided=kwargs.pop("submodule_stack_options", None),
            ),
            "layer_controller_options": _layer_controller_options_from_kwargs(
                kwargs,
                config_module,
                provided=kwargs.pop("layer_controller_options", None),
            ),
            "dynamic_memory_options": _dynamic_memory_options_from_kwargs(
                kwargs,
                config_module,
                provided=kwargs.pop("dynamic_memory_options", None),
            ),
            "recurrent_controller_options": _recurrent_controller_options_from_kwargs(
                kwargs,
                config_module,
                provided=kwargs.pop("recurrent_controller_options", None),
            ),
        }
    )
    builder_kwargs.update(kwargs)
    return builder_kwargs


def linear_adaptive_builder_kwargs_from_flat(
    flat_kwargs: dict[str, Any],
    config_module: ModuleType,
) -> dict[str, Any]:
    kwargs = dict(flat_kwargs)
    builder_kwargs = linear_builder_kwargs_from_flat(kwargs, config_module)
    leftovers = {
        key: builder_kwargs.pop(key)
        for key in list(builder_kwargs)
        if key not in _TOP_LEVEL_KEYS
        and key
        not in {
            "stack_options",
            "submodule_stack_options",
            "layer_controller_options",
            "dynamic_memory_options",
            "recurrent_controller_options",
        }
    }
    builder_kwargs.update(
        {
            "adaptive_generator_stack_options": (
                _adaptive_generator_stack_options_from_kwargs(
                    leftovers,
                    config_module,
                    provided=leftovers.pop("adaptive_generator_stack_options", None),
                )
            ),
            "hidden_adaptive_weight_options": (
                _hidden_adaptive_weight_options_from_kwargs(
                    leftovers,
                    config_module,
                    provided=leftovers.pop("hidden_adaptive_weight_options", None),
                )
            ),
            "hidden_adaptive_bias_options": (
                _hidden_adaptive_bias_options_from_kwargs(
                    leftovers,
                    config_module,
                    provided=leftovers.pop("hidden_adaptive_bias_options", None),
                )
            ),
            "hidden_adaptive_diagonal_options": (
                _hidden_adaptive_diagonal_options_from_kwargs(
                    leftovers,
                    config_module,
                    provided=leftovers.pop("hidden_adaptive_diagonal_options", None),
                )
            ),
            "hidden_adaptive_mask_options": (
                _hidden_adaptive_mask_options_from_kwargs(
                    leftovers,
                    config_module,
                    provided=leftovers.pop("hidden_adaptive_mask_options", None),
                )
            ),
            "input_boundary_options": _boundary_options_from_kwargs(
                leftovers,
                config_module,
                "input_layer",
                provided=leftovers.pop("input_boundary_options", None),
            ),
            "output_boundary_options": _boundary_options_from_kwargs(
                leftovers,
                config_module,
                "output_layer",
                provided=leftovers.pop("output_boundary_options", None),
            ),
        }
    )
    builder_kwargs.update(leftovers)
    return builder_kwargs


def linear_flat_defaults(config_module: ModuleType) -> dict[str, Any]:
    return {
        "batch_size": config_module.BATCH_SIZE,
        "learning_rate": config_module.LEARNING_RATE,
        "input_dim": config_module.INPUT_DIM,
        "output_dim": config_module.OUTPUT_DIM,
        "stack_hidden_dim": config_module.STACK_HIDDEN_DIM,
        "stack_bias_flag": config_module.STACK_BIAS_FLAG,
        "stack_layer_norm_position": config_module.STACK_LAYER_NORM_POSITION,
        "stack_num_layers": config_module.STACK_NUM_LAYERS,
        "stack_activation": config_module.STACK_ACTIVATION,
        "stack_residual_connection_option": (
            config_module.STACK_RESIDUAL_CONNECTION_OPTION
        ),
        "stack_dropout_probability": config_module.STACK_DROPOUT_PROBABILITY,
        "stack_last_layer_bias_option": config_module.STACK_LAST_LAYER_BIAS_OPTION,
        "stack_apply_output_pipeline_flag": (
            config_module.STACK_APPLY_OUTPUT_PIPELINE_FLAG
        ),
        "submodule_stack_hidden_dim": config_module.SUBMODULE_STACK_HIDDEN_DIM,
        "submodule_stack_num_layers": config_module.SUBMODULE_STACK_NUM_LAYERS,
        "submodule_stack_last_layer_bias_option": (
            config_module.SUBMODULE_STACK_LAST_LAYER_BIAS_OPTION
        ),
        "submodule_stack_apply_output_pipeline_flag": (
            config_module.SUBMODULE_STACK_APPLY_OUTPUT_PIPELINE_FLAG
        ),
        "submodule_stack_activation": config_module.SUBMODULE_STACK_ACTIVATION,
        "submodule_stack_layer_norm_position": (
            config_module.SUBMODULE_STACK_LAYER_NORM_POSITION
        ),
        "submodule_stack_residual_connection_option": (
            config_module.SUBMODULE_STACK_RESIDUAL_CONNECTION_OPTION
        ),
        "submodule_stack_dropout_probability": (
            config_module.SUBMODULE_STACK_DROPOUT_PROBABILITY
        ),
        "submodule_stack_bias_flag": config_module.SUBMODULE_STACK_BIAS_FLAG,
        "stack_gate_flag": config_module.GATE_FLAG,
        "gate_option": config_module.GATE_OPTION,
        "gate_activation": config_module.GATE_ACTIVATION,
        **_controller_stack_flat_defaults(config_module, "gate_stack"),
        "stack_halting_flag": config_module.HALTING_FLAG,
        "halting_threshold": config_module.HALTING_THRESHOLD,
        "halting_dropout": config_module.HALTING_DROPOUT,
        "halting_hidden_state_mode": config_module.HALTING_HIDDEN_STATE_MODE,
        **_controller_stack_flat_defaults(config_module, "halting_stack"),
        "memory_flag": config_module.MEMORY_FLAG,
        "memory_option": config_module.MEMORY_OPTION,
        "memory_position_option": config_module.MEMORY_POSITION_OPTION,
        "memory_test_time_training_learning_rate": (
            config_module.MEMORY_TEST_TIME_TRAINING_LEARNING_RATE
        ),
        "memory_test_time_training_num_inner_steps": (
            config_module.MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS
        ),
        **_controller_stack_flat_defaults(config_module, "memory_stack"),
        "recurrent_flag": config_module.RECURRENT_FLAG,
        "recurrent_max_steps": config_module.RECURRENT_MAX_STEPS,
        "recurrent_layer_norm_position": (
            config_module.RECURRENT_LAYER_NORM_POSITION
        ),
        "recurrent_gate_flag": config_module.RECURRENT_GATE_FLAG,
        "recurrent_gate_option": config_module.RECURRENT_GATE_OPTION,
        "recurrent_gate_activation": config_module.RECURRENT_GATE_ACTIVATION,
        **_controller_stack_flat_defaults(config_module, "recurrent_gate_stack"),
        "recurrent_halting_flag": config_module.RECURRENT_HALTING_FLAG,
        "recurrent_halting_threshold": config_module.RECURRENT_HALTING_THRESHOLD,
        "recurrent_halting_dropout": config_module.RECURRENT_HALTING_DROPOUT,
        "recurrent_halting_hidden_state_mode": (
            config_module.RECURRENT_HALTING_HIDDEN_STATE_MODE
        ),
        **_controller_stack_flat_defaults(
            config_module,
            "recurrent_halting_stack",
        ),
    }


def _default_linear_stack_options(
    config_module: ModuleType,
) -> linears_options.MainLayerStackOptions:
    return linears_options.MainLayerStackOptions(
        hidden_dim=config_module.STACK_HIDDEN_DIM,
        bias_flag=config_module.STACK_BIAS_FLAG,
        layer_norm_position=config_module.STACK_LAYER_NORM_POSITION,
        num_layers=config_module.STACK_NUM_LAYERS,
        activation=config_module.STACK_ACTIVATION,
        residual_connection_option=config_module.STACK_RESIDUAL_CONNECTION_OPTION,
        dropout_probability=config_module.STACK_DROPOUT_PROBABILITY,
        last_layer_bias_option=config_module.STACK_LAST_LAYER_BIAS_OPTION,
        apply_output_pipeline_flag=config_module.STACK_APPLY_OUTPUT_PIPELINE_FLAG,
    )


def _default_submodule_stack_options(
    config_module: ModuleType,
) -> SubmoduleStackOptions:
    return SubmoduleStackOptions(
        hidden_dim=config_module.SUBMODULE_STACK_HIDDEN_DIM,
        num_layers=config_module.SUBMODULE_STACK_NUM_LAYERS,
        last_layer_bias_option=config_module.SUBMODULE_STACK_LAST_LAYER_BIAS_OPTION,
        apply_output_pipeline_flag=(
            config_module.SUBMODULE_STACK_APPLY_OUTPUT_PIPELINE_FLAG
        ),
        activation=config_module.SUBMODULE_STACK_ACTIVATION,
        layer_norm_position=config_module.SUBMODULE_STACK_LAYER_NORM_POSITION,
        residual_connection_option=(
            config_module.SUBMODULE_STACK_RESIDUAL_CONNECTION_OPTION
        ),
        dropout_probability=config_module.SUBMODULE_STACK_DROPOUT_PROBABILITY,
        bias_flag=config_module.SUBMODULE_STACK_BIAS_FLAG,
    )


def _default_controller_stack_source(
    config_module: ModuleType,
    prefix: str,
) -> SubmoduleStackSource:
    config_prefix = prefix.upper()
    return SubmoduleStackSource(
        independent_flag=getattr(config_module, f"{config_prefix}_INDEPENDENT_FLAG"),
        hidden_dim=getattr(config_module, f"{config_prefix}_HIDDEN_DIM"),
        num_layers=getattr(config_module, f"{config_prefix}_NUM_LAYERS"),
        last_layer_bias_option=getattr(
            config_module,
            f"{config_prefix}_LAST_LAYER_BIAS_OPTION",
        ),
        apply_output_pipeline_flag=getattr(
            config_module,
            f"{config_prefix}_APPLY_OUTPUT_PIPELINE_FLAG",
        ),
        activation=getattr(config_module, f"{config_prefix}_ACTIVATION"),
        layer_norm_position=getattr(
            config_module,
            f"{config_prefix}_LAYER_NORM_POSITION",
        ),
        residual_connection_option=getattr(
            config_module,
            f"{config_prefix}_RESIDUAL_CONNECTION_OPTION",
        ),
        dropout_probability=getattr(
            config_module,
            f"{config_prefix}_DROPOUT_PROBABILITY",
        ),
        bias_flag=getattr(config_module, f"{config_prefix}_BIAS_FLAG"),
    )


def _default_layer_controller_options(
    config_module: ModuleType,
) -> linears_options.LayerControllerOptions:
    return linears_options.LayerControllerOptions(
        stack_gate_flag=config_module.GATE_FLAG,
        gate_option=config_module.GATE_OPTION,
        gate_activation=config_module.GATE_ACTIVATION,
        gate_stack_source=_default_controller_stack_source(
            config_module,
            "gate_stack",
        ),
        stack_halting_flag=config_module.HALTING_FLAG,
        halting_threshold=config_module.HALTING_THRESHOLD,
        halting_dropout=config_module.HALTING_DROPOUT,
        halting_hidden_state_mode=config_module.HALTING_HIDDEN_STATE_MODE,
        halting_stack_source=_default_controller_stack_source(
            config_module,
            "halting_stack",
        ),
    )


def _default_dynamic_memory_options(
    config_module: ModuleType,
) -> linears_options.DynamicMemoryOptions:
    return linears_options.DynamicMemoryOptions(
        memory_flag=config_module.MEMORY_FLAG,
        memory_option=config_module.MEMORY_OPTION,
        memory_position_option=config_module.MEMORY_POSITION_OPTION,
        memory_test_time_training_learning_rate=(
            config_module.MEMORY_TEST_TIME_TRAINING_LEARNING_RATE
        ),
        memory_test_time_training_num_inner_steps=(
            config_module.MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS
        ),
        memory_stack_source=_default_controller_stack_source(
            config_module,
            "memory_stack",
        ),
    )


def _default_recurrent_controller_options(
    config_module: ModuleType,
) -> linears_options.RecurrentControllerOptions:
    return linears_options.RecurrentControllerOptions(
        recurrent_flag=config_module.RECURRENT_FLAG,
        recurrent_max_steps=config_module.RECURRENT_MAX_STEPS,
        recurrent_layer_norm_position=config_module.RECURRENT_LAYER_NORM_POSITION,
        recurrent_gate_flag=config_module.RECURRENT_GATE_FLAG,
        recurrent_gate_option=config_module.RECURRENT_GATE_OPTION,
        recurrent_gate_activation=config_module.RECURRENT_GATE_ACTIVATION,
        recurrent_gate_stack_source=_default_controller_stack_source(
            config_module,
            "recurrent_gate_stack",
        ),
        recurrent_halting_flag=config_module.RECURRENT_HALTING_FLAG,
        recurrent_halting_threshold=config_module.RECURRENT_HALTING_THRESHOLD,
        recurrent_halting_dropout=config_module.RECURRENT_HALTING_DROPOUT,
        recurrent_halting_hidden_state_mode=(
            config_module.RECURRENT_HALTING_HIDDEN_STATE_MODE
        ),
        recurrent_halting_stack_source=_default_controller_stack_source(
            config_module,
            "recurrent_halting_stack",
        ),
    )


def _default_adaptive_generator_stack_options(
    config_module: ModuleType,
) -> Any:
    adaptive_options = _adaptive_options()
    return adaptive_options.AdaptiveGeneratorStackOptions(
        hidden_dim=config_module.ADAPTIVE_SUBMODULE_STACK_HIDDEN_DIM,
        layer_norm_position=config_module.ADAPTIVE_SUBMODULE_STACK_LAYER_NORM_POSITION,
        num_layers=config_module.ADAPTIVE_SUBMODULE_STACK_NUM_LAYERS,
        activation=config_module.ADAPTIVE_SUBMODULE_STACK_ACTIVATION,
        residual_connection_option=(
            config_module.ADAPTIVE_SUBMODULE_STACK_RESIDUAL_CONNECTION_OPTION
        ),
        dropout_probability=(
            config_module.ADAPTIVE_SUBMODULE_STACK_DROPOUT_PROBABILITY
        ),
        last_layer_bias_option=(
            config_module.ADAPTIVE_SUBMODULE_STACK_LAST_LAYER_BIAS_OPTION
        ),
        apply_output_pipeline_flag=(
            config_module.ADAPTIVE_SUBMODULE_STACK_APPLY_OUTPUT_PIPELINE_FLAG
        ),
        bias_flag=config_module.ADAPTIVE_SUBMODULE_STACK_BIAS_FLAG,
    )


def _default_adaptive_generator_stack_source(
    config_module: ModuleType,
    prefix: str,
) -> Any:
    adaptive_options = _adaptive_options()
    config_prefix = prefix.upper()
    return adaptive_options.AdaptiveGeneratorStackSource(
        independent_flag=getattr(config_module, f"{config_prefix}_INDEPENDENT_FLAG"),
        hidden_dim=getattr(config_module, f"{config_prefix}_HIDDEN_DIM"),
        layer_norm_position=getattr(
            config_module,
            f"{config_prefix}_LAYER_NORM_POSITION",
        ),
        num_layers=getattr(config_module, f"{config_prefix}_NUM_LAYERS"),
        activation=getattr(config_module, f"{config_prefix}_ACTIVATION"),
        residual_connection_option=getattr(
            config_module,
            f"{config_prefix}_RESIDUAL_CONNECTION_OPTION",
        ),
        dropout_probability=getattr(
            config_module,
            f"{config_prefix}_DROPOUT_PROBABILITY",
        ),
        last_layer_bias_option=getattr(
            config_module,
            f"{config_prefix}_LAST_LAYER_BIAS_OPTION",
        ),
        apply_output_pipeline_flag=getattr(
            config_module,
            f"{config_prefix}_APPLY_OUTPUT_PIPELINE_FLAG",
        ),
        bias_flag=getattr(config_module, f"{config_prefix}_BIAS_FLAG"),
    )


def _default_hidden_adaptive_weight_options(
    config_module: ModuleType,
) -> Any:
    adaptive_options = _adaptive_options()
    return adaptive_options.HiddenAdaptiveWeightOptions(
        generator_depth=config_module.WEIGHT_GENERATOR_DEPTH,
        option_flag=config_module.WEIGHT_OPTION_FLAG,
        option=config_module.WEIGHT_OPTION,
        normalization_option=config_module.WEIGHT_NORMALIZATION_OPTION,
        normalization_position_option=(
            config_module.WEIGHT_NORMALIZATION_POSITION_OPTION
        ),
        decay_schedule=config_module.WEIGHT_DECAY_SCHEDULE,
        decay_rate=config_module.WEIGHT_DECAY_RATE,
        decay_warmup_batches=config_module.WEIGHT_DECAY_WARMUP_BATCHES,
        bank_expansion_factor=config_module.WEIGHT_BANK_EXPANSION_FACTOR,
        generator_stack_source=_default_adaptive_generator_stack_source(
            config_module,
            "weight_generator_stack",
        ),
    )


def _default_hidden_adaptive_bias_options(
    config_module: ModuleType,
) -> Any:
    adaptive_options = _adaptive_options()
    return adaptive_options.HiddenAdaptiveBiasOptions(
        option_flag=config_module.BIAS_OPTION_FLAG,
        option=config_module.BIAS_OPTION,
        decay_schedule=config_module.BIAS_DECAY_SCHEDULE,
        decay_rate=config_module.BIAS_DECAY_RATE,
        decay_warmup_batches=config_module.BIAS_DECAY_WARMUP_BATCHES,
        bank_expansion_factor=config_module.BIAS_BANK_EXPANSION_FACTOR,
        generator_stack_source=_default_adaptive_generator_stack_source(
            config_module,
            "bias_generator_stack",
        ),
    )


def _default_hidden_adaptive_diagonal_options(
    config_module: ModuleType,
) -> Any:
    adaptive_options = _adaptive_options()
    return adaptive_options.HiddenAdaptiveDiagonalOptions(
        option_flag=config_module.DIAGONAL_OPTION_FLAG,
        option=config_module.DIAGONAL_OPTION,
        generator_stack_source=_default_adaptive_generator_stack_source(
            config_module,
            "diagonal_generator_stack",
        ),
    )


def _default_hidden_adaptive_mask_options(
    config_module: ModuleType,
) -> Any:
    adaptive_options = _adaptive_options()
    return adaptive_options.HiddenAdaptiveMaskOptions(
        option_flag=config_module.MASK_OPTION_FLAG,
        row_mask_option=config_module.ROW_MASK_OPTION,
        mask_dimension_option=config_module.MASK_DIMENSION_OPTION,
        mask_threshold=config_module.MASK_THRESHOLD,
        mask_surrogate_scale=config_module.MASK_SURROGATE_SCALE,
        mask_floor=config_module.MASK_FLOOR,
        mask_transition_width=config_module.MASK_TRANSITION_WIDTH,
        generator_stack_source=_default_adaptive_generator_stack_source(
            config_module,
            "mask_generator_stack",
        ),
    )


def _default_boundary_options(
    config_module: ModuleType,
    prefix: str,
) -> Any:
    boundary_options = _boundary_options()
    config_prefix = prefix.upper()
    return boundary_options.AdaptiveBoundaryProjectionOptions(
        weight_option=getattr(config_module, f"{config_prefix}_WEIGHT_OPTION"),
        weight_generator_depth=getattr(
            config_module,
            f"{config_prefix}_WEIGHT_GENERATOR_DEPTH",
        ),
        weight_decay_schedule=getattr(
            config_module,
            f"{config_prefix}_WEIGHT_DECAY_SCHEDULE",
        ),
        weight_decay_rate=getattr(config_module, f"{config_prefix}_WEIGHT_DECAY_RATE"),
        weight_decay_warmup_batches=getattr(
            config_module,
            f"{config_prefix}_WEIGHT_DECAY_WARMUP_BATCHES",
        ),
        weight_normalization_option=getattr(
            config_module,
            f"{config_prefix}_WEIGHT_NORMALIZATION_OPTION",
        ),
        weight_normalization_position_option=getattr(
            config_module,
            f"{config_prefix}_WEIGHT_NORMALIZATION_POSITION_OPTION",
        ),
        weight_bank_expansion_factor=getattr(
            config_module,
            f"{config_prefix}_WEIGHT_BANK_EXPANSION_FACTOR",
        ),
        bias_option=getattr(config_module, f"{config_prefix}_BIAS_OPTION"),
        bias_decay_schedule=getattr(
            config_module,
            f"{config_prefix}_BIAS_DECAY_SCHEDULE",
        ),
        bias_decay_rate=getattr(config_module, f"{config_prefix}_BIAS_DECAY_RATE"),
        bias_decay_warmup_batches=getattr(
            config_module,
            f"{config_prefix}_BIAS_DECAY_WARMUP_BATCHES",
        ),
        bias_bank_expansion_factor=getattr(
            config_module,
            f"{config_prefix}_BIAS_BANK_EXPANSION_FACTOR",
        ),
        diagonal_option=getattr(config_module, f"{config_prefix}_DIAGONAL_OPTION"),
        row_mask_option=getattr(config_module, f"{config_prefix}_ROW_MASK_OPTION"),
        mask_dimension_option=getattr(
            config_module,
            f"{config_prefix}_MASK_DIMENSION_OPTION",
        ),
        mask_threshold=getattr(config_module, f"{config_prefix}_MASK_THRESHOLD"),
        mask_surrogate_scale=getattr(
            config_module,
            f"{config_prefix}_MASK_SURROGATE_SCALE",
        ),
        mask_floor=getattr(config_module, f"{config_prefix}_MASK_FLOOR"),
        mask_transition_width=getattr(
            config_module,
            f"{config_prefix}_MASK_TRANSITION_WIDTH",
        ),
    )


def _pop_top_level_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {key: kwargs.pop(key) for key in _TOP_LEVEL_KEYS if key in kwargs}


def _linear_stack_options_from_kwargs(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    *,
    provided: linears_options.MainLayerStackOptions | None,
) -> linears_options.MainLayerStackOptions:
    options = provided or _default_linear_stack_options(config_module)
    updates = _pop_updates(
        kwargs,
        {
            "stack_hidden_dim": "hidden_dim",
            "stack_bias_flag": "bias_flag",
            "layer_norm_position": "layer_norm_position",
            "stack_layer_norm_position": "layer_norm_position",
            "stack_num_layers": "num_layers",
            "stack_activation": "activation",
            "stack_residual_connection_option": "residual_connection_option",
            "stack_dropout_probability": "dropout_probability",
            "stack_last_layer_bias_option": "last_layer_bias_option",
            "stack_apply_output_pipeline_flag": "apply_output_pipeline_flag",
        },
    )
    return replace(options, **updates) if updates else options


def _submodule_stack_options_from_kwargs(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    *,
    provided: SubmoduleStackOptions | None,
) -> SubmoduleStackOptions:
    options = provided or _default_submodule_stack_options(config_module)
    updates = _pop_updates(
        kwargs,
        {
            "submodule_stack_hidden_dim": "hidden_dim",
            "submodule_stack_num_layers": "num_layers",
            "submodule_stack_last_layer_bias_option": "last_layer_bias_option",
            "submodule_stack_apply_output_pipeline_flag": (
                "apply_output_pipeline_flag"
            ),
            "submodule_stack_activation": "activation",
            "submodule_stack_layer_norm_position": "layer_norm_position",
            "submodule_stack_residual_connection_option": (
                "residual_connection_option"
            ),
            "submodule_stack_dropout_probability": "dropout_probability",
            "submodule_stack_bias_flag": "bias_flag",
        },
    )
    return replace(options, **updates) if updates else options


def _controller_stack_source_from_kwargs(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    prefix: str,
    *,
    provided: SubmoduleStackSource | None = None,
) -> SubmoduleStackSource:
    source = provided or _default_controller_stack_source(config_module, prefix)
    updates = _pop_updates(
        kwargs,
        {
            f"{prefix}_{flat_field}": dataclass_field
            for flat_field, dataclass_field in _CONTROLLER_STACK_FIELD_MAP.items()
        },
    )
    return replace(source, **updates) if updates else source


def _layer_controller_options_from_kwargs(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    *,
    provided: linears_options.LayerControllerOptions | None,
) -> linears_options.LayerControllerOptions:
    options = provided or _default_layer_controller_options(config_module)
    updates = _pop_updates(
        kwargs,
        {
            "stack_gate_flag": "stack_gate_flag",
            "gate_option": "gate_option",
            "gate_activation": "gate_activation",
            "stack_halting_flag": "stack_halting_flag",
            "halting_threshold": "halting_threshold",
            "halting_dropout": "halting_dropout",
            "halting_hidden_state_mode": "halting_hidden_state_mode",
            "shared_gate_config": "shared_gate_config",
        },
    )
    updates["gate_stack_source"] = _controller_stack_source_from_kwargs(
        kwargs,
        config_module,
        "gate_stack",
        provided=options.gate_stack_source,
    )
    updates["halting_stack_source"] = _controller_stack_source_from_kwargs(
        kwargs,
        config_module,
        "halting_stack",
        provided=options.halting_stack_source,
    )
    return replace(options, **updates)


def _dynamic_memory_options_from_kwargs(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    *,
    provided: linears_options.DynamicMemoryOptions | None,
) -> linears_options.DynamicMemoryOptions:
    options = provided or _default_dynamic_memory_options(config_module)
    updates = _pop_updates(
        kwargs,
        {
            "memory_flag": "memory_flag",
            "memory_option": "memory_option",
            "memory_position_option": "memory_position_option",
            "memory_test_time_training_learning_rate": (
                "memory_test_time_training_learning_rate"
            ),
            "memory_test_time_training_num_inner_steps": (
                "memory_test_time_training_num_inner_steps"
            ),
        },
    )
    updates["memory_stack_source"] = _controller_stack_source_from_kwargs(
        kwargs,
        config_module,
        "memory_stack",
        provided=options.memory_stack_source,
    )
    return replace(options, **updates)


def _recurrent_controller_options_from_kwargs(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    *,
    provided: linears_options.RecurrentControllerOptions | None,
) -> linears_options.RecurrentControllerOptions:
    options = provided or _default_recurrent_controller_options(config_module)
    updates = _pop_updates(
        kwargs,
        {
            "recurrent_flag": "recurrent_flag",
            "recurrent_max_steps": "recurrent_max_steps",
            "recurrent_layer_norm_position": "recurrent_layer_norm_position",
            "recurrent_gate_flag": "recurrent_gate_flag",
            "recurrent_gate_option": "recurrent_gate_option",
            "recurrent_gate_activation": "recurrent_gate_activation",
            "recurrent_halting_flag": "recurrent_halting_flag",
            "recurrent_halting_threshold": "recurrent_halting_threshold",
            "recurrent_halting_dropout": "recurrent_halting_dropout",
            "recurrent_halting_hidden_state_mode": (
                "recurrent_halting_hidden_state_mode"
            ),
        },
    )
    updates["recurrent_gate_stack_source"] = _controller_stack_source_from_kwargs(
        kwargs,
        config_module,
        "recurrent_gate_stack",
        provided=options.recurrent_gate_stack_source,
    )
    updates["recurrent_halting_stack_source"] = (
        _controller_stack_source_from_kwargs(
            kwargs,
            config_module,
            "recurrent_halting_stack",
            provided=options.recurrent_halting_stack_source,
        )
    )
    return replace(options, **updates)


def _adaptive_generator_stack_options_from_kwargs(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    *,
    provided: Any,
) -> Any:
    options = provided or _default_adaptive_generator_stack_options(config_module)
    updates = _pop_updates(
        kwargs,
        {
            "adaptive_generator_stack_hidden_dim": "hidden_dim",
            "adaptive_generator_stack_layer_norm_position": (
                "layer_norm_position"
            ),
            "adaptive_generator_stack_num_layers": "num_layers",
            "adaptive_generator_stack_activation": "activation",
            "adaptive_generator_stack_residual_connection_option": (
                "residual_connection_option"
            ),
            "adaptive_generator_stack_dropout_probability": (
                "dropout_probability"
            ),
            "adaptive_generator_stack_last_layer_bias_option": (
                "last_layer_bias_option"
            ),
            "adaptive_generator_stack_apply_output_pipeline_flag": (
                "apply_output_pipeline_flag"
            ),
            "adaptive_generator_stack_bias_flag": "bias_flag",
        },
    )
    return replace(options, **updates) if updates else options


def _adaptive_generator_stack_source_from_kwargs(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    prefix: str,
    *,
    provided: Any,
) -> Any:
    source = provided or _default_adaptive_generator_stack_source(
        config_module,
        prefix,
    )
    updates = _pop_updates(
        kwargs,
        {
            f"{prefix}_{flat_field}": dataclass_field
            for flat_field, dataclass_field in (
                _ADAPTIVE_GENERATOR_SOURCE_FIELD_MAP.items()
            )
        },
    )
    return replace(source, **updates) if updates else source


def _hidden_adaptive_weight_options_from_kwargs(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    *,
    provided: Any,
) -> Any:
    options = provided or _default_hidden_adaptive_weight_options(config_module)
    updates = _pop_updates(kwargs, _HIDDEN_ADAPTIVE_WEIGHT_FIELD_MAP)
    updates["generator_stack_source"] = _adaptive_generator_stack_source_from_kwargs(
        kwargs,
        config_module,
        "weight_generator_stack",
        provided=kwargs.pop(
            "weight_generator_stack_source",
            options.generator_stack_source,
        ),
    )
    return replace(options, **updates)


def _hidden_adaptive_bias_options_from_kwargs(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    *,
    provided: Any,
) -> Any:
    options = provided or _default_hidden_adaptive_bias_options(config_module)
    updates = _pop_updates(kwargs, _HIDDEN_ADAPTIVE_BIAS_FIELD_MAP)
    updates["generator_stack_source"] = _adaptive_generator_stack_source_from_kwargs(
        kwargs,
        config_module,
        "bias_generator_stack",
        provided=kwargs.pop(
            "bias_generator_stack_source",
            options.generator_stack_source,
        ),
    )
    return replace(options, **updates)


def _hidden_adaptive_diagonal_options_from_kwargs(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    *,
    provided: Any,
) -> Any:
    options = provided or _default_hidden_adaptive_diagonal_options(config_module)
    updates = _pop_updates(kwargs, _HIDDEN_ADAPTIVE_DIAGONAL_FIELD_MAP)
    updates["generator_stack_source"] = _adaptive_generator_stack_source_from_kwargs(
        kwargs,
        config_module,
        "diagonal_generator_stack",
        provided=kwargs.pop(
            "diagonal_generator_stack_source",
            options.generator_stack_source,
        ),
    )
    return replace(options, **updates)


def _hidden_adaptive_mask_options_from_kwargs(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    *,
    provided: Any,
) -> Any:
    options = provided or _default_hidden_adaptive_mask_options(config_module)
    updates = _pop_updates(kwargs, _HIDDEN_ADAPTIVE_MASK_FIELD_MAP)
    updates["generator_stack_source"] = _adaptive_generator_stack_source_from_kwargs(
        kwargs,
        config_module,
        "mask_generator_stack",
        provided=kwargs.pop(
            "mask_generator_stack_source",
            options.generator_stack_source,
        ),
    )
    return replace(options, **updates)


def _boundary_options_from_kwargs(
    kwargs: dict[str, Any],
    config_module: ModuleType,
    prefix: str,
    *,
    provided: Any,
) -> Any:
    options = provided or _default_boundary_options(config_module, prefix)
    updates = _pop_updates(
        kwargs,
        {
            f"{prefix}_weight_option": "weight_option",
            f"{prefix}_weight_generator_depth": "weight_generator_depth",
            f"{prefix}_weight_decay_schedule": "weight_decay_schedule",
            f"{prefix}_weight_decay_rate": "weight_decay_rate",
            f"{prefix}_weight_decay_warmup_batches": (
                "weight_decay_warmup_batches"
            ),
            f"{prefix}_weight_normalization_option": "weight_normalization_option",
            f"{prefix}_weight_normalization_position_option": (
                "weight_normalization_position_option"
            ),
            f"{prefix}_weight_bank_expansion_factor": (
                "weight_bank_expansion_factor"
            ),
            f"{prefix}_bias_option": "bias_option",
            f"{prefix}_bias_decay_schedule": "bias_decay_schedule",
            f"{prefix}_bias_decay_rate": "bias_decay_rate",
            f"{prefix}_bias_decay_warmup_batches": "bias_decay_warmup_batches",
            f"{prefix}_bias_bank_expansion_factor": "bias_bank_expansion_factor",
            f"{prefix}_diagonal_option": "diagonal_option",
            f"{prefix}_row_mask_option": "row_mask_option",
            f"{prefix}_mask_dimension_option": "mask_dimension_option",
            f"{prefix}_mask_threshold": "mask_threshold",
            f"{prefix}_mask_surrogate_scale": "mask_surrogate_scale",
            f"{prefix}_mask_floor": "mask_floor",
            f"{prefix}_mask_transition_width": "mask_transition_width",
        },
    )
    return replace(options, **updates) if updates else options


def _pop_updates(
    kwargs: dict[str, Any],
    mapping: dict[str, str],
) -> dict[str, Any]:
    updates: dict[str, Any] = {}
    for flat_key, option_field in mapping.items():
        if flat_key in kwargs:
            updates[option_field] = kwargs.pop(flat_key)
    return updates


def _controller_stack_flat_defaults(
    config_module: ModuleType,
    prefix: str,
) -> dict[str, Any]:
    source = _default_controller_stack_source(config_module, prefix)
    return {
        f"{prefix}_{field}": getattr(source, dataclass_field)
        for field, dataclass_field in _CONTROLLER_STACK_FIELD_MAP.items()
    }


def _adaptive_options():
    return import_module("models.linears.linear_adaptive._builder_options")


def _boundary_options():
    return import_module("models.linears.linear_adaptive._boundary_config_factory")

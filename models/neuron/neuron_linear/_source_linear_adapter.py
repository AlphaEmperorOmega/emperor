from typing import Any

import models.linears.linear.config as source_config
from models.linears._builder_adapter import (
    linear_builder_kwargs_from_flat,
    linear_flat_defaults,
)
from models.linears.linear.presets import (
    ExperimentPreset as SourceExperimentPreset,
)
from models.linears.linear.presets import (
    ExperimentPresets as SourceExperimentPresets,
)

SOURCE_LINEAR_KWARG_ALIASES = {
    "gate_hidden_dim": "gate_stack_hidden_dim",
    "gate_layer_norm_position": "gate_stack_layer_norm_position",
    "gate_bias_flag": "gate_stack_bias_flag",
    "halting_hidden_dim": "halting_stack_hidden_dim",
    "halting_layer_norm_position": "halting_stack_layer_norm_position",
    "halting_bias_flag": "halting_stack_bias_flag",
}


def canonical_source_kwarg_aliases() -> dict[str, str]:
    return dict(SOURCE_LINEAR_KWARG_ALIASES)


def normalize_source_kwargs(source_kwargs: dict[str, Any]) -> dict[str, Any]:
    return {
        SOURCE_LINEAR_KWARG_ALIASES.get(key, key): value
        for key, value in source_kwargs.items()
    }


def source_builder_kwargs_from_flat(source_kwargs: dict[str, Any]) -> dict[str, Any]:
    return linear_builder_kwargs_from_flat(
        normalize_source_kwargs(source_kwargs),
        source_config,
    )


def source_linear_default_kwargs() -> dict[str, Any]:
    return linear_flat_defaults(source_config)


def source_preset(preset) -> SourceExperimentPreset:
    return SourceExperimentPreset[preset.name]


def source_values_for_preset(preset) -> dict[str, object]:
    return SourceExperimentPresets().overrides_for_preset(source_preset(preset))


def source_locks_for_preset(preset) -> dict[str, object]:
    return dict(SourceExperimentPresets().locks_for_preset(source_preset(preset)))

from typing import Any

import models.linears.linear.config as source_config
from models.linears._builder_adapter import (
    linear_builder_kwargs_from_flat,
    linear_flat_defaults,
)
from models.linears.linear.config_builder import LinearConfigBuilder
from models.linears.linear.presets import (
    ExperimentPreset as SourceExperimentPreset,
)
from models.linears.linear.presets import (
    ExperimentPresets as SourceExperimentPresets,
)
from models.neuron._source_adapter import SourcePackageAdapter


SOURCE_LINEAR_KWARG_ALIASES = {
    "gate_hidden_dim": "gate_stack_hidden_dim",
    "gate_layer_norm_position": "gate_stack_layer_norm_position",
    "gate_bias_flag": "gate_stack_bias_flag",
    "halting_hidden_dim": "halting_stack_hidden_dim",
    "halting_layer_norm_position": "halting_stack_layer_norm_position",
    "halting_bias_flag": "halting_stack_bias_flag",
}

SOURCE_ADAPTER = SourcePackageAdapter(
    config_module=source_config,
    builder_type=LinearConfigBuilder,
    experiment_preset_type=SourceExperimentPreset,
    experiment_presets_type=SourceExperimentPresets,
    builder_kwargs_from_flat_fn=linear_builder_kwargs_from_flat,
    flat_defaults_fn=linear_flat_defaults,
    kwarg_aliases=SOURCE_LINEAR_KWARG_ALIASES,
)


def canonical_source_kwarg_aliases() -> dict[str, str]:
    return SOURCE_ADAPTER.canonical_kwarg_aliases()


def normalize_source_kwargs(source_kwargs: dict[str, Any]) -> dict[str, Any]:
    return SOURCE_ADAPTER.normalize_source_kwargs(source_kwargs)


def source_builder_kwargs_from_flat(source_kwargs: dict[str, Any]) -> dict[str, Any]:
    return SOURCE_ADAPTER.builder_kwargs_from_flat(source_kwargs)


def source_linear_default_kwargs() -> dict[str, Any]:
    return SOURCE_ADAPTER.source_default_kwargs()


def source_preset(preset) -> SourceExperimentPreset:
    return SOURCE_ADAPTER.source_preset(preset)


def source_values_for_preset(preset) -> dict[str, object]:
    return SOURCE_ADAPTER.source_values_for_preset(preset)


def source_locks_for_preset(preset) -> dict[str, object]:
    return SOURCE_ADAPTER.source_locks_for_preset(preset)

from typing import Any

import models.linears.linear_adaptive.config as source_config
from models.linears._builder_adapter import (
    linear_adaptive_builder_kwargs_from_flat,
    linear_flat_defaults,
)
from models.linears.linear_adaptive.config_builder import LinearAdaptiveConfigBuilder
from models.linears.linear_adaptive.presets import (
    ExperimentPreset as SourceExperimentPreset,
)
from models.linears.linear_adaptive.presets import (
    ExperimentPresets as SourceExperimentPresets,
)
from models.neuron._source_adapter import SourcePackageAdapter


SOURCE_ADAPTER = SourcePackageAdapter(
    config_module=source_config,
    builder_type=LinearAdaptiveConfigBuilder,
    experiment_preset_type=SourceExperimentPreset,
    experiment_presets_type=SourceExperimentPresets,
    builder_kwargs_from_flat_fn=linear_adaptive_builder_kwargs_from_flat,
    flat_defaults_fn=linear_flat_defaults,
)


def normalize_source_kwargs(source_kwargs: dict[str, Any]) -> dict[str, Any]:
    return SOURCE_ADAPTER.normalize_source_kwargs(source_kwargs)


def source_builder_kwargs_from_flat(source_kwargs: dict[str, Any]) -> dict[str, Any]:
    return SOURCE_ADAPTER.builder_kwargs_from_flat(source_kwargs)


def source_default_kwargs() -> dict[str, Any]:
    return SOURCE_ADAPTER.source_default_kwargs()


def source_preset(preset) -> SourceExperimentPreset:
    return SOURCE_ADAPTER.source_preset(preset)


def source_values_for_preset(preset) -> dict[str, object]:
    return SOURCE_ADAPTER.source_values_for_preset(preset)


def source_locks_for_preset(preset) -> dict[str, object]:
    return SOURCE_ADAPTER.source_locks_for_preset(preset)

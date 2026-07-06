from __future__ import annotations

from emperor.base.options import BaseOptions
from emperor.experiments.base import PresetDefinition

from models.neuron._source_adapter import SourcePackageAdapter


def create_experiment_preset(source_preset_type, module_name: str):
    members = {preset.name: preset.value for preset in source_preset_type}
    preset_type = BaseOptions("ExperimentPreset", members)
    preset_type.__module__ = module_name
    return preset_type


def create_preset_definitions(
    experiment_preset_type,
    source_adapter: SourcePackageAdapter,
) -> dict[object, PresetDefinition]:
    return {
        preset: PresetDefinition(
            preset_values=source_adapter.source_values_for_preset(preset),
            description=source_adapter.source_description_for_preset(preset),
        )
        for preset in experiment_preset_type
    }

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from enum import Enum
from types import ModuleType
from typing import Any

from emperor.experiments.monitors import MonitorOption
from emperor.experiments.tasks import (
    ExperimentTask,
    experiment_task_label,
    experiment_task_name,
    resolve_experiment_task,
)
from models.catalog import (
    discover_model_ids,
    is_safe_model_id,
    module_path_for_model_id,
)
from models.dataset_naming import (
    dataset_cli_name,
    dataset_label,
    dataset_name,
    normalize_dataset_name,
)
from models.model_metadata import ModelMetadata, load_model_metadata
from torch.nn import Module

from viewer.backend.inspector.errors import InspectorError

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


def _is_path_like_dataset_input(dataset: str) -> bool:
    stripped = dataset.strip()
    return "/" in stripped or "\\" in stripped


@dataclass(frozen=True)
class ModelParts:
    name: str
    metadata: ModelMetadata
    config_module: ModuleType
    dataset_options_module: ModuleType
    monitor_options_module: ModuleType
    search_space_module: ModuleType
    presets_module: ModuleType
    model_module: ModuleType
    experiment_preset_enum: type[Enum]
    presets: Any
    model_type: type[Module]
    default_experiment_task: ExperimentTask
    dataset_options_by_task: dict[ExperimentTask, list[type]]


def discover_models() -> list[str]:
    return discover_model_ids()


def validate_model_name(model_name: str) -> None:
    if not is_safe_model_id(model_name):
        raise InspectorError(f"Invalid model name: {model_name!r}")
    if module_path_for_model_id(model_name) is None:
        raise InspectorError(f"Unknown model: {model_name}")


def load_model_parts(model_name: str) -> ModelParts:
    validate_model_name(model_name)
    module_path = module_path_for_model_id(model_name)
    if module_path is None:
        raise InspectorError(f"Unknown model: {model_name}")
    try:
        metadata = load_model_metadata(model_name)
        config_module = metadata.config_module
        presets_module = importlib.import_module(f"{module_path}.presets")
        model_module = importlib.import_module(f"{module_path}.model")
    except Exception as exc:
        raise InspectorError(
            f"Failed to import model package '{model_name}': {exc}"
        ) from exc

    try:
        experiment_preset_enum = presets_module.ExperimentPreset
        presets = presets_module.ExperimentPresets()
        model_type = model_module.Model
    except AttributeError as exc:
        raise InspectorError(
            f"Model package '{model_name}' is missing ExperimentPreset, "
            "ExperimentPresets, or Model."
        ) from exc

    dataset_options_by_task = metadata.dataset_options_by_task
    if not dataset_options_by_task:
        raise InspectorError(
            f"Model package '{model_name}' does not define DATASET_OPTIONS_BY_TASK."
        )
    default_experiment_task = metadata.default_experiment_task

    return ModelParts(
        name=model_name,
        metadata=metadata,
        config_module=config_module,
        dataset_options_module=metadata.dataset_options_module,
        monitor_options_module=metadata.monitor_options_module,
        search_space_module=metadata.search_space_module,
        presets_module=presets_module,
        model_module=model_module,
        experiment_preset_enum=experiment_preset_enum,
        presets=presets,
        model_type=model_type,
        default_experiment_task=default_experiment_task,
        dataset_options_by_task=dataset_options_by_task,
    )


def preset_cli_name(experiment_preset_enum: type[Enum], preset: Enum) -> str:
    cli_name = getattr(experiment_preset_enum, "cli_name", None)
    if callable(cli_name):
        return cli_name(preset.name)
    return preset.name.lower().replace("_", "-")


def preset_description(preset: Enum, presets: Any | None = None) -> str:
    description_for_preset = getattr(presets, "description_for_preset", None)
    description = (
        description_for_preset(preset) if callable(description_for_preset) else None
    )
    if isinstance(description, str):
        return description
    return preset.value if isinstance(preset.value, str) else ""


def resolve_model_experiment_task(
    parts: ModelParts,
    experiment_task: str | ExperimentTask | None,
) -> ExperimentTask:
    if experiment_task is None:
        return parts.default_experiment_task
    try:
        task = resolve_experiment_task(experiment_task)
    except ValueError as exc:
        valid = ", ".join(
            experiment_task_name(candidate) for candidate in parts.dataset_options_by_task
        )
        raise InspectorError(
            f"Unknown experiment task '{experiment_task}' for model '{parts.name}'. "
            f"Valid tasks: {valid}."
        ) from exc
    if task not in parts.dataset_options_by_task:
        valid = ", ".join(
            experiment_task_name(candidate) for candidate in parts.dataset_options_by_task
        )
        raise InspectorError(
            f"Unknown experiment task '{experiment_task}' for model '{parts.name}'. "
            f"Valid tasks: {valid}."
        )
    return task


def dataset_options_for_task(
    parts: ModelParts,
    experiment_task: str | ExperimentTask | None,
) -> list[type]:
    if not hasattr(parts, "dataset_options_by_task"):
        return list(getattr(parts, "dataset_options", []) or [])
    task = resolve_model_experiment_task(parts, experiment_task)
    return list(parts.dataset_options_by_task[task])


def resolve_dataset(
    parts: ModelParts,
    dataset: str | None,
    experiment_task: str | ExperimentTask | None = None,
) -> type:
    dataset_options = dataset_options_for_task(parts, experiment_task)
    if dataset is None:
        return dataset_options[0]
    normalized = normalize_dataset_name(dataset)
    for dataset_type in dataset_options:
        names = {
            dataset_name(dataset_type),
            dataset_name(dataset_type).lower(),
            dataset_cli_name(dataset_type),
        }
        if dataset in names or dataset.lower() in names:
            return dataset_type
    if _is_path_like_dataset_input(dataset):
        valid = ", ".join(dataset_name(item) for item in dataset_options)
        raise InspectorError(
            f"Dataset input '{dataset}' for model '{parts.name}' looks like a "
            "filesystem path. Use a server-known dataset name instead. "
            f"Valid datasets: {valid}."
        )
    for dataset_type in dataset_options:
        names = {
            dataset_name(dataset_type),
            dataset_name(dataset_type).lower(),
            dataset_cli_name(dataset_type),
        }
        if normalized in names:
            return dataset_type
    valid = ", ".join(dataset_name(item) for item in dataset_options)
    raise InspectorError(
        f"Unknown dataset '{dataset}' for model '{parts.name}'. "
        f"Valid datasets: {valid}."
    )


def resolve_datasets(
    parts: ModelParts,
    datasets: list[str] | None,
    experiment_task: str | ExperimentTask | None = None,
) -> list[type]:
    if not datasets:
        return [resolve_dataset(parts, None, experiment_task)]
    resolved = [
        resolve_dataset(parts, dataset, experiment_task) for dataset in datasets
    ]
    seen = set()
    unique = []
    for dataset in resolved:
        name = dataset_name(dataset)
        if name in seen:
            continue
        seen.add(name)
        unique.append(dataset)
    return unique


def serialize_dataset(dataset: type) -> dict[str, Any]:
    return {
        "name": dataset_name(dataset),
        "label": dataset_label(dataset),
        "inputDim": int(getattr(dataset, "flattened_input_dim", 0) or 0),
        "outputDim": int(getattr(dataset, "num_classes", 0) or 0),
    }


def model_monitor_options(parts: ModelParts) -> list[MonitorOption]:
    monitor_options_module = getattr(
        parts,
        "monitor_options_module",
        parts.config_module,
    )
    raw_options = getattr(monitor_options_module, "MONITOR_OPTIONS", [])
    if raw_options is None:
        return []
    options = list(raw_options)
    invalid_options = [
        type(option).__name__
        for option in options
        if not isinstance(option, MonitorOption)
    ]
    if invalid_options:
        raise InspectorError(
            f"Model package '{parts.name}' has invalid MONITOR_OPTIONS entries: "
            f"{', '.join(invalid_options)}."
        )
    option_names = [option.name for option in options]
    duplicate_names = sorted(
        name for name in set(option_names) if option_names.count(name) > 1
    )
    if duplicate_names:
        raise InspectorError(
            f"Model package '{parts.name}' has duplicate monitor options: "
            f"{', '.join(duplicate_names)}."
        )
    return options


def list_model_monitors(model_name: str) -> list[dict[str, object]]:
    parts = load_model_parts(model_name)
    return [option.to_api() for option in model_monitor_options(parts)]


def resolve_model_monitors(
    parts: ModelParts,
    monitor_names: list[str] | None,
) -> list[MonitorOption]:
    if not monitor_names:
        return []
    options_by_name = {option.name: option for option in model_monitor_options(parts)}
    selected = []
    seen = set()
    unknown = []
    for name in monitor_names:
        if name in seen:
            continue
        seen.add(name)
        option = options_by_name.get(name)
        if option is None:
            unknown.append(name)
            continue
        selected.append(option)
    if unknown:
        valid = ", ".join(sorted(options_by_name)) or "none"
        raise InspectorError(
            f"Unknown monitor option(s) for model '{parts.name}': "
            f"{', '.join(unknown)}. Valid monitors: {valid}."
        )
    return selected


def serialize_dataset_group(
    task: ExperimentTask,
    datasets: list[type],
) -> dict[str, Any]:
    return {
        "experimentTask": experiment_task_name(task),
        "label": experiment_task_label(task),
        "datasets": [serialize_dataset(dataset) for dataset in datasets],
    }


def list_model_datasets(model_name: str) -> dict[str, Any]:
    parts = load_model_parts(model_name)
    return {
        "defaultExperimentTask": experiment_task_name(parts.default_experiment_task),
        "datasetGroups": [
            serialize_dataset_group(task, datasets)
            for task, datasets in parts.dataset_options_by_task.items()
        ],
    }


def list_model_presets(model_name: str) -> list[dict[str, str]]:
    parts = load_model_parts(model_name)
    return [
        {
            "name": preset_cli_name(parts.experiment_preset_enum, preset),
            "label": preset.name,
            "description": preset_description(preset, parts.presets),
        }
        for preset in parts.experiment_preset_enum
    ]

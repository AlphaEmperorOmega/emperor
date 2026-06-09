from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from enum import Enum
from types import ModuleType
from typing import Any

from torch.nn import Module

from emperor.experiments.monitors import MonitorOption
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
from viewer.backend.inspector.errors import InspectorError

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


def _is_path_like_dataset_input(dataset: str) -> bool:
    stripped = dataset.strip()
    return "/" in stripped or "\\" in stripped


@dataclass(frozen=True)
class ModelParts:
    name: str
    config_module: ModuleType
    presets_module: ModuleType
    model_module: ModuleType
    experiment_options: type[Enum]
    presets: Any
    model_type: type[Module]
    dataset_options: list[type]
    dataset: type


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
        config_module = importlib.import_module(f"{module_path}.config")
        presets_module = importlib.import_module(f"{module_path}.presets")
        model_module = importlib.import_module(f"{module_path}.model")
    except Exception as exc:
        raise InspectorError(f"Failed to import model package '{model_name}': {exc}") from exc

    try:
        experiment_options = presets_module.ExperimentOptions
        presets = presets_module.ExperimentPresets()
        model_type = model_module.Model
    except AttributeError as exc:
        raise InspectorError(
            f"Model package '{model_name}' is missing ExperimentOptions, "
            "ExperimentPresets, or Model."
        ) from exc

    dataset_options = getattr(config_module, "DATASET_OPTIONS", None)
    if not dataset_options:
        raise InspectorError(f"Model package '{model_name}' does not define DATASET_OPTIONS.")

    return ModelParts(
        name=model_name,
        config_module=config_module,
        presets_module=presets_module,
        model_module=model_module,
        experiment_options=experiment_options,
        presets=presets,
        model_type=model_type,
        dataset_options=list(dataset_options),
        dataset=dataset_options[0],
    )


def option_cli_name(experiment_options: type[Enum], option: Enum) -> str:
    cli_name = getattr(experiment_options, "cli_name", None)
    if callable(cli_name):
        return cli_name(option.name)
    return option.name.lower().replace("_", "-")


def option_description(option: Enum) -> str:
    return option.value if isinstance(option.value, str) else ""


def resolve_dataset(parts: ModelParts, dataset: str | None) -> type:
    if dataset is None:
        return parts.dataset
    normalized = normalize_dataset_name(dataset)
    for dataset_type in parts.dataset_options:
        names = {
            dataset_name(dataset_type),
            dataset_name(dataset_type).lower(),
            dataset_cli_name(dataset_type),
        }
        if dataset in names or dataset.lower() in names:
            return dataset_type
    if _is_path_like_dataset_input(dataset):
        valid = ", ".join(dataset_name(item) for item in parts.dataset_options)
        raise InspectorError(
            f"Dataset input '{dataset}' for model '{parts.name}' looks like a "
            "filesystem path. Use a server-known dataset name instead. "
            f"Valid datasets: {valid}."
        )
    for dataset_type in parts.dataset_options:
        names = {
            dataset_name(dataset_type),
            dataset_name(dataset_type).lower(),
            dataset_cli_name(dataset_type),
        }
        if normalized in names:
            return dataset_type
    valid = ", ".join(dataset_name(item) for item in parts.dataset_options)
    raise InspectorError(
        f"Unknown dataset '{dataset}' for model '{parts.name}'. Valid datasets: {valid}."
    )


def resolve_datasets(parts: ModelParts, datasets: list[str] | None) -> list[type]:
    if not datasets:
        return [parts.dataset]
    resolved = [resolve_dataset(parts, dataset) for dataset in datasets]
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
    raw_options = getattr(parts.config_module, "MONITOR_OPTIONS", [])
    if raw_options is None:
        return []
    options = list(raw_options)
    invalid_options = [
        type(option).__name__ for option in options if not isinstance(option, MonitorOption)
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


def list_model_datasets(model_name: str) -> list[dict[str, Any]]:
    parts = load_model_parts(model_name)
    return [serialize_dataset(dataset) for dataset in parts.dataset_options]


def list_model_presets(model_name: str) -> list[dict[str, str]]:
    parts = load_model_parts(model_name)
    return [
        {
            "name": option_cli_name(parts.experiment_options, option),
            "label": option.name,
            "description": option_description(option),
        }
        for option in parts.experiment_options
    ]

from __future__ import annotations

import importlib
import os
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import ModuleType
from typing import Any

from torch.nn import Module

from viewer.backend.inspector.errors import InspectorError

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

MODEL_NAME_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


@dataclass(frozen=True)
class ModelParts:
    name: str
    config_module: ModuleType
    presets_module: ModuleType
    model_module: ModuleType
    experiment_options: type[Enum]
    presets: Any
    model_type: type[Module]
    dataset: type


def models_root() -> Path:
    models_package = importlib.import_module("models")
    package_file = getattr(models_package, "__file__", None)
    if package_file is None:
        raise InspectorError("Could not resolve the models package path.")
    return Path(package_file).parent


def discover_models() -> list[str]:
    root = models_root()
    model_names = []
    for path in root.iterdir():
        if not path.is_dir() or path.name.startswith("__"):
            continue
        if not (path / "__init__.py").exists():
            continue
        if not (path / "config.py").exists():
            continue
        if not (path / "presets.py").exists():
            continue
        if not (path / "model.py").exists():
            continue
        model_names.append(path.name)
    return sorted(model_names)


def validate_model_name(model_name: str) -> None:
    if not MODEL_NAME_PATTERN.match(model_name):
        raise InspectorError(f"Invalid model name: {model_name!r}")
    if model_name not in discover_models():
        raise InspectorError(f"Unknown model: {model_name}")


def load_model_parts(model_name: str) -> ModelParts:
    validate_model_name(model_name)
    module_path = f"models.{model_name}"
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
        dataset=dataset_options[0],
    )


def option_cli_name(experiment_options: type[Enum], option: Enum) -> str:
    cli_name = getattr(experiment_options, "cli_name", None)
    if callable(cli_name):
        return cli_name(option.name)
    return option.name.lower().replace("_", "-")


def option_description(option: Enum) -> str:
    return option.value if isinstance(option.value, str) else ""


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

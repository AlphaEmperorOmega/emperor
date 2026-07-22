from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import ModuleType
from typing import Any

from emperor.experiments import (
    ExperimentTask,
    experiment_task_label,
    experiment_task_name,
    resolve_experiment_task,
)
from model_runtime.packages.identity import ModelIdentity


def _coerce_dataset_options_by_task(
    identity: ModelIdentity,
    dataset_options: ModuleType,
) -> dict[ExperimentTask, list[type]]:
    raw_options = getattr(dataset_options, "DATASET_OPTIONS_BY_TASK", None)
    if not isinstance(raw_options, dict) or not raw_options:
        raise ValueError(
            f"Model Package '{identity.catalog_key}' must define non-empty "
            "DATASET_OPTIONS_BY_TASK."
        )

    options_by_task: dict[ExperimentTask, list[type]] = {}
    for raw_task, raw_datasets in raw_options.items():
        task = resolve_experiment_task(raw_task)
        if task is None:
            raise ValueError(
                f"Model Package '{identity.catalog_key}' has invalid Experiment "
                f"Task {raw_task!r}."
            )
        if not isinstance(raw_datasets, list) or not raw_datasets:
            raise ValueError(
                f"Model Package '{identity.catalog_key}' must define a non-empty "
                f"dataset list for {experiment_task_name(task)}."
            )
        options_by_task[task] = list(raw_datasets)
    return options_by_task


def _coerce_default_experiment_task(
    identity: ModelIdentity,
    dataset_options: ModuleType,
    options_by_task: dict[ExperimentTask, list[type]],
) -> ExperimentTask:
    default_task = resolve_experiment_task(
        getattr(dataset_options, "DEFAULT_EXPERIMENT_TASK", None)
    )
    if default_task is None:
        raise ValueError(
            f"Model Package '{identity.catalog_key}' must define "
            "DEFAULT_EXPERIMENT_TASK."
        )
    if default_task not in options_by_task:
        raise ValueError(
            f"Model Package '{identity.catalog_key}' default Experiment Task "
            f"{experiment_task_name(default_task)!r} is not present in "
            "DATASET_OPTIONS_BY_TASK."
        )
    return default_task


@dataclass(frozen=True)
class ModelMetadata:
    """Descriptive metadata supplied by one package-local adapter."""

    identity: ModelIdentity
    runtime_defaults: ModuleType
    dataset_options: ModuleType
    monitor_options_source: ModuleType
    search_space: ModuleType
    module_path: str | None = None

    @property
    def model_name(self) -> str:
        return self.identity.catalog_key

    @property
    def config_module(self) -> ModuleType:
        return self.runtime_defaults

    @property
    def dataset_options_module(self) -> ModuleType:
        return self.dataset_options

    @property
    def monitor_options_module(self) -> ModuleType:
        return self.monitor_options_source

    @property
    def search_space_module(self) -> ModuleType:
        return self.search_space

    @property
    def dataset_options_by_task(self) -> dict[ExperimentTask, list[type]]:
        return _coerce_dataset_options_by_task(self.identity, self.dataset_options)

    @property
    def default_experiment_task(self) -> ExperimentTask:
        return _coerce_default_experiment_task(
            self.identity,
            self.dataset_options,
            self.dataset_options_by_task,
        )

    @property
    def experiment_tasks(self) -> list[ExperimentTask]:
        return list(self.dataset_options_by_task)

    def dataset_options_for_task(
        self,
        task: str | ExperimentTask | None = None,
    ) -> list[type]:
        options_by_task = self.dataset_options_by_task
        resolved_task = (
            self.default_experiment_task
            if task is None
            else resolve_experiment_task(task)
        )
        if resolved_task is None:
            raise ValueError(f"Invalid Experiment Task: {task!r}")
        try:
            return list(options_by_task[resolved_task])
        except KeyError as exc:
            valid = ", ".join(
                experiment_task_name(candidate) for candidate in options_by_task
            )
            raise ValueError(
                f"Unknown Experiment Task {task!r} for Model Package "
                f"'{self.identity.catalog_key}'. Valid tasks: {valid}."
            ) from exc

    def dataset_groups(self) -> list[dict[str, Any]]:
        return [
            {
                "experimentTask": experiment_task_name(task),
                "label": experiment_task_label(task),
                "datasets": list(datasets),
            }
            for task, datasets in self.dataset_options_by_task.items()
        ]

    @property
    def monitor_options(self) -> list[Any]:
        return list(getattr(self.monitor_options_source, "MONITOR_OPTIONS", []) or [])

    @property
    def search_space_items(self) -> dict[str, list[Any]]:
        return {
            key: value
            for key, value in vars(self.search_space).items()
            if key.startswith("SEARCH_SPACE_") and isinstance(value, list)
        }


def load_model_metadata_from_module_path(
    module_path: str,
    *,
    model_name: str | None = None,
) -> ModelMetadata:
    identity_name = model_name or module_path.removeprefix("models.").replace(".", "/")
    segments = identity_name.split("/")
    if len(segments) != 2:
        raise ValueError(f"Invalid model identity: {identity_name!r}")
    identity = ModelIdentity(segments[0], segments[1])
    return ModelMetadata(
        identity=identity,
        runtime_defaults=importlib.import_module(f"{module_path}.config"),
        dataset_options=importlib.import_module(f"{module_path}.dataset_options"),
        monitor_options_source=importlib.import_module(
            f"{module_path}.monitor_options"
        ),
        search_space=importlib.import_module(f"{module_path}.search_space"),
        module_path=module_path,
    )


def load_model_metadata_for_config_module(config_module: ModuleType) -> ModelMetadata:
    module_name = config_module.__name__
    if not module_name.endswith(".config"):
        raise ValueError(f"Expected a config module, got {module_name!r}")
    return load_model_metadata_from_module_path(module_name[: -len(".config")])


__all__ = [
    "ModelMetadata",
    "load_model_metadata_for_config_module",
    "load_model_metadata_from_module_path",
]

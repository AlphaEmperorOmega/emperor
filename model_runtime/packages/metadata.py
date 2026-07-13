from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import ModuleType
from typing import Any

from emperor.experiments.tasks import (
    ExperimentTask,
    experiment_task_label,
    experiment_task_name,
    resolve_experiment_task,
)


def _coerce_dataset_options_by_task(
    dataset_options_module: ModuleType,
) -> dict[ExperimentTask, list[type]]:
    raw_options = getattr(dataset_options_module, "DATASET_OPTIONS_BY_TASK", None)
    if not isinstance(raw_options, dict) or not raw_options:
        raise ValueError(
            f"Model dataset options '{dataset_options_module.__name__}' must define "
            "non-empty DATASET_OPTIONS_BY_TASK."
        )

    options_by_task: dict[ExperimentTask, list[type]] = {}
    for raw_task, raw_datasets in raw_options.items():
        task = resolve_experiment_task(raw_task)
        if task is None:
            raise ValueError(
                f"Model dataset options '{dataset_options_module.__name__}' has "
                f"invalid experiment task {raw_task!r}."
            )
        if not isinstance(raw_datasets, list) or not raw_datasets:
            raise ValueError(
                f"Model dataset options '{dataset_options_module.__name__}' must "
                f"define a non-empty dataset list for {experiment_task_name(task)}."
            )
        options_by_task[task] = list(raw_datasets)
    return options_by_task


def _coerce_default_experiment_task(
    dataset_options_module: ModuleType,
    options_by_task: dict[ExperimentTask, list[type]],
) -> ExperimentTask:
    default_task = resolve_experiment_task(
        getattr(dataset_options_module, "DEFAULT_EXPERIMENT_TASK", None)
    )
    if default_task is None:
        raise ValueError(
            f"Model dataset options '{dataset_options_module.__name__}' must define "
            "DEFAULT_EXPERIMENT_TASK."
        )
    if default_task not in options_by_task:
        raise ValueError(
            f"Model dataset options '{dataset_options_module.__name__}' default "
            f"experiment task {experiment_task_name(default_task)!r} is not present "
            "in DATASET_OPTIONS_BY_TASK."
        )
    return default_task


@dataclass(frozen=True)
class ModelMetadata:
    """Descriptive Model Metadata loaded through a Model Package."""

    model_name: str
    module_path: str
    config_module: ModuleType
    dataset_options_module: ModuleType
    monitor_options_module: ModuleType
    search_space_module: ModuleType

    @property
    def dataset_options_by_task(self) -> dict[ExperimentTask, list[type]]:
        return _coerce_dataset_options_by_task(self.dataset_options_module)

    @property
    def default_experiment_task(self) -> ExperimentTask:
        return _coerce_default_experiment_task(
            self.dataset_options_module,
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
            raise ValueError(f"Invalid experiment task: {task!r}")
        try:
            return list(options_by_task[resolved_task])
        except KeyError as exc:
            valid = ", ".join(
                experiment_task_name(candidate) for candidate in options_by_task
            )
            raise ValueError(
                f"Unknown experiment task {task!r} for model '{self.model_name}'. "
                f"Valid tasks: {valid}."
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
        return list(getattr(self.monitor_options_module, "MONITOR_OPTIONS", []) or [])

    @property
    def search_space_items(self) -> dict[str, list[Any]]:
        return {
            key: value
            for key, value in vars(self.search_space_module).items()
            if key.startswith("SEARCH_SPACE_") and isinstance(value, list)
        }


def load_model_metadata_from_module_path(
    module_path: str,
    *,
    model_name: str | None = None,
) -> ModelMetadata:
    return ModelMetadata(
        model_name=model_name or module_path.removeprefix("models.").replace(".", "/"),
        module_path=module_path,
        config_module=importlib.import_module(f"{module_path}.config"),
        dataset_options_module=importlib.import_module(
            f"{module_path}.dataset_options"
        ),
        monitor_options_module=importlib.import_module(
            f"{module_path}.monitor_options"
        ),
        search_space_module=importlib.import_module(f"{module_path}.search_space"),
    )


def load_model_metadata_for_config_module(config_module: ModuleType) -> ModelMetadata:
    module_name = config_module.__name__
    if not module_name.endswith(".config"):
        raise ValueError(f"Expected a config module, got {module_name!r}")
    return load_model_metadata_from_module_path(
        module_name[: -len(".config")],
    )


__all__ = [
    "ModelMetadata",
    "load_model_metadata_for_config_module",
    "load_model_metadata_from_module_path",
]

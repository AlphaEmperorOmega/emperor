from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType, ModuleType
from typing import Any

from emperor.experiments import (
    ExperimentTask,
    experiment_task_name,
    resolve_experiment_task,
)
from emperor.monitoring import MonitorOption
from model_runtime.packages.identity import ModelIdentity


def _coerce_dataset_options_by_task(
    identity: ModelIdentity,
    dataset_options: ModuleType,
) -> dict[ExperimentTask, tuple[type, ...]]:
    raw_options = getattr(dataset_options, "DATASET_OPTIONS_BY_TASK", None)
    if not isinstance(raw_options, dict) or not raw_options:
        raise ValueError(
            f"Model Package '{identity.catalog_key}' must define non-empty "
            "DATASET_OPTIONS_BY_TASK."
        )

    options_by_task: dict[ExperimentTask, tuple[type, ...]] = {}
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
        if any(not isinstance(dataset, type) for dataset in raw_datasets):
            raise ValueError(
                f"Model Package '{identity.catalog_key}' dataset options for "
                f"{experiment_task_name(task)} must contain dataset types."
            )
        if task in options_by_task:
            raise ValueError(
                f"Model Package '{identity.catalog_key}' defines duplicate "
                f"Experiment Task {experiment_task_name(task)!r}."
            )
        options_by_task[task] = tuple(raw_datasets)
    return options_by_task


def _coerce_default_experiment_task(
    identity: ModelIdentity,
    dataset_options: ModuleType,
    options_by_task: Mapping[ExperimentTask, tuple[type, ...]],
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
    _dataset_options_by_task: Mapping[ExperimentTask, tuple[type, ...]] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _default_task: ExperimentTask = field(init=False, repr=False, compare=False)
    _monitor_options: tuple[MonitorOption, ...] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _search_space_items: Mapping[str, tuple[Any, ...]] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        options_by_task = _coerce_dataset_options_by_task(
            self.identity,
            self.dataset_options,
        )
        default_task = _coerce_default_experiment_task(
            self.identity,
            self.dataset_options,
            options_by_task,
        )
        raw_monitors = getattr(self.monitor_options_source, "MONITOR_OPTIONS", []) or []
        if not isinstance(raw_monitors, list):
            raise ValueError(
                f"Model Package '{self.identity.catalog_key}' MONITOR_OPTIONS "
                "must be a list."
            )
        invalid_monitors = [
            type(option).__name__
            for option in raw_monitors
            if not isinstance(option, MonitorOption)
        ]
        if invalid_monitors:
            raise ValueError(
                f"Model package '{self.identity.catalog_key}' has invalid "
                f"MONITOR_OPTIONS entries: {', '.join(invalid_monitors)}."
            )
        monitor_names = [option.name for option in raw_monitors]
        duplicate_monitors = sorted(
            name for name in set(monitor_names) if monitor_names.count(name) > 1
        )
        if duplicate_monitors:
            raise ValueError(
                f"Model package '{self.identity.catalog_key}' has duplicate "
                f"monitor options: {', '.join(duplicate_monitors)}."
            )
        search_items = {
            key: tuple(value)
            for key, value in vars(self.search_space).items()
            if key.startswith("SEARCH_SPACE_") and isinstance(value, list)
        }
        object.__setattr__(
            self,
            "_dataset_options_by_task",
            MappingProxyType(options_by_task),
        )
        object.__setattr__(self, "_default_task", default_task)
        object.__setattr__(self, "_monitor_options", tuple(raw_monitors))
        object.__setattr__(
            self,
            "_search_space_items",
            MappingProxyType(search_items),
        )

    @property
    def dataset_options_by_task(self) -> dict[ExperimentTask, list[type]]:
        return {
            task: list(datasets)
            for task, datasets in self._dataset_options_by_task.items()
        }

    @property
    def default_experiment_task(self) -> ExperimentTask:
        return self._default_task

    @property
    def experiment_tasks(self) -> list[ExperimentTask]:
        return list(self._dataset_options_by_task)

    def dataset_options_for_task(
        self,
        task: str | ExperimentTask | None = None,
    ) -> list[type]:
        options_by_task = self._dataset_options_by_task
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

    @property
    def monitor_options(self) -> list[Any]:
        return list(self._monitor_options)

    @property
    def search_space_items(self) -> dict[str, list[Any]]:
        return {key: list(values) for key, values in self._search_space_items.items()}


__all__ = ["ModelMetadata"]

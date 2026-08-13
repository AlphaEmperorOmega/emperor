from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType, ModuleType
from typing import TYPE_CHECKING, Any, cast

from emperor.experiments import (
    ExperimentTask,
    experiment_task_name,
    resolve_experiment_task,
)
from model_runtime.packages.configuration import (
    config_key_to_model_param,
    iter_supported_config_keys,
    normalize_key,
)
from model_runtime.packages.configuration_metadata import configuration_field_metadata
from model_runtime.packages.identity import ModelIdentity
from model_runtime.packages.runtime_defaults import (
    RuntimeDefaultsError,
    RuntimeDefaultsSpec,
)

if TYPE_CHECKING:
    from emperor.monitoring import MonitorOption
    from model_runtime.packages.definition import ModelPackage
    from model_runtime.packages.inspection_limits import InspectionConstructionLimits


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
    raw_options_mapping = cast(dict[object, object], raw_options)
    for raw_task, raw_datasets in raw_options_mapping.items():
        task = resolve_experiment_task(cast(str | ExperimentTask | None, raw_task))
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
        raw_dataset_list = cast(list[object], raw_datasets)
        if any(not isinstance(dataset, type) for dataset in raw_dataset_list):
            raise ValueError(
                f"Model Package '{identity.catalog_key}' dataset options for "
                f"{experiment_task_name(task)} must contain dataset types."
            )
        if task in options_by_task:
            raise ValueError(
                f"Model Package '{identity.catalog_key}' defines duplicate "
                f"Experiment Task {experiment_task_name(task)!r}."
            )
        options_by_task[task] = tuple(cast(list[type[Any]], raw_dataset_list))
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


def _keys_by_alias(supported_keys: tuple[str, ...]) -> dict[str, str]:
    keys_by_alias: dict[str, str] = {}
    for config_key in supported_keys:
        keys_by_alias[normalize_key(config_key)] = config_key
        keys_by_alias[normalize_key(config_key_to_model_param(config_key))] = config_key
    return keys_by_alias


@dataclass(frozen=True, slots=True)
class _MetadataListSnapshot:
    values: tuple[Any, ...]


def _snapshot_metadata_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        nested_mapping = cast(Mapping[object, Any], value)
        return MappingProxyType(
            {
                key: _snapshot_metadata_value(nested_value)
                for key, nested_value in nested_mapping.items()
            }
        )
    if isinstance(value, list):
        nested_list = cast(list[Any], value)
        return _MetadataListSnapshot(
            tuple(_snapshot_metadata_value(item) for item in nested_list)
        )
    return value


def _project_metadata_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        nested_mapping = cast(Mapping[object, Any], value)
        return MappingProxyType(
            {
                key: _project_metadata_value(nested_value)
                for key, nested_value in nested_mapping.items()
            }
        )
    if isinstance(value, _MetadataListSnapshot):
        return [_project_metadata_value(item) for item in value.values]
    return value


class _MetadataSnapshot(Mapping[str, Mapping[str, Any]]):
    """Own immutable metadata while projecting fresh read-only entries."""

    __slots__ = ("_entries",)

    def __init__(self, metadata: Mapping[str, Mapping[str, Any]]) -> None:
        self._entries = MappingProxyType(
            {
                key: MappingProxyType(
                    {
                        field: _snapshot_metadata_value(value)
                        for field, value in entry.items()
                    }
                )
                for key, entry in metadata.items()
            }
        )

    def __getitem__(self, key: str) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                field: _project_metadata_value(value)
                for field, value in self._entries[key].items()
            }
        )

    def __iter__(self) -> Iterator[str]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return repr(MappingProxyType({key: self[key] for key in self}))


def _nested_metadata(
    metadata: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Mapping[str, Any]]:
    return _MetadataSnapshot(metadata)


def _coerce_monitor_options(
    identity: ModelIdentity,
    source: ModuleType,
) -> tuple[MonitorOption, ...]:
    from emperor.monitoring import MonitorOption

    raw_monitors = getattr(source, "MONITOR_OPTIONS", []) or []
    if not isinstance(raw_monitors, list):
        raise ValueError(
            f"Model Package '{identity.catalog_key}' MONITOR_OPTIONS must be a list."
        )
    raw_monitor_list = cast(list[object], raw_monitors)
    invalid_monitors = [
        type(option).__name__
        for option in raw_monitor_list
        if not isinstance(option, MonitorOption)
    ]
    if invalid_monitors:
        raise ValueError(
            f"Model package '{identity.catalog_key}' has invalid MONITOR_OPTIONS "
            f"entries: {', '.join(invalid_monitors)}."
        )
    monitor_options = cast(list[MonitorOption], raw_monitor_list)
    monitor_names = [option.name for option in monitor_options]
    duplicate_monitors = sorted(
        name for name in set(monitor_names) if monitor_names.count(name) > 1
    )
    if duplicate_monitors:
        raise ValueError(
            f"Model package '{identity.catalog_key}' has duplicate monitor options: "
            f"{', '.join(duplicate_monitors)}."
        )
    return tuple(monitor_options)


def _coerce_search_space_items(source: ModuleType) -> dict[str, tuple[Any, ...]]:
    search_items: dict[str, tuple[Any, ...]] = {}
    for key, value in cast(dict[str, object], vars(source)).items():
        if key.startswith("SEARCH_SPACE_") and isinstance(value, list):
            search_items[key] = tuple(cast(list[Any], value))
    return search_items


@dataclass(frozen=True, init=False)
class ModelMetadata:
    """Descriptive metadata supplied by one package-local adapter."""

    identity: ModelIdentity
    _runtime_defaults_source: ModuleType = field(repr=False)
    _dataset_options_source: ModuleType = field(repr=False)
    _monitor_options_source: ModuleType = field(repr=False)
    _search_space_source: ModuleType = field(repr=False)
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

    def __init__(
        self,
        identity: ModelIdentity,
        runtime_defaults: ModuleType,
        dataset_options: ModuleType,
        monitor_options_source: ModuleType,
        search_space: ModuleType,
    ) -> None:
        object.__setattr__(self, "identity", identity)
        object.__setattr__(self, "_runtime_defaults_source", runtime_defaults)
        object.__setattr__(self, "_dataset_options_source", dataset_options)
        object.__setattr__(self, "_monitor_options_source", monitor_options_source)
        object.__setattr__(self, "_search_space_source", search_space)
        self.__post_init__()

    def __post_init__(self) -> None:
        options_by_task = _coerce_dataset_options_by_task(
            self.identity,
            self._dataset_options_source,
        )
        default_task = _coerce_default_experiment_task(
            self.identity,
            self._dataset_options_source,
            options_by_task,
        )
        monitor_options = _coerce_monitor_options(
            self.identity,
            self._monitor_options_source,
        )
        search_items = _coerce_search_space_items(self._search_space_source)
        object.__setattr__(
            self,
            "_dataset_options_by_task",
            MappingProxyType(options_by_task),
        )
        object.__setattr__(self, "_default_task", default_task)
        object.__setattr__(self, "_monitor_options", monitor_options)
        object.__setattr__(
            self,
            "_search_space_items",
            MappingProxyType(search_items),
        )

    def compile_runtime_defaults_spec(
        self,
        package: ModelPackage,
        inspection_limits: InspectionConstructionLimits,
    ) -> RuntimeDefaultsSpec:
        try:
            config_module = self._runtime_defaults_source
            search_space_module = self._search_space_source
            supported_keys = tuple(iter_supported_config_keys(config_module))
            keys_by_alias = _keys_by_alias(supported_keys)
            config_metadata = configuration_field_metadata(config_module)
            search_metadata = configuration_field_metadata(
                search_space_module,
                include_search_space=True,
            )
            search_values = {
                key: tuple(values) for key, values in self.search_space_items.items()
            }
            skipped_schema_keys = frozenset(
                key
                for key in getattr(config_module, "CONFIG_SCHEMA_SKIP_KEYS", ())
                if isinstance(key, str) and key in supported_keys
            )
        except ValueError as exc:
            raise RuntimeDefaultsError(str(exc)) from exc
        except Exception as exc:
            raise RuntimeDefaultsError(
                f"Failed to import model package '{package.catalog_key}': {exc}"
            ) from exc

        return RuntimeDefaultsSpec(
            package=package,
            _config_module=config_module,
            _search_space_module=search_space_module,
            supported_keys=supported_keys,
            keys_by_alias=MappingProxyType(keys_by_alias),
            annotations=MappingProxyType(
                dict(getattr(config_module, "__annotations__", {}))
            ),
            search_annotations=MappingProxyType(
                dict(getattr(search_space_module, "__annotations__", {}))
            ),
            configuration_metadata=_nested_metadata(config_metadata),
            search_metadata=_nested_metadata(search_metadata),
            search_values=MappingProxyType(search_values),
            skipped_schema_keys=skipped_schema_keys,
            inspection_limits=inspection_limits,
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

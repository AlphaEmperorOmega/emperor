from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from threading import RLock
from types import ModuleType
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar, cast

from emperor.experiments.monitors import MonitorOption
from emperor.experiments.tasks import (
    ExperimentTask,
    experiment_task_label,
    experiment_task_name,
    resolve_experiment_task,
)

from model_runtime.packages.configuration_metadata import (
    configuration_field_metadata,
)
from model_runtime.packages.datasets import (
    dataset_cli_name,
    dataset_name,
    normalize_dataset_name,
)
from model_runtime.packages.identity import ModelIdentity, model_key
from model_runtime.packages.inspection_limits import (
    DEFAULT_INSPECTION_CONSTRUCTION_LIMITS,
    InspectionConstructionLimits,
)
from model_runtime.packages.metadata import (
    ModelMetadata,
    load_model_metadata_from_module_path,
)

if TYPE_CHECKING:
    from emperor.config import ModelConfig


_INITIALIZATION_MISSING = object()
_MODEL_PACKAGE_INITIALIZATION_LOCK = RLock()
_InitializationValue = TypeVar("_InitializationValue")


@dataclass(frozen=True, init=False)
class ModelPackage:
    """Catalog-owned Interface for one isolated Model Package.

    ``module_path`` is the temporary legacy adapter seam. All configuration and
    model construction remains implemented inside that concrete package.
    """

    model_type: str
    model: str
    module_path: str
    checkpoint_metadata_module: ClassVar[str | None] = None
    inspection_construction_limits: ClassVar[InspectionConstructionLimits] = (
        DEFAULT_INSPECTION_CONSTRUCTION_LIMITS
    )

    def __init__(
        self,
        model_type: str,
        model: str,
        module_path: str,
        checkpoint_metadata_module: str | None = None,
        inspection_construction_limits: InspectionConstructionLimits = (
            DEFAULT_INSPECTION_CONSTRUCTION_LIMITS
        ),
    ) -> None:
        object.__setattr__(self, "model_type", model_type)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "module_path", module_path)
        if checkpoint_metadata_module is not None:
            object.__setattr__(
                self,
                "checkpoint_metadata_module",
                checkpoint_metadata_module,
            )
        if inspection_construction_limits != DEFAULT_INSPECTION_CONSTRUCTION_LIMITS:
            object.__setattr__(
                self,
                "inspection_construction_limits",
                inspection_construction_limits,
            )

    def _initialize_once(
        self,
        attribute: str,
        loader: Callable[[], _InitializationValue],
    ) -> _InitializationValue:
        value = self.__dict__.get(attribute, _INITIALIZATION_MISSING)
        if value is not _INITIALIZATION_MISSING:
            return cast(_InitializationValue, value)
        with _MODEL_PACKAGE_INITIALIZATION_LOCK:
            value = self.__dict__.get(attribute, _INITIALIZATION_MISSING)
            if value is _INITIALIZATION_MISSING:
                value = loader()
                object.__setattr__(self, attribute, value)
            return cast(_InitializationValue, value)

    @property
    def identity(self) -> ModelIdentity:
        return ModelIdentity(self.model_type, self.model)

    @property
    def catalog_key(self) -> str:
        return model_key(self.model_type, self.model)

    @property
    def public_id(self) -> str:
        return self.catalog_key

    def to_identity_payload(self) -> dict[str, str]:
        return self.identity.to_payload()

    @property
    def metadata(self) -> ModelMetadata:
        return self._initialize_once(
            "_metadata",
            lambda: load_model_metadata_from_module_path(
                self.module_path,
                model_name=self.catalog_key,
            ),
        )

    @property
    def runtime_defaults(self) -> ModuleType:
        return self.metadata.config_module

    @property
    def default_experiment_task(self) -> ExperimentTask:
        return self.metadata.default_experiment_task

    @property
    def dataset_metadata(self) -> dict[ExperimentTask, list[type]]:
        return self.metadata.dataset_options_by_task

    @property
    def monitor_metadata(self) -> list[Any]:
        return self.metadata.monitor_options

    @property
    def search_metadata(self) -> dict[str, list[Any]]:
        return self.metadata.search_space_items

    @property
    def presets_module(self) -> ModuleType:
        return self._initialize_once(
            "_presets_module",
            lambda: importlib.import_module(f"{self.module_path}.presets"),
        )

    @property
    def model_module(self) -> ModuleType:
        return self._initialize_once(
            "_model_module",
            lambda: importlib.import_module(f"{self.module_path}.model"),
        )

    def checkpoint_config_overrides(
        self,
        tensor_shapes: Mapping[str, tuple[int, ...]],
    ) -> dict[str, Any]:
        """Interpret package-local checkpoint shapes when explicitly declared."""

        module_path = self.checkpoint_metadata_module
        if module_path is None:
            return {}
        module = self._initialize_once(
            "_checkpoint_metadata",
            lambda: importlib.import_module(module_path),
        )
        interpreter = getattr(module, "checkpoint_config_overrides", None)
        if not callable(interpreter):
            raise ValueError(
                f"Model package '{self.catalog_key}' checkpoint metadata does not "
                "define checkpoint_config_overrides()."
            )
        overrides = interpreter(tensor_shapes)
        if not isinstance(overrides, Mapping) or any(
            not isinstance(key, str) for key in overrides
        ):
            raise ValueError(
                f"Model package '{self.catalog_key}' returned invalid checkpoint "
                "configuration overrides."
            )
        return dict(overrides)

    @property
    def preset_type(self) -> type:
        return self.presets_module.ExperimentPreset

    @property
    def presets(self) -> Any:
        return self._initialize_once(
            "_presets",
            lambda: self.presets_module.ExperimentPresets(),
        )

    @property
    def experiment_type(self) -> type:
        return self.presets_module.Experiment

    @property
    def model_class(self) -> type:
        return self.model_module.Model

    def build_configurations(
        self,
        preset=None,
        dataset: type | None = None,
        **kwargs,
    ) -> list[ModelConfig]:
        selected_preset = preset or next(iter(self.preset_type))
        if dataset is None:
            return self.presets.get_config(selected_preset, **kwargs)
        return self.presets.get_config(selected_preset, dataset, **kwargs)

    def build_model(self, config: ModelConfig):
        return self.model_class(config)

    def resolve_preset(self, preset_name: str):
        preset_type = self.preset_type
        try:
            return preset_type.get_member(preset_name)
        except ValueError as exc:
            raise ValueError(
                f"Unknown preset '{preset_name}' for model '{self.catalog_key}'."
            ) from exc

    def preset_name(self, preset: Any) -> str:
        cli_name = getattr(self.preset_type, "cli_name", None)
        if callable(cli_name):
            return cli_name(preset.name)
        return preset.name.lower().replace("_", "-")

    def preset_description(self, preset: Any) -> str:
        description_for_preset = getattr(self.presets, "description_for_preset", None)
        description = (
            description_for_preset(preset)
            if callable(description_for_preset)
            else None
        )
        if isinstance(description, str):
            return description
        return preset.value if isinstance(preset.value, str) else ""

    def preset_locks(self, preset: Any) -> dict[str, Any]:
        locked_fields = getattr(self.presets, "locked_fields", None)
        if not callable(locked_fields):
            return {}
        return dict(locked_fields(preset))

    def resolve_experiment_task(
        self,
        experiment_task: str | ExperimentTask | None,
    ) -> ExperimentTask:
        if experiment_task is None:
            return self.default_experiment_task
        try:
            task = resolve_experiment_task(experiment_task)
        except ValueError as exc:
            task = None
            cause = exc
        else:
            cause = None
        if task not in self.dataset_metadata:
            valid = ", ".join(
                experiment_task_name(candidate) for candidate in self.dataset_metadata
            )
            error = ValueError(
                f"Unknown experiment task '{experiment_task}' for model "
                f"'{self.catalog_key}'. Valid tasks: {valid}."
            )
            if cause is not None:
                raise error from cause
            raise error
        assert task is not None
        return task

    def dataset_options_for_task(
        self,
        experiment_task: str | ExperimentTask | None = None,
    ) -> list[type]:
        task = self.resolve_experiment_task(experiment_task)
        return list(self.dataset_metadata[task])

    def task_name(self, task: ExperimentTask) -> str:
        return experiment_task_name(task)

    def task_label(self, task: ExperimentTask) -> str:
        return experiment_task_label(task)

    def resolve_dataset(
        self,
        dataset: str | None,
        experiment_task: str | ExperimentTask | None = None,
    ) -> type:
        options = self.dataset_options_for_task(experiment_task)
        if dataset is None:
            return options[0]
        stripped = dataset.strip()
        valid = ", ".join(dataset_name(item) for item in options)
        if "/" in stripped or "\\" in stripped:
            raise ValueError(
                f"Dataset input '{dataset}' for model '{self.catalog_key}' looks "
                "like a filesystem path. Use a server-known dataset name instead. "
                f"Valid datasets: {valid}."
            )
        normalized = normalize_dataset_name(dataset)
        for dataset_type in options:
            names = {
                dataset_name(dataset_type),
                dataset_name(dataset_type).lower(),
                dataset_cli_name(dataset_type),
            }
            if (
                dataset in names
                or dataset.lower() in names
                or normalized in names
            ):
                return dataset_type
        raise ValueError(
            f"Unknown dataset '{dataset}' for model '{self.catalog_key}'. "
            f"Valid datasets: {valid}."
        )

    def resolve_datasets(
        self,
        datasets: list[str] | None,
        experiment_task: str | ExperimentTask | None = None,
    ) -> list[type]:
        if not datasets:
            return [self.resolve_dataset(None, experiment_task)]
        resolved = [
            self.resolve_dataset(dataset, experiment_task) for dataset in datasets
        ]
        unique: list[type] = []
        seen: set[str] = set()
        for dataset in resolved:
            name = dataset_name(dataset)
            if name not in seen:
                seen.add(name)
                unique.append(dataset)
        return unique

    def monitor_options(self) -> list[MonitorOption]:
        options = list(self.monitor_metadata or [])
        invalid = [
            type(option).__name__
            for option in options
            if not isinstance(option, MonitorOption)
        ]
        if invalid:
            raise ValueError(
                f"Model package '{self.catalog_key}' has invalid MONITOR_OPTIONS "
                f"entries: {', '.join(invalid)}."
            )
        names = [option.name for option in options]
        duplicates = sorted(name for name in set(names) if names.count(name) > 1)
        if duplicates:
            raise ValueError(
                f"Model package '{self.catalog_key}' has duplicate monitor options: "
                f"{', '.join(duplicates)}."
            )
        return options

    def resolve_monitors(
        self,
        monitor_names: list[str] | None,
    ) -> list[MonitorOption]:
        if not monitor_names:
            return []
        options_by_name = {option.name: option for option in self.monitor_options()}
        selected: list[MonitorOption] = []
        seen: set[str] = set()
        unknown: list[str] = []
        for name in monitor_names:
            if name in seen:
                continue
            seen.add(name)
            option = options_by_name.get(name)
            if option is None:
                unknown.append(name)
            else:
                selected.append(option)
        if unknown:
            valid = ", ".join(sorted(options_by_name)) or "none"
            raise ValueError(
                f"Unknown monitor option(s) for model '{self.catalog_key}': "
                f"{', '.join(unknown)}. Valid monitors: {valid}."
            )
        return selected

    def configuration_field_metadata(
        self,
        *,
        include_search_space: bool = False,
    ) -> dict[str, dict[str, Any]]:
        module = (
            self.metadata.search_space_module
            if include_search_space
            else self.runtime_defaults
        )
        return configuration_field_metadata(
            module,
            include_search_space=include_search_space,
        )


ModelCatalogEntry = ModelPackage


def model_package_from_module_path(module_path: str) -> ModelPackage | None:
    """Derive the conventional package identity without importing a catalog.

    This is a source-compatibility fallback for direct construction of concrete
    Experiment classes. Project composition and Run execution pass the canonical
    catalog-owned :class:`ModelPackage` explicitly.
    """

    parts = module_path.split(".")
    if len(parts) < 3 or parts[0] != "models":
        return None
    return ModelPackage(parts[1], parts[2], ".".join(parts[:3]))


__all__ = [
    "ModelCatalogEntry",
    "ModelPackage",
    "model_package_from_module_path",
]

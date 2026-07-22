from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from threading import RLock
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, cast

from emperor.experiments import (
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
from model_runtime.packages.identity import ModelIdentity
from model_runtime.packages.inspection_limits import (
    DEFAULT_INSPECTION_CONSTRUCTION_LIMITS,
    InspectionConstructionLimits,
)
from model_runtime.packages.metadata import ModelMetadata

if TYPE_CHECKING:
    from emperor.config import ModelConfig
    from model_runtime.runs.artifacts import RunArtifacts


class _PackageAdapter(Protocol):
    """Package-local lazy operations consumed by :class:`ModelPackage`."""

    def load_metadata(self) -> ModelMetadata: ...

    def load_runtime_options_type(self) -> type: ...

    def bind_runtime_defaults(self, values: Mapping[str, object] | None) -> Any: ...

    def load_preset_type(self) -> type: ...

    def load_presets(self) -> Any: ...

    def build_configuration(
        self,
        presets: Any,
        preset: Any,
        dataset: type,
        **kwargs: Any,
    ) -> ModelConfig: ...

    def build_model(self, configuration: ModelConfig) -> Any: ...

    def build_experiment(
        self,
        preset: Any,
        *,
        experiment_task: ExperimentTask,
        model_package: ModelPackage,
        run_artifacts: RunArtifacts,
    ) -> Any: ...


_INITIALIZATION_MISSING = object()
_MODEL_PACKAGE_INITIALIZATION_LOCK = RLock()
_InitializationValue = TypeVar("_InitializationValue")


@dataclass(frozen=True)
class ModelPackage:
    """Canonical Interface for one isolated Model Package.

    Identity is data, while every implementation operation belongs to the
    selected package's lightweight adapter. Importing the catalog therefore
    never imports a concrete model, Runtime Defaults module, or training stack.
    """

    identity: ModelIdentity
    _adapter: _PackageAdapter = field(repr=False, compare=False)
    inspection_construction_limits: InspectionConstructionLimits = (
        DEFAULT_INSPECTION_CONSTRUCTION_LIMITS
    )

    def __post_init__(self) -> None:
        if not isinstance(self.identity, ModelIdentity):
            raise TypeError("ModelPackage identity must be a ModelIdentity.")
        if not isinstance(
            self.inspection_construction_limits,
            InspectionConstructionLimits,
        ):
            raise TypeError(
                "ModelPackage inspection limits must be InspectionConstructionLimits."
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
    def catalog_key(self) -> str:
        return self.identity.catalog_key

    def to_identity_payload(self) -> dict[str, str]:
        return self.identity.to_payload()

    @property
    def metadata(self) -> ModelMetadata:
        return self._initialize_once("_metadata", self._adapter.load_metadata)

    @property
    def runtime_defaults(self):
        return self.metadata.runtime_defaults

    @property
    def runtime_options_type(self) -> type:
        return self._initialize_once(
            "_runtime_options_type",
            self._adapter.load_runtime_options_type,
        )

    def bind_runtime_defaults(
        self,
        values: Mapping[str, object] | None = None,
    ) -> Any:
        if values is not None and not isinstance(values, Mapping):
            raise TypeError("Runtime Defaults values must be a mapping.")
        runtime = self._adapter.bind_runtime_defaults(values)
        if type(runtime) is not self.runtime_options_type:
            raise TypeError(
                f"Model Package '{self.catalog_key}' returned "
                f"{type(runtime).__name__}, expected "
                f"{self.runtime_options_type.__name__}."
            )
        return runtime

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
    def preset_type(self) -> type:
        return self._initialize_once("_preset_type", self._adapter.load_preset_type)

    @property
    def presets(self) -> Any:
        return self._initialize_once("_presets", self._adapter.load_presets)

    @property
    def default_preset(self) -> Any:
        return self.presets.default_preset

    def checkpoint_config_overrides(
        self,
        tensor_shapes: Mapping[str, tuple[int, ...]],
    ) -> dict[str, Any]:
        """Interpret package-owned checkpoint shapes when supported."""

        interpreter = getattr(self._adapter, "checkpoint_config_overrides", None)
        if interpreter is None:
            return {}
        if not callable(interpreter):
            raise ValueError(
                f"Model package '{self.catalog_key}' has an invalid checkpoint "
                "interpreter."
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

    def build_configuration(
        self,
        preset: Any = None,
        dataset: type | None = None,
        **kwargs: Any,
    ) -> ModelConfig:
        selected_preset = self.default_preset if preset is None else preset
        selected_dataset = self.resolve_dataset(None) if dataset is None else dataset
        return self._adapter.build_configuration(
            self.presets,
            selected_preset,
            selected_dataset,
            **kwargs,
        )

    def build_model(self, configuration: ModelConfig) -> Any:
        return self._adapter.build_model(configuration)

    def build_experiment(
        self,
        preset: Any,
        *,
        experiment_task: ExperimentTask,
        run_artifacts: RunArtifacts,
    ) -> Any:
        return self._adapter.build_experiment(
            preset,
            experiment_task=experiment_task,
            model_package=self,
            run_artifacts=run_artifacts,
        )

    def resolve_preset(self, preset_name: str):
        try:
            return self.preset_type.get_member(preset_name)
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
            description_for_preset(preset) if callable(description_for_preset) else None
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
            if dataset in names or dataset.lower() in names or normalized in names:
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

    def monitor_options(self):
        return list(self.monitor_metadata)

    def resolve_monitors(self, monitor_names: list[str] | None):
        if not monitor_names:
            return []
        options_by_name = {option.name: option for option in self.monitor_options()}
        selected = []
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
            self.metadata.search_space
            if include_search_space
            else self.runtime_defaults
        )
        return configuration_field_metadata(
            module,
            include_search_space=include_search_space,
        )


__all__ = ["ModelPackage"]

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType
from typing import Any

from model_runtime.packages import model_key, split_model_id

_METADATA_ERROR = "The project Adapter returned invalid Model Package metadata"


def _invalid_metadata(path: str, requirement: str) -> ValueError:
    return ValueError(f"{_METADATA_ERROR}: {path} {requirement}.")


def _required(
    payload: Mapping[str, Any],
    key: str,
    *,
    path: str,
) -> Any:
    if key not in payload:
        raise _invalid_metadata(f"{path}.{key}", "is required")
    return payload[key]


def _mapping(value: object, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _invalid_metadata(path, "must be an object")
    if any(not isinstance(key, str) for key in value):
        raise _invalid_metadata(path, "must use string keys")
    return value


def _list(value: object, *, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise _invalid_metadata(path, "must be a list")
    return value


def _string(value: object, *, path: str) -> str:
    if not isinstance(value, str):
        raise _invalid_metadata(path, "must be a string")
    return value


def _integer(value: object, *, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _invalid_metadata(path, "must be an integer")
    if value < 0:
        raise _invalid_metadata(path, "must be non-negative")
    return value


def _boolean(value: object, *, path: str) -> bool:
    if not isinstance(value, bool):
        raise _invalid_metadata(path, "must be a boolean")
    return value


def _freeze_json(value: object, *, path: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise _invalid_metadata(path, "must be finite")
        return value
    if isinstance(value, Mapping):
        mapping = _mapping(value, path=path)
        return MappingProxyType(
            {
                key: _freeze_json(item, path=f"{path}.{key}")
                for key, item in mapping.items()
            }
        )
    if isinstance(value, list):
        return tuple(
            _freeze_json(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    raise _invalid_metadata(path, "must be JSON-compatible")


def _preset(value: object, *, index: int) -> ModelPreset:
    path = f"presets[{index}]"
    payload = _mapping(value, path=path)
    return ModelPreset(
        name=_string(_required(payload, "name", path=path), path=f"{path}.name"),
        label=_string(
            _required(payload, "label", path=path),
            path=f"{path}.label",
        ),
        description=_string(
            _required(payload, "description", path=path),
            path=f"{path}.description",
        ),
    )


def _dataset(value: object, *, group_index: int, index: int) -> ModelDataset:
    path = f"dataset_groups[{group_index}].datasets[{index}]"
    payload = _mapping(value, path=path)
    return ModelDataset(
        name=_string(_required(payload, "name", path=path), path=f"{path}.name"),
        label=_string(
            _required(payload, "label", path=path),
            path=f"{path}.label",
        ),
        input_dim=_integer(
            _required(payload, "input_dim", path=path),
            path=f"{path}.input_dim",
        ),
        output_dim=_integer(
            _required(payload, "output_dim", path=path),
            path=f"{path}.output_dim",
        ),
    )


def _dataset_group(value: object, *, index: int) -> ModelDatasetGroup:
    path = f"dataset_groups[{index}]"
    payload = _mapping(value, path=path)
    datasets = _list(
        _required(payload, "datasets", path=path),
        path=f"{path}.datasets",
    )
    return ModelDatasetGroup(
        experiment_task=_string(
            _required(payload, "experiment_task", path=path),
            path=f"{path}.experiment_task",
        ),
        label=_string(
            _required(payload, "label", path=path),
            path=f"{path}.label",
        ),
        datasets=tuple(
            _dataset(dataset, group_index=index, index=dataset_index)
            for dataset_index, dataset in enumerate(datasets)
        ),
    )


def _monitor(value: object, *, index: int) -> ModelMonitor:
    path = f"monitors[{index}]"
    payload = _mapping(value, path=path)
    kinds = _list(
        _required(payload, "kinds", path=path),
        path=f"{path}.kinds",
    )
    return ModelMonitor(
        name=_string(_required(payload, "name", path=path), path=f"{path}.name"),
        label=_string(
            _required(payload, "label", path=path),
            path=f"{path}.label",
        ),
        description=_string(
            _required(payload, "description", path=path),
            path=f"{path}.description",
        ),
        kinds=tuple(
            _string(kind, path=f"{path}.kinds[{kind_index}]")
            for kind_index, kind in enumerate(kinds)
        ),
        default_enabled=_boolean(
            _required(payload, "defaultEnabled", path=path),
            path=f"{path}.defaultEnabled",
        ),
    )


@dataclass(frozen=True, slots=True)
class ModelPackageIdentity:
    model_type: str
    model: str

    def __post_init__(self) -> None:
        model_key(self.model_type, self.model)

    @property
    def catalog_key(self) -> str:
        return model_key(self.model_type, self.model)

    @classmethod
    def from_id(cls, model_id: str) -> ModelPackageIdentity:
        identity = split_model_id(model_id)
        if identity is None:
            raise ValueError(f"Invalid model id: {model_id}")
        return cls(identity.model_type, identity.model)

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
    ) -> ModelPackageIdentity | None:
        model_type = payload.get("modelType")
        if not isinstance(model_type, str):
            model_type = payload.get("model_type")
        model = payload.get("model")
        if isinstance(model_type, str) and isinstance(model, str):
            try:
                return cls(model_type, model)
            except ValueError:
                return None
        if not isinstance(model, str):
            return None
        identity = split_model_id(model)
        if identity is None:
            return None
        return cls(identity.model_type, identity.model)


@dataclass(frozen=True, slots=True)
class ModelPreset:
    name: str
    label: str
    description: str


@dataclass(frozen=True, slots=True)
class ModelDataset:
    name: str
    label: str
    input_dim: int
    output_dim: int


@dataclass(frozen=True, slots=True)
class ModelDatasetGroup:
    experiment_task: str
    label: str
    datasets: tuple[ModelDataset, ...]


@dataclass(frozen=True, slots=True)
class ModelMonitor:
    name: str
    label: str
    description: str
    kinds: tuple[str, ...]
    default_enabled: bool = False


@dataclass(frozen=True, slots=True)
class ModelMetadata:
    default_experiment_task: str
    presets: tuple[ModelPreset, ...]
    dataset_groups: tuple[ModelDatasetGroup, ...]
    monitors: tuple[ModelMonitor, ...]
    runtime_defaults: Mapping[str, Any]

    def __post_init__(self) -> None:
        frozen_defaults = _freeze_json(
            self.runtime_defaults,
            path="runtime_defaults",
        )
        if not isinstance(frozen_defaults, Mapping):
            raise _invalid_metadata("runtime_defaults", "must be an object")
        object.__setattr__(
            self,
            "runtime_defaults",
            frozen_defaults,
        )


def metadata_from_mapping(value: object) -> ModelMetadata:
    payload = _mapping(value, path="metadata")
    raw_presets = _list(
        _required(payload, "presets", path="metadata"),
        path="presets",
    )
    raw_groups = _list(
        _required(payload, "dataset_groups", path="metadata"),
        path="dataset_groups",
    )
    raw_monitors = _list(
        _required(payload, "monitors", path="metadata"),
        path="monitors",
    )
    raw_defaults = _mapping(
        _required(payload, "runtime_defaults", path="metadata"),
        path="runtime_defaults",
    )
    return ModelMetadata(
        default_experiment_task=_string(
            _required(payload, "default_experiment_task", path="metadata"),
            path="default_experiment_task",
        ),
        presets=tuple(
            _preset(preset, index=index) for index, preset in enumerate(raw_presets)
        ),
        dataset_groups=tuple(
            _dataset_group(group, index=index) for index, group in enumerate(raw_groups)
        ),
        monitors=tuple(
            _monitor(monitor, index=index) for index, monitor in enumerate(raw_monitors)
        ),
        runtime_defaults=raw_defaults,
    )


__all__ = [
    "ModelDataset",
    "ModelDatasetGroup",
    "ModelMetadata",
    "ModelMonitor",
    "ModelPackageIdentity",
    "ModelPreset",
]

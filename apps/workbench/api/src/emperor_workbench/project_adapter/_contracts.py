from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from model_runtime.cli import json_value_to_wire
from model_runtime.packages import ModelIdentity

from emperor_workbench.project_adapter._wire import (
    ProjectAdapterProtocolFailure,
    require_field,
    require_list,
    require_mapping,
    require_string,
)

if TYPE_CHECKING:
    from emperor_workbench.project_adapter._client import ProjectAdapterClient


@dataclass(frozen=True, slots=True)
class PresetReference:
    name: str
    public_name: str


@dataclass(frozen=True, slots=True)
class DatasetReference:
    __name__: str


@dataclass(frozen=True, slots=True)
class MonitorReference:
    name: str


@dataclass(frozen=True, slots=True)
class ModelPackageReference:
    model_type: str
    model: str
    client: ProjectAdapterClient

    def __post_init__(self) -> None:
        try:
            _identity = self.identity
        except ValueError as exc:
            raise ProjectAdapterProtocolFailure(
                "The project Adapter Model Package identity is invalid."
            ) from exc

    @property
    def catalog_key(self) -> str:
        return self.identity.catalog_key

    @property
    def identity(self) -> ModelIdentity:
        return ModelIdentity(self.model_type, self.model)

    def metadata_payload(self) -> dict[str, Any]:
        return self.client._package_metadata(self.catalog_key)

    @property
    def runtime_defaults(self) -> SimpleNamespace:
        metadata = self.metadata_payload()
        runtime_defaults = require_mapping(require_field(metadata, "runtime_defaults"))
        try:
            return SimpleNamespace(**runtime_defaults)
        except TypeError as exc:
            raise ProjectAdapterProtocolFailure(
                "The project Adapter runtime defaults are invalid."
            ) from exc

    def resolve_experiment_task(self, value: str | None) -> str:
        resolution = self._resolve(experiment_task=value)
        return require_string(require_field(resolution, "experiment_task"))

    def task_name(self, task: object) -> str:
        return str(task)

    def resolve_datasets(
        self,
        values: list[str],
        task: object,
    ) -> list[DatasetReference]:
        resolution = self._resolve(experiment_task=str(task), datasets=values)
        dataset_names = require_list(require_field(resolution, "datasets"))
        return [DatasetReference(require_string(name)) for name in dataset_names]

    def resolve_preset(self, value: str) -> PresetReference:
        resolution = self._resolve(presets=[value])
        resolved_presets = require_list(require_field(resolution, "presets"))
        if len(resolved_presets) != 1:
            raise ProjectAdapterProtocolFailure(
                "The project Adapter preset result is invalid."
            )
        preset_payload = require_mapping(resolved_presets[0])
        return PresetReference(
            name=require_string(require_field(preset_payload, "key")),
            public_name=require_string(require_field(preset_payload, "name")),
        )

    def preset_name(self, preset: PresetReference) -> str:
        return preset.public_name

    def resolve_monitors(self, values: list[str] | None) -> list[MonitorReference]:
        resolution = self._resolve(monitors=list(values or ()))
        monitor_names = require_list(require_field(resolution, "monitors"))
        return [MonitorReference(require_string(name)) for name in monitor_names]

    def checkpoint_config_overrides(
        self,
        tensor_shapes: Mapping[str, tuple[int, ...]],
    ) -> dict[str, Any]:
        return require_mapping(
            self.client.call(
                "checkpoint_config_overrides",
                {
                    "model_id": self.catalog_key,
                    "tensor_shapes": json_value_to_wire(tensor_shapes),
                },
            )
        )

    def _resolve(self, **values: Any) -> dict[str, Any]:
        return require_mapping(
            self.client.call(
                "resolve",
                {"model_id": self.catalog_key, **values},
            )
        )

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from model_runtime.cli import to_wire
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

    @property
    def catalog_key(self) -> str:
        return f"{self.model_type}/{self.model}"

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
        result = self._resolve(experiment_task=value)
        return require_string(require_field(result, "experiment_task"))

    def task_name(self, task: object) -> str:
        return str(task)

    def resolve_datasets(
        self,
        values: list[str],
        task: object,
    ) -> list[DatasetReference]:
        result = self._resolve(experiment_task=str(task), datasets=values)
        datasets = require_list(require_field(result, "datasets"))
        return [DatasetReference(require_string(name)) for name in datasets]

    def resolve_preset(self, value: str) -> PresetReference:
        result = self._resolve(presets=[value])
        presets = require_list(require_field(result, "presets"))
        if len(presets) != 1:
            raise ProjectAdapterProtocolFailure(
                "The project Adapter preset result is invalid."
            )
        preset = require_mapping(presets[0])
        return PresetReference(
            name=require_string(require_field(preset, "key")),
            public_name=require_string(require_field(preset, "name")),
        )

    def preset_name(self, preset: PresetReference) -> str:
        return preset.public_name

    def resolve_monitors(self, values: list[str] | None) -> list[MonitorReference]:
        result = self._resolve(monitors=list(values or ()))
        monitors = require_list(require_field(result, "monitors"))
        return [MonitorReference(require_string(name)) for name in monitors]

    def checkpoint_config_overrides(
        self,
        tensor_shapes: Mapping[str, tuple[int, ...]],
    ) -> dict[str, Any]:
        return require_mapping(
            self.client.call(
                "checkpoint_config_overrides",
                {
                    "model_id": self.catalog_key,
                    "tensor_shapes": to_wire(tensor_shapes),
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

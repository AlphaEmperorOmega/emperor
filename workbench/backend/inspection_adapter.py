from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, ParamSpec, TypeVar

from model_runtime.inspection import (
    ConfigurationSchema,
    InspectionRequest,
    InspectionResult,
    ParsedOverrides,
    SearchSpace,
)
from workbench.backend.inspection_errors import InspectionFailure
from workbench.backend.project_adapter import (
    ModelPackageReference,
    ProjectAdapterFailure,
    project_adapter,
)

P = ParamSpec("P")
T = TypeVar("T")


def _inspection_failure(exc: ProjectAdapterFailure) -> InspectionFailure:
    return InspectionFailure(exc.detail, kind=exc.kind)


@dataclass(frozen=True)
class WorkbenchInspectionAdapter:
    """Workbench projection for exactly one remote Model Package."""

    package: ModelPackageReference

    @classmethod
    def select(cls, model_id: str) -> WorkbenchInspectionAdapter:
        try:
            return cls(project_adapter().package(model_id))
        except ProjectAdapterFailure as exc:
            raise _inspection_failure(exc) from exc

    @classmethod
    def select_parts(
        cls,
        model_type: str,
        model: str,
    ) -> WorkbenchInspectionAdapter:
        return cls.select(f"{model_type}/{model}")

    @classmethod
    def from_package(cls, package: object) -> WorkbenchInspectionAdapter:
        if isinstance(package, ModelPackageReference):
            return cls(package)
        model_id = getattr(package, "catalog_key", None)
        if not isinstance(model_id, str):
            raise InspectionFailure("Inspection requires a selected Model Package.")
        return cls.select(model_id)

    @staticmethod
    def catalog_payload() -> list[dict[str, str]]:
        try:
            return [
                package.identity.to_payload()
                for package in project_adapter().catalog()
            ]
        except ProjectAdapterFailure as exc:
            raise _inspection_failure(exc) from exc

    def call_package(
        self,
        call: Callable[P, T],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> T:
        try:
            return call(self.package, *args, **kwargs)
        except ProjectAdapterFailure as exc:
            raise _inspection_failure(exc) from exc

    def inspect(self, request: InspectionRequest) -> InspectionResult:
        try:
            return self.package.client.inspect(self.package.catalog_key, request)
        except ProjectAdapterFailure as exc:
            raise _inspection_failure(exc) from exc

    def inspect_payload(self, request: InspectionRequest) -> dict[str, Any]:
        from workbench.backend.inspection_serialization import inspection_result_payload

        return inspection_result_payload(self.inspect(request))

    def configuration(self, preset: str | None = None) -> ConfigurationSchema:
        try:
            return self.package.client.configuration(self.package.catalog_key, preset)
        except ProjectAdapterFailure as exc:
            raise _inspection_failure(exc) from exc

    def configuration_payload(self, preset: str | None = None) -> dict[str, Any]:
        from workbench.backend.inspection_serialization import (
            configuration_schema_payload,
        )

        return configuration_schema_payload(self.configuration(preset))

    def search_space(
        self,
        preset: str | None = None,
        presets: list[str] | tuple[str, ...] | None = None,
    ) -> SearchSpace:
        try:
            return self.package.client.search_space(
                self.package.catalog_key,
                preset,
                presets,
            )
        except ProjectAdapterFailure as exc:
            raise _inspection_failure(exc) from exc

    def search_space_payload(
        self,
        preset: str | None = None,
        presets: list[str] | tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        from workbench.backend.inspection_serialization import search_space_payload

        return search_space_payload(self.search_space(preset, presets))

    def parse_overrides(
        self,
        overrides: Mapping[str, Any] | None,
        *,
        preset: str | None = None,
        ignore_unknown: bool = False,
    ) -> ParsedOverrides:
        try:
            result = self.package.client.call(
                "parse_overrides",
                {
                    "model_id": self.package.catalog_key,
                    "overrides": dict(overrides or {}),
                    "preset": preset,
                    "ignore_unknown": ignore_unknown,
                },
            )
        except ProjectAdapterFailure as exc:
            raise _inspection_failure(exc) from exc
        if not isinstance(result, dict):
            raise InspectionFailure("The project Adapter returned invalid overrides.")
        return ParsedOverrides(result)

    def validate(self, request: InspectionRequest) -> None:
        raw_overrides = (
            request.overrides.values
            if isinstance(request.overrides, ParsedOverrides)
            else request.overrides
        )
        try:
            self.package.client.call(
                "validate",
                {
                    "model_id": self.package.catalog_key,
                    "preset": request.preset,
                    "overrides": dict(raw_overrides),
                    "dataset": request.dataset,
                    "experiment_task": request.experiment_task,
                },
            )
        except ProjectAdapterFailure as exc:
            raise _inspection_failure(exc) from exc

    def preset_locks(self, preset: str | None) -> dict[str, Any]:
        try:
            result = self.package.client.call(
                "preset_locks",
                {"model_id": self.package.catalog_key, "preset": preset},
            )
        except ProjectAdapterFailure as exc:
            raise _inspection_failure(exc) from exc
        return dict(result or {})

    def reject_locked_overrides(
        self,
        preset: str,
        overrides: Mapping[str, Any],
    ) -> None:
        try:
            self.package.client.call(
                "reject_locked_overrides",
                {
                    "model_id": self.package.catalog_key,
                    "preset": preset,
                    "overrides": dict(overrides),
                },
            )
        except ProjectAdapterFailure as exc:
            raise _inspection_failure(exc) from exc

    def serialize_overrides(
        self,
        overrides: Mapping[str, Any] | None,
        *,
        ignore_unknown: bool = False,
    ) -> dict[str, Any]:
        try:
            result = self.package.client.call(
                "serialize_overrides",
                {
                    "model_id": self.package.catalog_key,
                    "overrides": dict(overrides or {}),
                    "ignore_unknown": ignore_unknown,
                },
            )
        except ProjectAdapterFailure as exc:
            raise _inspection_failure(exc) from exc
        return dict(result or {})

    def presets_payload(self) -> list[dict[str, str]]:
        return [
            {
                "name": str(preset["name"]),
                "label": str(preset["label"]),
                "description": str(preset["description"]),
            }
            for preset in self._metadata_payload()["presets"]
        ]

    def datasets_payload(self) -> dict[str, Any]:
        metadata = self._metadata_payload()
        return {
            "defaultExperimentTask": metadata["default_experiment_task"],
            "datasetGroups": [
                {
                    "experimentTask": group["experiment_task"],
                    "label": group["label"],
                    "datasets": [
                        {
                            "name": dataset["name"],
                            "label": dataset["label"],
                            "inputDim": dataset["input_dim"],
                            "outputDim": dataset["output_dim"],
                        }
                        for dataset in group["datasets"]
                    ],
                }
                for group in metadata["dataset_groups"]
            ],
        }

    def monitors_payload(self) -> list[dict[str, Any]]:
        return list(self._metadata_payload()["monitors"])

    def _metadata_payload(self) -> dict[str, Any]:
        try:
            return self.package.metadata_payload()
        except ProjectAdapterFailure as exc:
            raise _inspection_failure(exc) from exc


def inspect_module_graph_payload(module: object):
    """Legacy raw-module graph projection retained for local tooling tests."""

    from model_runtime.inspection import inspect_model_graph
    from workbench.backend.inspection_serialization import model_graph_payload

    return model_graph_payload(inspect_model_graph(module))


__all__ = ["WorkbenchInspectionAdapter", "inspect_module_graph_payload"]

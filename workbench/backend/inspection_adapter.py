"""Canonical Workbench Adapter for selected Model Package Inspection.

The semantic records stay owned by :mod:`emperor.inspection`. This Adapter owns
the repeated Workbench sequence around them: select one Model Package, map its
failures to the stable Workbench error, and serialize only at transport-facing
call sites.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

from emperor.model_packages import (
    ModelPackage,
    discover_model_packages,
    model_id_from_parts,
    model_package,
)

from workbench.backend.inspection_errors import (
    InspectionFailure,
    call_inspection,
    call_model_package,
)

if TYPE_CHECKING:
    from emperor.inspection import (
        ConfigurationSchema,
        InspectionRequest,
        InspectionResult,
        ParsedOverrides,
        SearchSpace,
    )
    from torch.nn import Module

P = ParamSpec("P")
T = TypeVar("T")


@dataclass(frozen=True)
class WorkbenchInspectionAdapter:
    """Workbench adaptation for exactly one selected Model Package."""

    package: ModelPackage

    @classmethod
    def select(cls, model_id: str) -> WorkbenchInspectionAdapter:
        package = model_package(model_id)
        if package is None:
            raise InspectionFailure(f"Unknown model: {model_id}")
        return cls(package)

    @classmethod
    def select_parts(
        cls,
        model_type: str,
        model: str,
    ) -> WorkbenchInspectionAdapter:
        model_id = model_id_from_parts(model_type, model)
        if model_id is None:
            raise InspectionFailure(
                f"Unknown model: --model-type {model_type} --model {model}"
            )
        return cls.select(model_id)

    @classmethod
    def from_package(cls, package: ModelPackage) -> WorkbenchInspectionAdapter:
        return cls(package)

    @staticmethod
    def catalog_payload() -> list[dict[str, str]]:
        return [package.identity.to_payload() for package in discover_model_packages()]

    def call_package(
        self,
        call: Callable[P, T],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> T:
        return call_model_package(self.package, call, *args, **kwargs)

    def inspect(self, request: InspectionRequest) -> InspectionResult:
        from emperor.inspection import inspect_model

        return call_inspection(inspect_model, self.package, request)

    def inspect_payload(self, request: InspectionRequest) -> dict[str, Any]:
        from workbench.backend.inspection_serialization import (
            inspection_result_payload,
        )

        return inspection_result_payload(self.inspect(request))

    def configuration(self, preset: str | None = None) -> ConfigurationSchema:
        from emperor.inspection import configuration_schema

        return call_inspection(configuration_schema, self.package, preset)

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
        from emperor.inspection import search_space_schema

        return call_inspection(
            search_space_schema,
            self.package,
            preset,
            presets,
        )

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
        from emperor.inspection import parse_overrides

        return call_inspection(
            parse_overrides,
            self.package,
            overrides,
            preset=preset,
            ignore_unknown=ignore_unknown,
        )

    def validate(self, request: InspectionRequest) -> None:
        from emperor.inspection import validate_configuration

        call_inspection(validate_configuration, self.package, request)

    def preset_locks(self, preset: str | None) -> dict[str, Any]:
        from emperor.inspection import preset_locks

        return call_inspection(preset_locks, self.package, preset)

    def reject_locked_overrides(
        self,
        preset: str,
        overrides: Mapping[str, Any],
    ) -> None:
        from emperor.inspection import reject_locked_overrides

        call_inspection(
            reject_locked_overrides,
            self.package,
            preset,
            overrides,
        )

    def serialize_overrides(
        self,
        overrides: Mapping[str, Any] | None,
        *,
        ignore_unknown: bool = False,
    ) -> dict[str, Any]:
        from emperor.inspection import serialize_overrides

        return call_inspection(
            serialize_overrides,
            self.package,
            overrides,
            ignore_unknown=ignore_unknown,
        )

    def presets_payload(self) -> list[dict[str, str]]:
        from workbench.backend.inspection_serialization import model_presets_payload

        return self.call_package(model_presets_payload, self.package)

    def datasets_payload(self) -> dict[str, Any]:
        from workbench.backend.inspection_serialization import model_datasets_payload

        return self.call_package(model_datasets_payload, self.package)

    def monitors_payload(self) -> list[dict[str, Any]]:
        from workbench.backend.inspection_serialization import model_monitors_payload

        return self.call_package(model_monitors_payload, self.package)


def inspect_module_graph_payload(
    module: Module,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Preserve the legacy raw-Module graph Adapter through canonical mapping."""

    from emperor.inspection import inspect_model_graph

    from workbench.backend.inspection_serialization import model_graph_payload

    return model_graph_payload(inspect_model_graph(module))


__all__ = ["WorkbenchInspectionAdapter", "inspect_module_graph_payload"]

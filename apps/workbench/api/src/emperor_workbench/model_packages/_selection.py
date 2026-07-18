from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, ParamSpec, TypeVar

from model_runtime.inspection import (
    ConfigurationSchema,
    InspectionRequest,
    InspectionResult,
    ParsedOverrides,
    SearchSpace,
)

from emperor_workbench.model_packages._errors import ModelPackageFailure
from emperor_workbench.model_packages._records import (
    ModelMetadata,
    ModelPackageIdentity,
    metadata_from_mapping,
)
from emperor_workbench.project_adapter import (
    ModelPackageReference,
    ProjectAdapterFailure,
)

P = ParamSpec("P")
T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class SelectedModelPackage:
    """Semantic Workbench Interface for exactly one selected Model Package."""

    reference: ModelPackageReference

    @property
    def identity(self) -> ModelPackageIdentity:
        return ModelPackageIdentity(
            self.reference.model_type,
            self.reference.model,
        )

    @property
    def catalog_key(self) -> str:
        return self.reference.catalog_key

    @property
    def runtime_defaults(self) -> SimpleNamespace:
        return self.reference.runtime_defaults

    def metadata(self) -> ModelMetadata:
        return self._call(
            lambda: metadata_from_mapping(self.reference.metadata_payload())
        )

    def configuration(self, preset: str | None = None) -> ConfigurationSchema:
        return self._call(
            self.reference.client.configuration,
            self.catalog_key,
            preset,
        )

    def search_space(
        self,
        preset: str | None = None,
        presets: list[str] | tuple[str, ...] | None = None,
    ) -> SearchSpace:
        return self._call(
            self.reference.client.search_space,
            self.catalog_key,
            preset,
            presets,
        )

    def inspect(self, request: InspectionRequest) -> InspectionResult:
        return self._call(
            self.reference.client.inspect,
            self.catalog_key,
            request,
        )

    def parse_overrides(
        self,
        overrides: Mapping[str, Any] | None,
        *,
        preset: str | None = None,
        ignore_unknown: bool = False,
    ) -> ParsedOverrides:
        parsed_overrides_payload = self._project_call(
            "parse_overrides",
            {
                "model_id": self.catalog_key,
                "overrides": dict(overrides or {}),
                "preset": preset,
                "ignore_unknown": ignore_unknown,
            },
        )
        if not isinstance(parsed_overrides_payload, dict):
            raise ModelPackageFailure("The project Adapter returned invalid overrides.")
        return ParsedOverrides(parsed_overrides_payload)

    def validate(self, request: InspectionRequest) -> None:
        raw_overrides = (
            request.overrides.values
            if isinstance(request.overrides, ParsedOverrides)
            else request.overrides
        )
        self._project_call(
            "validate",
            {
                "model_id": self.catalog_key,
                "preset": request.preset,
                "overrides": dict(raw_overrides),
                "dataset": request.dataset,
                "experiment_task": request.experiment_task,
            },
        )

    def preset_locks(self, preset: str | None) -> dict[str, Any]:
        preset_locks_payload = self._project_call(
            "preset_locks",
            {"model_id": self.catalog_key, "preset": preset},
        )
        return self._mapping_result(preset_locks_payload, name="preset locks")

    def reject_locked_overrides(
        self,
        preset: str,
        overrides: Mapping[str, Any],
    ) -> None:
        self._project_call(
            "reject_locked_overrides",
            {
                "model_id": self.catalog_key,
                "preset": preset,
                "overrides": dict(overrides),
            },
        )

    def serialize_overrides(
        self,
        overrides: Mapping[str, Any] | None,
        *,
        ignore_unknown: bool = False,
    ) -> dict[str, Any]:
        serialized_overrides_payload = self._project_call(
            "serialize_overrides",
            {
                "model_id": self.catalog_key,
                "overrides": dict(overrides or {}),
                "ignore_unknown": ignore_unknown,
            },
        )
        return self._mapping_result(
            serialized_overrides_payload,
            name="serialized overrides",
        )

    def checkpoint_config_overrides(
        self,
        tensor_shapes: Mapping[str, tuple[int, ...]],
    ) -> dict[str, Any]:
        return self._mapping_result(
            self._call(
                self.reference.checkpoint_config_overrides,
                tensor_shapes,
            ),
            name="checkpoint config overrides",
        )

    @staticmethod
    def _mapping_result(value: object, *, name: str) -> dict[str, Any]:
        if not isinstance(value, Mapping) or any(
            not isinstance(key, str) for key in value
        ):
            raise ModelPackageFailure(f"The project Adapter returned invalid {name}.")
        return dict(value)

    def _project_call(
        self,
        operation: str,
        payload: dict[str, Any],
    ) -> Any:
        return self._call(self.reference.client.call, operation, payload)

    @staticmethod
    def _call(
        call: Callable[P, T],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> T:
        try:
            return call(*args, **kwargs)
        except ProjectAdapterFailure as exc:
            raise ModelPackageFailure(exc.detail, kind=exc.kind) from exc
        except ValueError as exc:
            raise ModelPackageFailure(str(exc)) from exc


__all__ = ["SelectedModelPackage"]

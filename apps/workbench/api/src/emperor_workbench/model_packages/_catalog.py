from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from model_runtime.packages import model_key

from emperor_workbench.model_packages._errors import ModelPackageFailure
from emperor_workbench.model_packages._records import ModelPackageIdentity
from emperor_workbench.model_packages._selection import SelectedModelPackage
from emperor_workbench.project_adapter import (
    ProjectAdapterClient,
    ProjectAdapterFailure,
)


class ModelPackageCatalog:
    """Discover and select Model Packages through one injected project Adapter."""

    def __init__(self, project_adapter: ProjectAdapterClient) -> None:
        self._project_adapter = project_adapter

    def identities(self) -> tuple[ModelPackageIdentity, ...]:
        try:
            return tuple(
                ModelPackageIdentity(package.model_type, package.model)
                for package in self._project_adapter.catalog()
            )
        except ProjectAdapterFailure as exc:
            raise ModelPackageFailure(exc.detail, kind=exc.kind) from exc

    def select(self, model_id: str) -> SelectedModelPackage:
        try:
            return SelectedModelPackage(self._project_adapter.package(model_id))
        except ProjectAdapterFailure as exc:
            raise ModelPackageFailure(exc.detail, kind=exc.kind) from exc

    def select_parts(self, model_type: str, model: str) -> SelectedModelPackage:
        try:
            model_id = model_key(model_type, model)
        except ValueError as exc:
            raise ModelPackageFailure(str(exc)) from exc
        return self.select(model_id)

    def require_id(self, model_type: str, model: str) -> str:
        try:
            model_id = model_key(model_type, model)
            self.select(model_id)
        except (ModelPackageFailure, ValueError) as exc:
            raise ModelPackageFailure(
                f"Unknown model: --model-type {model_type} --model {model}"
            ) from exc
        return model_id

    def model_id_from_mapping(
        self,
        payload: Mapping[str, Any],
    ) -> str | None:
        identity = ModelPackageIdentity.from_mapping(payload)
        return identity.catalog_key if identity is not None else None

    def identity_lookup(self) -> Mapping[str, str]:
        """Return one finite canonical lookup for accepted identity tokens."""

        return MappingProxyType(
            {
                identity.catalog_key: identity.catalog_key
                for identity in self.identities()
            }
        )


__all__ = ["ModelPackageCatalog"]

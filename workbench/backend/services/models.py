"""Model catalog and model-schema use cases."""

from __future__ import annotations

from typing import Any

from models.catalog import (
    discover_model_identity_payloads,
    model_identity_payload_from_id,
)

from workbench.backend.model_identity import require_model_id


class ModelCatalogService:
    def list_models(self) -> list[dict[str, str]]:
        return discover_model_identity_payloads()

    def list_presets(self, model_type: str, model: str) -> list[dict[str, Any]]:
        from workbench.backend.inspector.discovery import list_model_presets

        return list_model_presets(require_model_id(model_type, model))

    def list_datasets(self, model_type: str, model: str) -> dict[str, Any]:
        from workbench.backend.inspector.discovery import list_model_datasets

        return list_model_datasets(require_model_id(model_type, model))

    def list_monitors(self, model_type: str, model: str) -> list[dict[str, Any]]:
        from workbench.backend.inspector.discovery import list_model_monitors

        return list_model_monitors(require_model_id(model_type, model))

    def config_schema(
        self,
        model_type: str,
        model: str,
        preset: str | None,
    ) -> dict[str, Any]:
        from workbench.backend.inspector.schema import config_schema

        return config_schema(require_model_id(model_type, model), preset)

    def search_space_schema(
        self,
        model_type: str,
        model: str,
        preset: str | None,
        presets: list[str] | None = None,
    ) -> dict[str, Any]:
        from workbench.backend.inspector.schema import search_space_schema

        return search_space_schema(require_model_id(model_type, model), preset, presets)


def identity_payload(model_id: str) -> dict[str, str]:
    return model_identity_payload_from_id(model_id)

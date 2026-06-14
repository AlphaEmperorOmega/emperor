"""Model catalog and model-schema use cases."""

from __future__ import annotations

from typing import Any


class ModelCatalogService:
    def list_models(self) -> list[str]:
        from viewer.backend.inspector.discovery import discover_models

        return discover_models()

    def list_presets(self, model: str) -> list[dict[str, Any]]:
        from viewer.backend.inspector.discovery import list_model_presets

        return list_model_presets(model)

    def list_datasets(self, model: str) -> list[dict[str, Any]]:
        from viewer.backend.inspector.discovery import list_model_datasets

        return list_model_datasets(model)

    def list_monitors(self, model: str) -> list[dict[str, Any]]:
        from viewer.backend.inspector.discovery import list_model_monitors

        return list_model_monitors(model)

    def config_schema(self, model: str, preset: str | None) -> dict[str, Any]:
        from viewer.backend.inspector.schema import config_schema

        return config_schema(model, preset)

    def search_space_schema(self, model: str, preset: str | None) -> dict[str, Any]:
        from viewer.backend.inspector.schema import search_space_schema

        return search_space_schema(model, preset)

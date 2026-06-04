"""Model catalog and model-schema use cases."""

from __future__ import annotations

from typing import Any

from viewer.backend.inspector.discovery import (
    discover_models,
    list_model_datasets,
    list_model_monitors,
    list_model_presets,
)
from viewer.backend.inspector.schema import config_schema, search_space_schema


class ModelCatalogService:
    def list_models(self) -> list[str]:
        return discover_models()

    def list_presets(self, model: str) -> list[dict[str, Any]]:
        return list_model_presets(model)

    def list_datasets(self, model: str) -> list[dict[str, Any]]:
        return list_model_datasets(model)

    def list_monitors(self, model: str) -> list[dict[str, Any]]:
        return list_model_monitors(model)

    def config_schema(self, model: str, preset: str | None) -> dict[str, Any]:
        return config_schema(model, preset)

    def search_space_schema(self, model: str, preset: str | None) -> dict[str, Any]:
        return search_space_schema(model, preset)

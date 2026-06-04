"""Model inspection use cases."""

from __future__ import annotations

from typing import Any

from viewer.backend.inspector.service import inspect_model


class InspectionService:
    def inspect(
        self,
        *,
        model: str,
        preset: str,
        overrides: dict[str, Any],
        dataset: str | None,
    ) -> dict[str, Any]:
        return inspect_model(
            model,
            preset,
            overrides,
            dataset=dataset,
        )

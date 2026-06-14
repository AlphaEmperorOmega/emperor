"""Model inspection use cases."""

from __future__ import annotations

from typing import Any


class InspectionService:
    def inspect(
        self,
        *,
        model: str,
        preset: str,
        overrides: dict[str, Any],
        dataset: str | None,
    ) -> dict[str, Any]:
        from viewer.backend.inspector.service import inspect_model

        return inspect_model(
            model,
            preset,
            overrides,
            dataset=dataset,
        )

    def inspect_operation_graph(
        self,
        *,
        model: str,
        preset: str,
        overrides: dict[str, Any],
        dataset: str | None,
    ) -> dict[str, Any]:
        from viewer.backend.inspector.operation_graph import inspect_operation_graph

        return inspect_operation_graph(
            model,
            preset,
            overrides,
            dataset=dataset,
        )

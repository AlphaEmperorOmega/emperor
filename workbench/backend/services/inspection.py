"""Model inspection use cases."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from workbench.backend.inspection_adapter import WorkbenchInspectionAdapter
from workbench.backend.inspector.errors import InspectorError
from workbench.backend.run_history import HistoricalInspectionSource

if TYPE_CHECKING:
    from workbench.backend.inspector.checkpoint_shapes import CheckpointLoadBudgets


class InspectionService:
    def __init__(
        self,
        historical_runs: HistoricalInspectionSource | None = None,
        *,
        checkpoint_load_budgets: CheckpointLoadBudgets | None = None,
    ) -> None:
        self._historical_runs = historical_runs
        self._checkpoint_load_budgets = checkpoint_load_budgets

    def inspect(
        self,
        *,
        model_type: str,
        model: str,
        preset: str,
        overrides: dict[str, Any],
        dataset: str | None,
        experiment_task: str | None = None,
        log_run_id: str | None = None,
    ) -> dict[str, Any]:
        from emperor.inspection import InspectionRequest

        adapter = WorkbenchInspectionAdapter.select_parts(model_type, model)
        parsed_request_overrides = adapter.parse_overrides(overrides)
        if log_run_id is not None:
            from workbench.backend.historical_inspection import (
                HistoricalInspectionRequest,
                WorkbenchHistoricalInspection,
            )

            if self._historical_runs is None:
                raise InspectorError("Log run inspection is not configured.")
            context = self._historical_runs.inspection_context(log_run_id)
            historical_inspection = (
                WorkbenchHistoricalInspection(adapter.package)
                if self._checkpoint_load_budgets is None
                else WorkbenchHistoricalInspection(
                    adapter.package,
                    checkpoint_budgets=self._checkpoint_load_budgets,
                )
            )
            return historical_inspection.inspect_payload(
                context,
                HistoricalInspectionRequest(
                    preset=preset,
                    request_overrides=parsed_request_overrides.values,
                    dataset=dataset,
                    experiment_task=experiment_task,
                ),
            )
        return adapter.inspect_payload(
            InspectionRequest(
                preset=preset,
                overrides=parsed_request_overrides,
                dataset=dataset,
                experiment_task=experiment_task,
            )
        )


__all__ = ["InspectionService"]

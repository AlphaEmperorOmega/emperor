"""Model inspection use cases."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from emperor.inspection import InspectionResult

from workbench.backend.inspection_adapter import WorkbenchInspectionAdapter
from workbench.backend.inspection_errors import InspectionFailure
from workbench.backend.inspection_worker import (
    InProcessInspectionExecutor,
    InspectionExecutor,
)
from workbench.backend.run_history import HistoricalInspectionSource

if TYPE_CHECKING:
    from workbench.backend.inspector.checkpoint_shapes import CheckpointLoadBudgets


class InspectionService:
    def __init__(
        self,
        historical_runs: HistoricalInspectionSource | None = None,
        *,
        checkpoint_load_budgets: CheckpointLoadBudgets | None = None,
        executor: InspectionExecutor | None = None,
    ) -> None:
        self._historical_runs = historical_runs
        self._checkpoint_load_budgets = checkpoint_load_budgets
        self._executor = executor or InProcessInspectionExecutor()

    @property
    def executor(self) -> InspectionExecutor:
        return self._executor

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
    ) -> InspectionResult:
        from emperor.inspection import InspectionRequest

        adapter = WorkbenchInspectionAdapter.select_parts(model_type, model)
        parsed_request_overrides = adapter.parse_overrides(overrides)
        if log_run_id is not None:
            from workbench.backend.historical_inspection import (
                HistoricalInspectionRequest,
                WorkbenchHistoricalInspection,
            )

            if self._historical_runs is None:
                raise InspectionFailure("Log run inspection is not configured.")
            context = self._historical_runs.inspection_context(log_run_id)
            if self._checkpoint_load_budgets is None:
                historical_inspection = WorkbenchHistoricalInspection(
                    adapter.package,
                    inspection_executor=self._executor,
                )
            else:
                historical_inspection = WorkbenchHistoricalInspection(
                    adapter.package,
                    checkpoint_budgets=self._checkpoint_load_budgets,
                    inspection_executor=self._executor,
                )
            return historical_inspection.inspect(
                context,
                HistoricalInspectionRequest(
                    preset=preset,
                    request_overrides=parsed_request_overrides.values,
                    dataset=dataset,
                    experiment_task=experiment_task,
                ),
            )
        return self._executor.inspect(
            adapter.package,
            InspectionRequest(
                preset=preset,
                overrides=parsed_request_overrides,
                dataset=dataset,
                experiment_task=experiment_task,
            ),
        )

    def inspect_payload(self, **request: Any) -> dict[str, Any]:
        """Compatibility Adapter for non-HTTP callers expecting camel-case data."""
        from workbench.backend.inspection_serialization import (
            inspection_result_payload,
        )

        return inspection_result_payload(self.inspect(**request))


__all__ = ["InspectionService"]

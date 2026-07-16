from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from model_runtime.inspection import InspectionRequest, InspectionResult

from emperor_workbench.inspection._errors import (
    InspectionFailure,
    inspection_failure,
)
from emperor_workbench.inspection._executor import InspectionExecutor
from emperor_workbench.model_packages import (
    ModelPackageFailure,
    SelectedModelPackage,
)
from emperor_workbench.run_history import HistoricalInspectionSource


class InspectionService:
    """Interpret one selected Model Package as a semantic model graph."""

    def __init__(
        self,
        executor: InspectionExecutor,
        *,
        historical_source: HistoricalInspectionSource | None = None,
    ) -> None:
        self._executor = executor
        self._historical_source = historical_source

    def inspect(
        self,
        selected: SelectedModelPackage,
        *,
        preset: str,
        overrides: Mapping[str, Any],
        dataset: str | None,
        experiment_task: str | None = None,
        log_run_id: str | None = None,
    ) -> InspectionResult:
        try:
            parsed_overrides = selected.parse_overrides(overrides)
        except ModelPackageFailure as exc:
            raise inspection_failure(exc) from exc

        if log_run_id is not None:
            if self._historical_source is None:
                raise InspectionFailure("Log run inspection is not configured.")
            from emperor_workbench.inspection._historical._inspection import (
                HistoricalInspection,
            )

            return HistoricalInspection(
                selected,
                executor=self._executor,
                source=self._historical_source,
            ).inspect(
                log_run_id=log_run_id,
                preset=preset,
                request_overrides=parsed_overrides.values,
                dataset=dataset,
                experiment_task=experiment_task,
            )

        return self._executor.inspect(
            selected,
            InspectionRequest(
                preset=preset,
                overrides=parsed_overrides,
                dataset=dataset,
                experiment_task=experiment_task,
            ),
        )


__all__ = ["InspectionService"]

"""Filesystem/TensorBoard data-access adapter for log-run state.

This repository is intentionally thin: it is an extension point between
services and the concrete local ``LogRunIndex`` data-access object.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from workbench.backend.log_runs import (
    LogExperiment,
    LogExperimentDeleteResult,
    LogRun,
    LogRunDeleteFilters,
    LogRunDeletePlan,
    LogRunDeleteResult,
    LogRunIndex,
)


class LogRunRepository:
    def __init__(self, index: LogRunIndex) -> None:
        self._index = index

    def list_runs(self) -> list[LogRun]:
        return self._index.list_runs()

    def cached_layer_monitor_data_for_run(self, run: LogRun) -> bool | None:
        return self._index.cached_layer_monitor_data_for_run(run)

    def list_experiments(self) -> list[LogExperiment]:
        return self._index.list_experiments()

    def delete_experiment(
        self,
        experiment: str,
    ) -> LogExperimentDeleteResult:
        return self._index.delete_experiment(experiment)

    def create_delete_plan(
        self,
        filters: LogRunDeleteFilters,
        *,
        active_jobs: list[dict[str, Any]],
    ) -> LogRunDeletePlan:
        return self._index.create_delete_plan(filters, active_jobs=active_jobs)

    def delete_runs(
        self,
        filters: LogRunDeleteFilters,
        *,
        active_jobs: list[dict[str, Any]],
    ) -> LogRunDeleteResult:
        return self._index.delete_runs(filters, active_jobs=active_jobs)

    def import_archive(
        self,
        *,
        archive: bytes,
        filename: str,
        max_upload_size: int | None,
        max_extracted_size: int | None,
    ) -> dict[str, object]:
        return self._index.import_archive(
            archive=archive,
            filename=filename,
            max_upload_size=max_upload_size,
            max_extracted_size=max_extracted_size,
        )

    def tags_for_runs(self, run_ids: list[str]) -> list[dict[str, Any]]:
        return self._index.tags_for_runs(run_ids)

    def scalars_for_runs(
        self,
        *,
        run_ids: list[str],
        tags: list[str],
        max_points: int,
        sampling: str,
    ) -> list[dict[str, Any]]:
        return self._index.scalars_for_runs(
            run_ids=run_ids,
            tags=tags,
            max_points=max_points,
            sampling=sampling,
        )

    def media_for_runs(
        self,
        *,
        run_ids: list[str],
        image_tags: list[str],
        text_tags: list[str],
    ) -> dict[str, Any]:
        return self._index.media_for_runs(
            run_ids=run_ids,
            image_tags=image_tags,
            text_tags=text_tags,
        )

    def monitor_data_for_run(self, run_id: str, node_path: str) -> dict[str, Any]:
        return self._index.monitor_data_for_run(run_id, node_path=node_path)

    def parameter_status_for_runs(self, run_ids: list[str]) -> list[dict[str, Any]]:
        return self._index.parameter_status_for_runs(run_ids)

    def checkpoints_for_runs(self, run_ids: list[str]) -> list[dict[str, Any]]:
        return self._index.checkpoints_for_runs(run_ids)

    def checkpoint_paths_for_run(self, run_id: str) -> list[Path]:
        return self._index.checkpoint_paths_for_run(run_id)

    def artifacts_for_run(self, run_id: str) -> dict[str, Any]:
        return self._index.artifacts_for_run(run_id)

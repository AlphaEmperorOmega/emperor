"""Compatibility exports and Log Run index orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from viewer.backend import log_run_models as _log_run_models
from viewer.backend import log_run_query as _log_run_query
from viewer.backend.log_run_deletion import (
    LogRunDeletionExecutor,
    LogRunDeletionPlanner,
)
from viewer.backend.log_run_models import (
    ActiveLogRunDeleteBlocker,
    LogCheckpoint,
    LogExperiment,
    LogExperimentDeleteResult,
    LogRun,
    LogRunArtifact,
    LogRunArtifacts,
    LogRunDeleteCandidate,
    LogRunDeleteFilters,
    LogRunDeletePlan,
    LogRunDeleteResult,
)
from viewer.backend.log_run_names import (
    LOG_EXPERIMENT_NAME_RE,
    is_valid_log_experiment_name,
    validate_log_experiment_name,
)
from viewer.backend.log_run_query import LogRunQueryService
from viewer.backend.log_run_scanner import LogRunScanner
from viewer.backend.monitor_data import (
    DEFAULT_SCALAR_POINT_LIMIT,
    TensorBoardMonitorReader,
    TensorBoardParameterStatusReader,
)
from viewer.backend.services.log_import import import_log_archive

LOG_RESPONSE_ITEM_LIMIT = _log_run_models.LOG_RESPONSE_ITEM_LIMIT
LOG_EVENT_CACHE_MAX_ENTRIES = _log_run_query.LOG_EVENT_CACHE_MAX_ENTRIES
LOG_TAG_READ_MAX_EVENT_BYTES = _log_run_query.LOG_TAG_READ_MAX_EVENT_BYTES
LOG_TAG_BATCH_READ_MAX_EVENT_BYTES = _log_run_query.LOG_TAG_BATCH_READ_MAX_EVENT_BYTES
LOG_SCALAR_ACCUMULATOR_CACHE_MAX_ENTRIES = (
    _log_run_query.LOG_SCALAR_ACCUMULATOR_CACHE_MAX_ENTRIES
)
LOG_TAG_KEYS = _log_run_query.LOG_TAG_KEYS

__all__ = [
    "LOG_EXPERIMENT_NAME_RE",
    "ActiveLogRunDeleteBlocker",
    "LogCheckpoint",
    "LogExperiment",
    "LogExperimentDeleteResult",
    "LogRun",
    "LogRunArtifact",
    "LogRunArtifacts",
    "LogRunDeleteCandidate",
    "LogRunDeleteFilters",
    "LogRunDeletePlan",
    "LogRunDeleteResult",
    "LogRunDeletionExecutor",
    "LogRunDeletionPlanner",
    "LogRunIndex",
    "LogRunQueryService",
    "LogRunScanner",
    "is_valid_log_experiment_name",
    "validate_log_experiment_name",
]


class LogRunIndex:
    def __init__(
        self,
        *,
        logs_root: Path | str = "logs",
        scalar_point_limit: int = DEFAULT_SCALAR_POINT_LIMIT,
        monitor_reader: TensorBoardMonitorReader | None = None,
        parameter_status_reader: TensorBoardParameterStatusReader | None = None,
        scanner: LogRunScanner | None = None,
        query_service: LogRunQueryService | None = None,
        deletion_planner: LogRunDeletionPlanner | None = None,
        deletion_executor: LogRunDeletionExecutor | None = None,
    ) -> None:
        self.logs_root = Path(logs_root)
        self.scalar_point_limit = scalar_point_limit
        self.scanner = scanner or LogRunScanner(logs_root=self.logs_root)
        self.monitor_reader = monitor_reader or TensorBoardMonitorReader(
            scalar_point_limit=scalar_point_limit,
        )
        self.parameter_status_reader = (
            parameter_status_reader or TensorBoardParameterStatusReader()
        )
        self.query_service = query_service or LogRunQueryService(
            scanner=self.scanner,
            scalar_point_limit=scalar_point_limit,
            monitor_reader=self.monitor_reader,
            parameter_status_reader=self.parameter_status_reader,
        )
        self.deletion_planner = deletion_planner or LogRunDeletionPlanner(
            scanner=self.scanner,
        )
        self.deletion_executor = deletion_executor or LogRunDeletionExecutor(
            scanner=self.scanner,
        )

    def list_runs(self) -> list[LogRun]:
        return self.scanner.list_runs()

    def list_experiments(self) -> list[LogExperiment]:
        return self.scanner.list_experiments()

    def tags_for_runs(self, run_ids: list[str]) -> list[dict[str, Any]]:
        return self.query_service.tags_for_runs(run_ids)

    def cached_layer_monitor_data_for_run(self, run: LogRun) -> bool | None:
        return self.query_service.cached_layer_monitor_data_for_run(run)

    def scalars_for_runs(
        self,
        *,
        run_ids: list[str],
        tags: list[str],
        max_points: int | None = None,
        sampling: str = "tail",
    ) -> list[dict[str, Any]]:
        return self.query_service.scalars_for_runs(
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
        return self.query_service.media_for_runs(
            run_ids=run_ids,
            image_tags=image_tags,
            text_tags=text_tags,
        )

    def monitor_data_for_run(self, run_id: str, node_path: str) -> dict[str, Any]:
        return self.query_service.monitor_data_for_run(
            run_id,
            node_path=node_path,
        )

    def parameter_status_for_runs(self, run_ids: list[str]) -> list[dict[str, Any]]:
        return self.query_service.parameter_status_for_runs(run_ids)

    def checkpoints_for_runs(self, run_ids: list[str]) -> list[dict[str, Any]]:
        return self.query_service.checkpoints_for_runs(run_ids)

    def checkpoint_paths_for_run(self, run_id: str) -> list[Path]:
        return self.query_service.checkpoint_paths_for_run(run_id)

    def artifacts_for_run(self, run_id: str) -> dict[str, Any]:
        return self.query_service.artifacts_for_run(run_id)

    def delete_experiment(
        self,
        experiment: str,
    ) -> LogExperimentDeleteResult:
        affected_runs = [
            run for run in self.scanner.list_runs() if run.experiment == experiment
        ]
        result = self.deletion_executor.delete_experiment(experiment)
        self.scanner.clear_cache()
        self.query_service.clear_run_caches([run.path for run in affected_runs])
        return result

    def create_delete_plan(
        self,
        filters: LogRunDeleteFilters,
        *,
        active_jobs: list[dict[str, Any]],
    ) -> LogRunDeletePlan:
        return self.deletion_planner.create_delete_plan(
            filters,
            active_jobs=active_jobs,
        )

    def delete_runs(
        self,
        filters: LogRunDeleteFilters,
        *,
        active_jobs: list[dict[str, Any]],
    ) -> LogRunDeleteResult:
        plan = self.create_delete_plan(filters, active_jobs=active_jobs)
        affected_run_paths = [candidate.path for candidate in plan.candidates]
        result = self.deletion_executor.delete_runs(plan)
        self.scanner.clear_cache()
        self.query_service.clear_run_caches(affected_run_paths)
        return result

    def import_archive(
        self,
        *,
        archive: bytes,
        filename: str,
        max_upload_size: int | None,
        max_extracted_size: int | None,
    ) -> dict[str, object]:
        result = import_log_archive(
            archive=archive,
            filename=filename,
            logs_root=self.logs_root,
            max_upload_size=max_upload_size,
            max_extracted_size=max_extracted_size,
        )
        self.scanner.clear_cache()
        self.query_service.clear_cache()
        return result

    def _resolved_root(self) -> Path:
        return self.scanner.resolved_root()

    def _resolve_under_root(self, path: Path, root: Path) -> Path | None:
        return self.scanner.resolve_under_root(path, root)

    def _parse_run(self, root: Path, version_dir: Path) -> LogRun | None:
        return self.scanner.parse_run(root, version_dir)

    def _filtered_runs(self, filters: LogRunDeleteFilters) -> list[LogRun]:
        return self.deletion_planner.filtered_runs(filters)

    def _validated_delete_candidate_path(
        self,
        candidate: LogRunDeleteCandidate,
        root: Path,
    ) -> Path:
        return self.deletion_executor.validated_delete_candidate_path(
            candidate,
            root,
        )

    def _prune_empty_run_parents(
        self,
        *,
        start: Path,
        experiment_dir: Path,
        root: Path,
    ) -> None:
        self.deletion_executor.prune_empty_run_parents(
            start=start,
            experiment_dir=experiment_dir,
            root=root,
        )

    def _resolve_runs(self, run_ids: list[str]) -> list[LogRun]:
        return self.scanner.resolve_runs(run_ids)

    def _read_tags(self, run_dir: Path) -> dict[str, list[str]]:
        return self.query_service.read_tags(run_dir)

    def _read_scalar_points(self, run_dir: Path, tag: str) -> list[dict[str, Any]]:
        return self.query_service.read_scalar_points(run_dir, tag)

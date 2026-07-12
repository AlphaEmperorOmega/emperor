"""Log Run deletion planning and execution."""

from __future__ import annotations

import shutil
from collections.abc import Iterable
from pathlib import Path

from workbench.backend.inspector.errors import InspectorError
from workbench.backend.log_experiments import is_valid_log_experiment_name
from workbench.backend.run_history.contracts import ActiveLogWriter
from workbench.backend.run_history.paths import resolved_under_root
from workbench.backend.run_history.records import (
    ActiveLogRunDeleteBlocker,
    LogExperimentDeleteResult,
    LogRun,
    LogRunDeleteCandidate,
    LogRunDeleteFilters,
    LogRunDeletePlan,
    LogRunDeleteResult,
)
from workbench.backend.run_history.scanner import LogRunScanner


def _is_log_path_under_root(path: Path, root: Path) -> bool:
    return resolved_under_root(path, root) is not None


def _validate_log_experiment_delete_name(experiment: str) -> None:
    if not experiment:
        raise InspectorError("Log experiment name is required")
    if "/" in experiment or "\\" in experiment or experiment in {".", ".."}:
        raise InspectorError(f"Invalid log experiment name: {experiment}")
    if not is_valid_log_experiment_name(experiment):
        raise InspectorError(f"Invalid log experiment name: {experiment}")


def _validated_log_experiment_delete_path(root: Path, experiment: str) -> Path:
    target = root / experiment
    if target.is_symlink():
        raise InspectorError(
            f"Refusing to delete symlink log experiment: {experiment}"
        )
    if not _is_log_path_under_root(target, root):
        raise InspectorError(f"Invalid log experiment path: {experiment}") from None
    return target


def _validated_log_run_delete_candidate_path(
    candidate: LogRunDeleteCandidate,
    root: Path,
) -> Path:
    target = root / candidate.relativePath
    if target.is_symlink():
        raise InspectorError(
            f"Refusing to delete symlink log run: {candidate.relativePath}"
        )
    if not target.name.startswith("version_"):
        raise InspectorError(
            f"Refusing to delete non-version log folder: {candidate.relativePath}"
        )
    if not _is_log_path_under_root(target, root):
        raise InspectorError(
            f"Invalid log run path: {candidate.relativePath}"
        ) from None
    if not target.is_dir():
        raise InspectorError(
            f"Log run is not a directory: {candidate.relativePath}"
        )
    return target


def _prune_empty_log_run_parents(
    *,
    start: Path,
    experiment_dir: Path,
    root: Path,
) -> None:
    current = start
    while current != root:
        if not _is_log_path_under_root(current, root):
            return
        try:
            current.rmdir()
        except OSError:
            return
        if current == experiment_dir:
            return
        current = current.parent


class LogRunDeletionPlanner:
    def __init__(self, *, scanner: LogRunScanner) -> None:
        self.scanner = scanner

    def create_delete_plan(
        self,
        filters: LogRunDeleteFilters,
        *,
        active_writers: Iterable[ActiveLogWriter],
    ) -> LogRunDeletePlan:
        candidates = [
            LogRunDeleteCandidate.from_run(run)
            for run in self.filtered_runs(filters)
        ]
        candidate_experiments = {candidate.experiment for candidate in candidates}
        blockers = [
            ActiveLogRunDeleteBlocker(
                id=writer.id,
                logFolder=writer.log_folder,
                status=writer.status,
            )
            for writer in active_writers
            if writer.log_folder in candidate_experiments
        ]
        return LogRunDeletePlan(candidates=candidates, blockedByActiveJobs=blockers)

    def filtered_runs(self, filters: LogRunDeleteFilters) -> list[LogRun]:
        experiment_set = set(filters.experiments)
        dataset_set = set(filters.datasets)
        model_set = set(filters.models)
        preset_set = set(filters.presets)
        run_id_set = set(filters.runIds)
        if not all(
            (
                experiment_set,
                dataset_set,
                model_set,
                preset_set,
                run_id_set,
            )
        ):
            return []
        return [
            run
            for run in self.scanner.list_runs(result_projection="none")
            if run.experiment in experiment_set
            and run.dataset in dataset_set
            and run.model in model_set
            and run.preset in preset_set
            and run.id in run_id_set
        ]


class LogRunDeletionExecutor:
    def __init__(self, *, scanner: LogRunScanner) -> None:
        self.scanner = scanner

    def delete_experiment(
        self,
        experiment: str,
    ) -> LogExperimentDeleteResult:
        _validate_log_experiment_delete_name(experiment)
        root = self.scanner.resolved_root()
        target = _validated_log_experiment_delete_path(root, experiment)
        runs = [
            run
            for run in self.scanner.list_runs(result_projection="none")
            if run.experiment == experiment
        ]

        if not target.is_dir():
            if not runs:
                raise InspectorError(f"Unknown log experiment: {experiment}")
            raise InspectorError(f"Log experiment is not a directory: {experiment}")

        deleted_run_ids = [run.id for run in runs]
        shutil.rmtree(target)
        return LogExperimentDeleteResult(
            experiment=experiment,
            deletedRunIds=deleted_run_ids,
            deletedRunCount=len(deleted_run_ids),
            deletedRelativePath=experiment,
        )

    def delete_runs(
        self,
        plan: LogRunDeletePlan,
    ) -> LogRunDeleteResult:
        if not plan.candidates:
            raise InspectorError("No log runs match the selected filters.")
        if plan.blockedByActiveJobs:
            raise InspectorError(
                "A training job is still writing to this log folder."
            )

        deleted_run_ids: list[str] = []
        deleted_relative_paths: list[str] = []
        root = self.scanner.resolved_root()
        for candidate in plan.candidates:
            delete_dir = _validated_log_run_delete_candidate_path(candidate, root)
            shutil.rmtree(delete_dir)
            deleted_run_ids.append(candidate.id)
            deleted_relative_paths.append(candidate.relativePath)
            _prune_empty_log_run_parents(
                start=delete_dir.parent,
                experiment_dir=root / candidate.experiment,
                root=root,
            )

        return LogRunDeleteResult(
            candidates=plan.candidates,
            deletedRunIds=deleted_run_ids,
            deletedRelativePaths=deleted_relative_paths,
        )


__all__ = [
    "LogRunDeletionExecutor",
    "LogRunDeletionPlanner",
]

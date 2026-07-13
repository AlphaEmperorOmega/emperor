"""Log Run deletion planning and execution."""

from __future__ import annotations

import shutil
from collections.abc import Iterable
from pathlib import Path

from workbench.backend.log_experiments import is_valid_log_experiment_name
from workbench.backend.run_history.contracts import ActiveLogWriter
from workbench.backend.run_history.errors import RunHistoryFailure
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
        raise RunHistoryFailure("Log experiment name is required")
    if "/" in experiment or "\\" in experiment or experiment in {".", ".."}:
        raise RunHistoryFailure(f"Invalid log experiment name: {experiment}")
    if not is_valid_log_experiment_name(experiment):
        raise RunHistoryFailure(f"Invalid log experiment name: {experiment}")


def _validated_log_experiment_delete_path(root: Path, experiment: str) -> Path:
    target = root / experiment
    if target.is_symlink():
        raise RunHistoryFailure(
            f"Refusing to delete symlink log experiment: {experiment}"
        )
    if not _is_log_path_under_root(target, root):
        raise RunHistoryFailure(f"Invalid log experiment path: {experiment}") from None
    return target


def _validated_log_run_delete_candidate_path(
    candidate: LogRunDeleteCandidate,
    root: Path,
) -> Path:
    target = root / candidate.relative_path
    if target.is_symlink():
        raise RunHistoryFailure(
            f"Refusing to delete symlink log run: {candidate.relative_path}"
        )
    if not target.name.startswith("version_"):
        raise RunHistoryFailure(
            f"Refusing to delete non-version log folder: {candidate.relative_path}"
        )
    if not _is_log_path_under_root(target, root):
        raise RunHistoryFailure(
            f"Invalid log run path: {candidate.relative_path}"
        ) from None
    if not target.is_dir():
        raise RunHistoryFailure(
            f"Log run is not a directory: {candidate.relative_path}"
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
        candidates = tuple(
            LogRunDeleteCandidate.from_run(run) for run in self.filtered_runs(filters)
        )
        candidate_experiments = {candidate.experiment for candidate in candidates}
        blockers = tuple(
            ActiveLogRunDeleteBlocker(
                id=writer.id,
                log_folder=writer.log_folder,
                status=writer.status,
            )
            for writer in active_writers
            if writer.log_folder in candidate_experiments
        )
        return LogRunDeletePlan(candidates=candidates, blocked_by_active_jobs=blockers)

    def create_preset_delete_plan(
        self,
        *,
        experiment: str,
        preset: str,
        active_writers: Iterable[ActiveLogWriter],
    ) -> LogRunDeletePlan:
        _validate_log_experiment_delete_name(experiment)
        if not preset:
            raise RunHistoryFailure("Log preset name is required")
        candidates = tuple(
            LogRunDeleteCandidate.from_run(run)
            for run in self.scanner.list_runs(result_projection="none")
            if run.experiment == experiment and run.preset == preset
        )
        blockers = tuple(
            ActiveLogRunDeleteBlocker(
                id=writer.id,
                log_folder=writer.log_folder,
                status=writer.status,
            )
            for writer in active_writers
            if candidates and writer.log_folder == experiment
        )
        return LogRunDeletePlan(candidates=candidates, blocked_by_active_jobs=blockers)

    def filtered_runs(self, filters: LogRunDeleteFilters) -> list[LogRun]:
        experiment_set = set(filters.experiments)
        dataset_set = set(filters.datasets)
        model_set = set(filters.models)
        preset_set = set(filters.presets)
        run_id_set = set(filters.run_ids)
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
                raise RunHistoryFailure(f"Unknown log experiment: {experiment}")
            raise RunHistoryFailure(f"Log experiment is not a directory: {experiment}")

        deleted_run_ids = [run.id for run in runs]
        shutil.rmtree(target)
        return LogExperimentDeleteResult(
            experiment=experiment,
            deleted_run_ids=tuple(deleted_run_ids),
            deleted_run_count=len(deleted_run_ids),
            deleted_relative_path=experiment,
        )

    def delete_runs(
        self,
        plan: LogRunDeletePlan,
    ) -> LogRunDeleteResult:
        if not plan.candidates:
            raise RunHistoryFailure("No log runs match the selected filters.")
        if plan.blocked_by_active_jobs:
            raise RunHistoryFailure(
                "A training job is still writing to this log folder."
            )

        deleted_run_ids: list[str] = []
        deleted_relative_paths: list[str] = []
        root = self.scanner.resolved_root()
        for candidate in plan.candidates:
            delete_dir = _validated_log_run_delete_candidate_path(candidate, root)
            shutil.rmtree(delete_dir)
            deleted_run_ids.append(candidate.id)
            deleted_relative_paths.append(candidate.relative_path)
            _prune_empty_log_run_parents(
                start=delete_dir.parent,
                experiment_dir=root / candidate.experiment,
                root=root,
            )

        return LogRunDeleteResult(
            candidates=plan.candidates,
            deleted_run_ids=tuple(deleted_run_ids),
            deleted_relative_paths=tuple(deleted_relative_paths),
        )


__all__ = [
    "LogRunDeletionExecutor",
    "LogRunDeletionPlanner",
]

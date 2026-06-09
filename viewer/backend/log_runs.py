from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from models.catalog import MODEL_CATALOG, public_id_for_flat_name
from viewer.backend.inspector.errors import InspectorError
from viewer.backend.monitor_data import (
    DEFAULT_SCALAR_POINT_LIMIT,
    TensorBoardMonitorReader,
    TensorBoardParameterStatusReader,
)
from viewer.backend.tensorboard_reader import (
    event_dirs,
    load_event_accumulator,
    scalar_points,
)


RUN_TIMESTAMP_RE = re.compile(r"(?P<timestamp>\d{8}_\d{6})$")
LOG_EXPERIMENT_NAME_RE = re.compile(r"^[A-Za-z0-9]+(?:_[A-Za-z0-9]+)*$")


def is_valid_log_experiment_name(name: str) -> bool:
    return bool(LOG_EXPERIMENT_NAME_RE.fullmatch(name))


def validate_log_experiment_name(name: str) -> str:
    if not name:
        raise InspectorError("Log experiment folder is required")
    if not is_valid_log_experiment_name(name):
        raise InspectorError(
            "Log experiment folder must use letters and numbers separated by "
            "single underscores."
        )
    return name


def _resolved_logs_root(logs_root: Path) -> Path:
    return logs_root.resolve()


def _resolve_log_path_under_root(path: Path, root: Path) -> Path | None:
    try:
        resolved = path.resolve()
        resolved.relative_to(root)
    except (OSError, ValueError):
        return None
    return resolved


def _is_log_path_under_root(path: Path, root: Path) -> bool:
    return _resolve_log_path_under_root(path, root) is not None


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


def _run_id(relative_path: str) -> str:
    return hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:16]


def _display_timestamp(run_name: str) -> str | None:
    match = RUN_TIMESTAMP_RE.search(run_name)
    if not match:
        return None
    value = match.group("timestamp")
    return (
        f"{value[0:4]}-{value[4:6]}-{value[6:8]} "
        f"{value[9:11]}:{value[11:13]}:{value[13:15]}"
    )


def _read_result_metrics(result_path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    metrics = payload.get("metrics") if isinstance(payload, dict) else None
    return metrics if isinstance(metrics, dict) else {}


def _split_log_model_prefix(
    prefix_parts: tuple[str, ...],
) -> tuple[tuple[str, ...], str] | None:
    for index in range(len(prefix_parts)):
        candidate = "/".join(prefix_parts[index:])
        if candidate in MODEL_CATALOG:
            return prefix_parts[:index], candidate
        if "/" not in candidate:
            public_id = public_id_for_flat_name(candidate)
            if public_id is not None:
                return prefix_parts[:index], public_id
    return None


@dataclass(frozen=True)
class LogRun:
    id: str
    group: str | None
    experiment: str
    model: str
    preset: str
    dataset: str
    runName: str
    timestamp: str | None
    version: str
    relativePath: str
    hasResult: bool
    eventFileCount: int
    checkpointCount: int
    hasHparams: bool
    metrics: dict[str, Any] = field(default_factory=dict)
    path: Path = field(repr=False, compare=False, default=Path())

    def to_response(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "group": self.group,
            "experiment": self.experiment,
            "model": self.model,
            "preset": self.preset,
            "dataset": self.dataset,
            "runName": self.runName,
            "timestamp": self.timestamp,
            "version": self.version,
            "relativePath": self.relativePath,
            "hasResult": self.hasResult,
            "eventFileCount": self.eventFileCount,
            "checkpointCount": self.checkpointCount,
            "hasHparams": self.hasHparams,
            "metrics": self.metrics,
        }


@dataclass(frozen=True)
class LogExperimentDeleteResult:
    experiment: str
    deletedRunIds: list[str]
    deletedRunCount: int
    deletedRelativePath: str

    def to_response(self) -> dict[str, Any]:
        return {
            "experiment": self.experiment,
            "deletedRunIds": self.deletedRunIds,
            "deletedRunCount": self.deletedRunCount,
            "deletedRelativePath": self.deletedRelativePath,
        }


@dataclass(frozen=True)
class LogRunDeleteFilters:
    experiments: list[str]
    datasets: list[str]
    models: list[str]
    presets: list[str]
    runIds: list[str]


@dataclass(frozen=True)
class LogRunDeleteCandidate:
    id: str
    experiment: str
    model: str
    preset: str
    dataset: str
    runName: str
    version: str
    relativePath: str
    path: Path = field(repr=False, compare=False, default=Path())

    @classmethod
    def from_run(cls, run: LogRun) -> "LogRunDeleteCandidate":
        return cls(
            id=run.id,
            experiment=run.experiment,
            model=run.model,
            preset=run.preset,
            dataset=run.dataset,
            runName=run.runName,
            version=run.version,
            relativePath=run.relativePath,
            path=run.path,
        )

    def to_response(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "experiment": self.experiment,
            "model": self.model,
            "preset": self.preset,
            "dataset": self.dataset,
            "runName": self.runName,
            "version": self.version,
            "relativePath": self.relativePath,
        }


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


@dataclass(frozen=True)
class ActiveLogRunDeleteBlocker:
    id: str
    logFolder: str
    status: str

    def to_response(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "logFolder": self.logFolder,
            "status": self.status,
        }


@dataclass(frozen=True)
class LogRunDeletePlan:
    candidates: list[LogRunDeleteCandidate]
    blockedByActiveJobs: list[ActiveLogRunDeleteBlocker] = field(default_factory=list)

    @property
    def canDelete(self) -> bool:
        return bool(self.candidates) and not self.blockedByActiveJobs

    def to_response(self) -> dict[str, Any]:
        return _delete_plan_response_fields(
            self.candidates,
            blocked_by_active_jobs=self.blockedByActiveJobs,
            can_delete=self.canDelete,
        )


@dataclass(frozen=True)
class LogRunDeleteResult:
    candidates: list[LogRunDeleteCandidate]
    deletedRunIds: list[str]
    deletedRelativePaths: list[str]

    def to_response(self) -> dict[str, Any]:
        return {
            "deletedRunIds": self.deletedRunIds,
            "deletedRunCount": len(self.deletedRunIds),
            "deletedRelativePaths": self.deletedRelativePaths,
            **_delete_plan_response_fields(
                self.candidates,
                blocked_by_active_jobs=[],
                can_delete=True,
            ),
        }


@dataclass(frozen=True)
class LogExperiment:
    experiment: str
    runCount: int
    relativePath: str

    def to_response(self) -> dict[str, Any]:
        return {
            "experiment": self.experiment,
            "runCount": self.runCount,
            "relativePath": self.relativePath,
        }


def _delete_plan_response_fields(
    candidates: list[LogRunDeleteCandidate],
    *,
    blocked_by_active_jobs: list[ActiveLogRunDeleteBlocker],
    can_delete: bool,
) -> dict[str, Any]:
    affected = _affected_values(candidates)
    return {
        "candidateCount": len(candidates),
        "counts": {
            "runs": len(candidates),
            "experiments": len(affected["experiments"]),
            "datasets": len(affected["datasets"]),
            "models": len(affected["models"]),
            "presets": len(affected["presets"]),
        },
        "affected": affected,
        "candidates": [candidate.to_response() for candidate in candidates],
        "blockedByActiveJobs": [
            blocker.to_response() for blocker in blocked_by_active_jobs
        ],
        "canDelete": can_delete,
    }


def _affected_values(
    candidates: list[LogRunDeleteCandidate],
) -> dict[str, list[str]]:
    return {
        "experiments": sorted({candidate.experiment for candidate in candidates}),
        "datasets": sorted({candidate.dataset for candidate in candidates}),
        "models": sorted({candidate.model for candidate in candidates}),
        "presets": sorted({candidate.preset for candidate in candidates}),
        "runIds": sorted({candidate.id for candidate in candidates}),
    }


class LogRunIndex:
    def __init__(
        self,
        *,
        logs_root: Path | str = "logs",
        scalar_point_limit: int = DEFAULT_SCALAR_POINT_LIMIT,
        monitor_reader: TensorBoardMonitorReader | None = None,
        parameter_status_reader: TensorBoardParameterStatusReader | None = None,
    ) -> None:
        self.logs_root = Path(logs_root)
        self.scalar_point_limit = scalar_point_limit
        self.monitor_reader = monitor_reader or TensorBoardMonitorReader(
            scalar_point_limit=scalar_point_limit,
        )
        self.parameter_status_reader = (
            parameter_status_reader or TensorBoardParameterStatusReader()
        )

    def list_runs(self) -> list[LogRun]:
        root = self._resolved_root()
        if not root.exists():
            return []

        runs: list[LogRun] = []
        for version_dir in sorted(root.rglob("version_*")):
            if not version_dir.is_dir():
                continue
            resolved = self._resolve_under_root(version_dir, root)
            if resolved is None:
                continue
            run = self._parse_run(root, resolved)
            if run is not None:
                runs.append(run)
        return sorted(
            runs,
            key=lambda run: (
                run.timestamp or "",
                run.group or "",
                run.model,
                run.preset,
                run.dataset,
                run.runName,
                run.version,
            ),
            reverse=True,
        )

    def list_experiments(self) -> list[LogExperiment]:
        root = self._resolved_root()
        if not root.exists():
            return []

        run_counts = Counter(run.experiment for run in self.list_runs())
        experiments: list[LogExperiment] = []
        for child in sorted(root.iterdir(), key=lambda path: path.name):
            if not child.is_dir() or child.is_symlink():
                continue
            if not is_valid_log_experiment_name(child.name):
                continue
            resolved = self._resolve_under_root(child, root)
            if resolved is None:
                continue
            experiments.append(
                LogExperiment(
                    experiment=child.name,
                    runCount=run_counts[child.name],
                    relativePath=child.name,
                )
            )
        return experiments

    def tags_for_runs(self, run_ids: list[str]) -> list[dict[str, Any]]:
        runs = self._resolve_runs(run_ids)
        return [
            {
                "runId": run.id,
                "scalarTags": tags["scalars"],
                "histogramTags": tags["histograms"],
                "imageTags": tags["images"],
            }
            for run in runs
            for tags in [self._read_tags(run.path)]
        ]

    def scalars_for_runs(
        self,
        *,
        run_ids: list[str],
        tags: list[str],
    ) -> list[dict[str, Any]]:
        runs = self._resolve_runs(run_ids)
        requested_tags = list(dict.fromkeys(tags))
        if not requested_tags:
            return []

        series: list[dict[str, Any]] = []
        for run in runs:
            run_tags = set(self._read_tags(run.path)["scalars"])
            for tag in requested_tags:
                if tag not in run_tags:
                    continue
                points = self._read_scalar_points(run.path, tag)
                if points:
                    series.append({"runId": run.id, "tag": tag, "points": points})
        return series

    def monitor_data_for_run(self, run_id: str, node_path: str) -> dict[str, Any]:
        run = self._resolve_runs([run_id])[0]
        return self.monitor_reader.read(
            job_id=run.id,
            node_path=node_path,
            dataset=run.dataset,
            log_dir=str(run.path),
        )

    def parameter_status_for_runs(self, run_ids: list[str]) -> list[dict[str, Any]]:
        runs = self._resolve_runs(run_ids)
        return [
            self.parameter_status_reader.read(
                source_id=run.id,
                preset=run.preset,
                dataset=run.dataset,
                log_dir=str(run.path),
            )
            for run in runs
        ]

    def delete_experiment(
        self,
        experiment: str,
    ) -> LogExperimentDeleteResult:
        _validate_log_experiment_delete_name(experiment)
        root = self._resolved_root()
        target = _validated_log_experiment_delete_path(root, experiment)

        runs = [run for run in self.list_runs() if run.experiment == experiment]

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

    def create_delete_plan(
        self,
        filters: LogRunDeleteFilters,
        *,
        active_jobs: list[dict[str, Any]],
    ) -> LogRunDeletePlan:
        candidates = [
            LogRunDeleteCandidate.from_run(run)
            for run in self._filtered_runs(filters)
        ]
        candidate_experiments = {candidate.experiment for candidate in candidates}
        blockers = [
            ActiveLogRunDeleteBlocker(
                id=str(job.get("id") or ""),
                logFolder=str(job.get("logFolder") or ""),
                status=str(job.get("status") or ""),
            )
            for job in active_jobs
            if str(job.get("logFolder") or "") in candidate_experiments
        ]
        return LogRunDeletePlan(candidates=candidates, blockedByActiveJobs=blockers)

    def delete_runs(
        self,
        filters: LogRunDeleteFilters,
        *,
        active_jobs: list[dict[str, Any]],
    ) -> LogRunDeleteResult:
        plan = self.create_delete_plan(filters, active_jobs=active_jobs)
        if not plan.candidates:
            raise InspectorError("No log runs match the selected filters.")
        if plan.blockedByActiveJobs:
            raise InspectorError(
                "A training job is still writing to this log folder."
            )

        deleted_run_ids: list[str] = []
        deleted_relative_paths: list[str] = []
        root = self._resolved_root()
        for candidate in plan.candidates:
            delete_dir = self._validated_delete_candidate_path(candidate, root)
            shutil.rmtree(delete_dir)
            deleted_run_ids.append(candidate.id)
            deleted_relative_paths.append(candidate.relativePath)
            self._prune_empty_run_parents(
                start=delete_dir.parent,
                experiment_dir=root / candidate.experiment,
                root=root,
            )

        return LogRunDeleteResult(
            candidates=plan.candidates,
            deletedRunIds=deleted_run_ids,
            deletedRelativePaths=deleted_relative_paths,
        )

    def _resolved_root(self) -> Path:
        return _resolved_logs_root(self.logs_root)

    def _resolve_under_root(self, path: Path, root: Path) -> Path | None:
        return _resolve_log_path_under_root(path, root)

    def _parse_run(self, root: Path, version_dir: Path) -> LogRun | None:
        try:
            relative = version_dir.relative_to(root)
        except ValueError:
            return None

        parts = relative.parts
        if len(parts) < 5:
            return None

        version = parts[-1]
        run_name = parts[-2]
        dataset = parts[-3]
        preset = parts[-4]
        model_prefix = _split_log_model_prefix(parts[:-4])
        if model_prefix is None:
            return None
        group_parts, model = model_prefix
        group = "/".join(group_parts) if group_parts else None
        experiment = parts[0]
        relative_path = relative.as_posix()
        result_path = version_dir / "result.json"
        event_files = list(version_dir.rglob("events.out.tfevents.*"))
        checkpoints = list((version_dir / "checkpoints").glob("*.ckpt"))

        return LogRun(
            id=_run_id(relative_path),
            group=group,
            experiment=experiment,
            model=model,
            preset=preset,
            dataset=dataset,
            runName=run_name,
            timestamp=_display_timestamp(run_name),
            version=version,
            relativePath=relative_path,
            hasResult=result_path.is_file(),
            eventFileCount=len(event_files),
            checkpointCount=len(checkpoints),
            hasHparams=(version_dir / "hparams.yaml").is_file(),
            metrics=_read_result_metrics(result_path) if result_path.is_file() else {},
            path=version_dir,
        )

    def _filtered_runs(self, filters: LogRunDeleteFilters) -> list[LogRun]:
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
            for run in self.list_runs()
            if run.experiment in experiment_set
            and run.dataset in dataset_set
            and run.model in model_set
            and run.preset in preset_set
            and run.id in run_id_set
        ]

    def _validated_delete_candidate_path(
        self,
        candidate: LogRunDeleteCandidate,
        root: Path,
    ) -> Path:
        return _validated_log_run_delete_candidate_path(candidate, root)

    def _prune_empty_run_parents(
        self,
        *,
        start: Path,
        experiment_dir: Path,
        root: Path,
    ) -> None:
        _prune_empty_log_run_parents(
            start=start,
            experiment_dir=experiment_dir,
            root=root,
        )

    def _resolve_runs(self, run_ids: list[str]) -> list[LogRun]:
        if not run_ids:
            return []
        runs_by_id = {run.id: run for run in self.list_runs()}
        unknown = [run_id for run_id in run_ids if run_id not in runs_by_id]
        if unknown:
            raise InspectorError(f"Unknown log run id: {unknown[0]}")
        return [runs_by_id[run_id] for run_id in dict.fromkeys(run_ids)]

    def _read_tags(self, run_dir: Path) -> dict[str, list[str]]:
        tags = {"scalars": set(), "histograms": set(), "images": set()}
        for event_dir in event_dirs(run_dir):
            accumulator = load_event_accumulator(event_dir)
            if accumulator is None:
                continue
            try:
                accumulator_tags = accumulator.Tags()
            except Exception:
                continue
            tags["scalars"].update(accumulator_tags.get("scalars", []))
            tags["histograms"].update(accumulator_tags.get("histograms", []))
            tags["images"].update(accumulator_tags.get("images", []))
        return {key: sorted(value) for key, value in tags.items()}

    def _read_scalar_points(self, run_dir: Path, tag: str) -> list[dict[str, Any]]:
        points: list[dict[str, Any]] = []
        for event_dir in event_dirs(run_dir):
            accumulator = load_event_accumulator(event_dir)
            if accumulator is None:
                continue
            try:
                points.extend(scalar_points(accumulator, tag, None))
            except Exception:
                continue

        points.sort(key=lambda point: (point["step"], point["wallTime"]))
        return points[-self.scalar_point_limit :]

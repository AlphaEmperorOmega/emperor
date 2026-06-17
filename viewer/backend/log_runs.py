from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
from collections import Counter, OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from models.catalog import (
    MODEL_CATALOG,
    model_identity_payload_from_id,
    public_id_for_flat_name,
)

from viewer.backend.inspector.errors import InspectorError
from viewer.backend.monitor_data import (
    DEFAULT_SCALAR_POINT_LIMIT,
    TensorBoardMonitorReader,
    TensorBoardParameterStatusReader,
)
from viewer.backend.tensorboard_reader import (
    TENSORBOARD_TAG_SIZE_GUIDANCE,
    event_dirs,
    event_file_fingerprint,
    event_file_index,
    event_file_total_size,
    image_summary,
    load_event_accumulator,
    scalar_points,
    text_summary,
)

RUN_TIMESTAMP_RE = re.compile(r"(?P<timestamp>\d{8}_\d{6})$")
LOG_EXPERIMENT_NAME_RE = re.compile(r"^[A-Za-z0-9]+(?:_[A-Za-z0-9]+)*$")
CHECKPOINT_EPOCH_RE = re.compile(r"(?:^|[-_])epoch=(?P<value>\d+)(?:[-_]|$)")
CHECKPOINT_STEP_RE = re.compile(r"(?:^|[-_])step=(?P<value>\d+)(?:[-_]|$)")
HPARAM_INT_RE = re.compile(r"^[+-]?\d+$")
HPARAM_FLOAT_RE = re.compile(
    r"^[+-]?(?:(?:\d+\.\d*)|(?:\.\d+)|(?:\d+))(?:[eE][+-]?\d+)$"
    r"|^[+-]?(?:(?:\d+\.\d*)|(?:\.\d+))$"
)
LOG_EVENT_CACHE_MAX_ENTRIES = 256
LOG_TAG_READ_MAX_EVENT_BYTES = 96 * 1024 * 1024
LOG_RESPONSE_ITEM_LIMIT = 500
LOG_TAG_KEYS = ("scalars", "histograms", "images", "texts")
EventFingerprint = tuple[tuple[str, int, int], ...]


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


def _read_result_payload(result_path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_result_object(result_path: Path, key: str) -> dict[str, Any]:
    value = _read_result_payload(result_path).get(key)
    return value if isinstance(value, dict) else {}


def _read_result_metrics(result_path: Path) -> dict[str, Any]:
    return _read_result_object(result_path, "metrics")


def _read_result_params(result_path: Path) -> dict[str, Any]:
    return _read_result_object(result_path, "params")


def _parse_hparam_value(raw_value: str) -> bool | int | float | str | None:
    value = raw_value.strip()
    if not value:
        return ""
    if " #" in value:
        value = value.split(" #", 1)[0].strip()
    normalized = value.lower()
    if normalized in {"null", "none", "~"}:
        return None
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    if (
        len(value) >= 2
        and value[0] == value[-1]
        and value[0] in {"'", '"'}
    ):
        return value[1:-1]
    if HPARAM_INT_RE.fullmatch(value):
        try:
            return int(value)
        except ValueError:
            return value
    if HPARAM_FLOAT_RE.fullmatch(value):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def _read_hparams_flat(hparams_path: Path) -> dict[str, Any]:
    try:
        lines = hparams_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return {}

    values: dict[str, Any] = {}
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        key = key.strip()
        value = raw_value.strip()
        if (
            not key
            or not value
            or value in {"|", ">"}
            or value.startswith(("[", "{", "- "))
        ):
            continue
        values[key] = _parse_hparam_value(value)
    return values


def _file_modified_at(path: Path) -> str:
    return (
        datetime.fromtimestamp(path.stat().st_mtime, UTC)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _relative_file_path(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _run_relative_file_label(run_dir: Path, path: Path) -> str:
    try:
        return path.relative_to(run_dir).as_posix()
    except ValueError:
        return path.name


def _safe_artifact_file(path: Path, root: Path) -> Path | None:
    if not path.is_file():
        return None
    resolved = _resolve_log_path_under_root(path, root)
    if resolved is None or not resolved.is_file():
        return None
    return resolved


def _safe_artifact_files(run_dir: Path, root: Path, pattern: str) -> list[Path]:
    files: list[Path] = []
    for path in sorted(run_dir.rglob(pattern)):
        resolved = _safe_artifact_file(path, root)
        if resolved is not None:
            files.append(resolved)
    return files


def _event_file_fingerprint(run_dir: Path) -> EventFingerprint:
    return event_file_fingerprint(run_dir)


def _file_id(run_id: str, relative_path: str) -> str:
    return hashlib.sha256(f"{run_id}:{relative_path}".encode()).hexdigest()[:16]


def _parse_checkpoint_field(pattern: re.Pattern[str], filename: str) -> int | None:
    match = pattern.search(filename)
    if not match:
        return None
    try:
        return int(match.group("value"))
    except ValueError:
        return None


def _parse_checkpoint_epoch(filename: str) -> int | None:
    return _parse_checkpoint_field(CHECKPOINT_EPOCH_RE, Path(filename).stem)


def _parse_checkpoint_step(filename: str) -> int | None:
    return _parse_checkpoint_field(CHECKPOINT_STEP_RE, Path(filename).stem)


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
            **model_identity_payload_from_id(self.model),
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
class LogCheckpoint:
    id: str
    runId: str
    filename: str
    relativePath: str
    epoch: int | None
    step: int | None
    sizeBytes: int
    modifiedAt: str

    def to_response(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "runId": self.runId,
            "filename": self.filename,
            "relativePath": self.relativePath,
            "epoch": self.epoch,
            "step": self.step,
            "sizeBytes": self.sizeBytes,
            "modifiedAt": self.modifiedAt,
        }


@dataclass(frozen=True)
class LogRunArtifact:
    id: str
    kind: str
    label: str
    relativePath: str
    sizeBytes: int
    modifiedAt: str

    def to_response(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "label": self.label,
            "relativePath": self.relativePath,
            "sizeBytes": self.sizeBytes,
            "modifiedAt": self.modifiedAt,
        }


@dataclass(frozen=True)
class LogRunArtifacts:
    runId: str
    params: dict[str, Any]
    metrics: dict[str, Any]
    artifacts: list[LogRunArtifact]
    checkpoints: list[LogCheckpoint]

    def to_response(self) -> dict[str, Any]:
        return {
            "runId": self.runId,
            "params": self.params,
            "metrics": self.metrics,
            "artifacts": [artifact.to_response() for artifact in self.artifacts],
            "checkpoints": [
                checkpoint.to_response() for checkpoint in self.checkpoints
            ],
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
    def from_run(cls, run: LogRun) -> LogRunDeleteCandidate:
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
            **model_identity_payload_from_id(self.model),
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
    returned_candidates = candidates[:LOG_RESPONSE_ITEM_LIMIT]
    truncated = len(candidates) > len(returned_candidates)
    return {
        "candidateCount": len(candidates),
        "sourceItemCount": len(candidates),
        "returnedItemCount": len(returned_candidates),
        "truncated": truncated,
        "truncationReason": (
            f"delete candidates capped at {LOG_RESPONSE_ITEM_LIMIT} rows"
            if truncated
            else None
        ),
        "counts": {
            "runs": len(candidates),
            "experiments": len(affected["experiments"]),
            "datasets": len(affected["datasets"]),
            "models": len(affected["models"]),
            "presets": len(affected["presets"]),
        },
        "affected": affected,
        "candidates": [candidate.to_response() for candidate in returned_candidates],
        "blockedByActiveJobs": [
            blocker.to_response() for blocker in blocked_by_active_jobs
        ],
        "canDelete": can_delete,
    }


def _affected_values(
    candidates: list[LogRunDeleteCandidate],
) -> dict[str, Any]:
    model_ids = sorted({candidate.model for candidate in candidates})
    return {
        "experiments": sorted({candidate.experiment for candidate in candidates}),
        "datasets": sorted({candidate.dataset for candidate in candidates}),
        "models": [
            model_identity_payload_from_id(model_id) for model_id in model_ids
        ],
        "presets": sorted({candidate.preset for candidate in candidates}),
        "runIds": sorted({candidate.id for candidate in candidates}),
    }


class LogRunScanner:
    def __init__(
        self,
        *,
        logs_root: Path | str = "logs",
        cache_ttl_seconds: float = 1.0,
    ) -> None:
        self.logs_root = Path(logs_root)
        self.cache_ttl_seconds = max(0.0, float(cache_ttl_seconds))
        self._runs_cache: list[LogRun] | None = None
        self._runs_cache_deadline = 0.0

    def list_runs(self) -> list[LogRun]:
        now = time.monotonic()
        if self._runs_cache is not None and now < self._runs_cache_deadline:
            return list(self._runs_cache)

        root = self.resolved_root()
        if not root.exists():
            return []

        runs: list[LogRun] = []
        for version_dir in sorted(root.rglob("version_*")):
            if not version_dir.is_dir():
                continue
            resolved = self.resolve_under_root(version_dir, root)
            if resolved is None:
                continue
            run = self.parse_run(root, resolved)
            if run is not None:
                runs.append(run)
        sorted_runs = sorted(
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
        self._runs_cache = list(sorted_runs)
        self._runs_cache_deadline = now + self.cache_ttl_seconds
        return sorted_runs

    def clear_cache(self) -> None:
        self._runs_cache = None
        self._runs_cache_deadline = 0.0

    def list_experiments(self) -> list[LogExperiment]:
        root = self.resolved_root()
        if not root.exists():
            return []

        run_counts = Counter(run.experiment for run in self.list_runs())
        experiments: list[LogExperiment] = []
        for child in sorted(root.iterdir(), key=lambda path: path.name):
            if not child.is_dir() or child.is_symlink():
                continue
            if not is_valid_log_experiment_name(child.name):
                continue
            resolved = self.resolve_under_root(child, root)
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

    def resolved_root(self) -> Path:
        return _resolved_logs_root(self.logs_root)

    def resolve_under_root(self, path: Path, root: Path) -> Path | None:
        return _resolve_log_path_under_root(path, root)

    def parse_run(self, root: Path, version_dir: Path) -> LogRun | None:
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
        result_path = _safe_artifact_file(version_dir / "result.json", root)
        hparams_path = _safe_artifact_file(version_dir / "hparams.yaml", root)
        event_files = _safe_artifact_files(
            version_dir,
            root,
            "events.out.tfevents.*",
        )
        checkpoints = _safe_artifact_files(version_dir, root, "*.ckpt")

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
            hasResult=result_path is not None,
            eventFileCount=len(event_files),
            checkpointCount=len(checkpoints),
            hasHparams=hparams_path is not None,
            metrics=_read_result_metrics(result_path) if result_path else {},
            path=version_dir,
        )

    def resolve_runs(self, run_ids: list[str]) -> list[LogRun]:
        if not run_ids:
            return []
        runs_by_id = {run.id: run for run in self.list_runs()}
        unknown = [run_id for run_id in run_ids if run_id not in runs_by_id]
        if unknown:
            raise InspectorError(f"Unknown log run id: {unknown[0]}")
        return [runs_by_id[run_id] for run_id in dict.fromkeys(run_ids)]

    def artifact_path(self, run: LogRun, filename: str) -> Path | None:
        return _safe_artifact_file(run.path / filename, self.resolved_root())

    def artifact_files(self, run: LogRun, pattern: str) -> list[Path]:
        return _safe_artifact_files(run.path, self.resolved_root(), pattern)


class LogRunQueryService:
    def __init__(
        self,
        *,
        scanner: LogRunScanner,
        scalar_point_limit: int = DEFAULT_SCALAR_POINT_LIMIT,
        max_tag_event_bytes: int = LOG_TAG_READ_MAX_EVENT_BYTES,
        monitor_reader: TensorBoardMonitorReader | None = None,
        parameter_status_reader: TensorBoardParameterStatusReader | None = None,
    ) -> None:
        self.scanner = scanner
        self.scalar_point_limit = scalar_point_limit
        self.max_tag_event_bytes = max(0, int(max_tag_event_bytes))
        self.monitor_reader = monitor_reader or TensorBoardMonitorReader(
            scalar_point_limit=scalar_point_limit,
        )
        self.parameter_status_reader = (
            parameter_status_reader or TensorBoardParameterStatusReader()
        )
        self._tags_cache: OrderedDict[
            tuple[str, EventFingerprint],
            dict[str, Any],
        ] = OrderedDict()
        self._scalar_cache: OrderedDict[
            tuple[str, EventFingerprint, str, int, str],
            dict[str, Any],
        ] = OrderedDict()

    def _cache_get(
        self,
        cache: OrderedDict[Any, Any],
        key: Any,
    ) -> Any | None:
        if key not in cache:
            return None
        cache.move_to_end(key)
        return cache[key]

    def _cache_set(
        self,
        cache: OrderedDict[Any, Any],
        key: Any,
        value: Any,
    ) -> None:
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > LOG_EVENT_CACHE_MAX_ENTRIES:
            cache.popitem(last=False)

    def clear_cache(self) -> None:
        self._tags_cache.clear()
        self._scalar_cache.clear()

    def clear_run_caches(self, run_paths: list[Path]) -> None:
        roots = {path.as_posix() for path in run_paths}
        if not roots:
            return
        for cache in (self._tags_cache, self._scalar_cache):
            for key in list(cache):
                if key and key[0] in roots:
                    cache.pop(key, None)
        self.monitor_reader.clear_roots(roots)
        self.parameter_status_reader.clear_roots(roots)

    def _tags_cache_key(
        self,
        run_dir: Path,
    ) -> tuple[str, EventFingerprint]:
        return (run_dir.as_posix(), _event_file_fingerprint(run_dir))

    def _copy_tags_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        copied = dict(payload)
        for key in LOG_TAG_KEYS:
            value = copied.get(key)
            copied[key] = list(value) if isinstance(value, list) else []
        return copied

    def _cached_tags_if_current(
        self,
        run_dir: Path,
    ) -> dict[str, Any] | None:
        cached = self._cache_get(self._tags_cache, self._tags_cache_key(run_dir))
        if cached is None:
            return None
        return self._copy_tags_payload(cached)

    def _scalar_cache_key(
        self,
        run_dir: Path,
        *,
        tag: str,
        max_points: int,
        sampling: str,
    ) -> tuple[str, EventFingerprint, str, int, str]:
        return (
            run_dir.as_posix(),
            _event_file_fingerprint(run_dir),
            tag,
            max_points,
            sampling,
        )

    def tags_for_runs(self, run_ids: list[str]) -> list[dict[str, Any]]:
        runs = self.scanner.resolve_runs(run_ids)
        return [
            {
                "runId": run.id,
                "scalarTags": tags["scalars"],
                "histogramTags": tags["histograms"],
                "imageTags": tags["images"],
                "textTags": tags["texts"],
                "eventBytes": tags.get("eventBytes"),
                "skippedEventFiles": tags.get("skippedEventFiles"),
                "truncated": tags.get("truncated"),
                "truncationReason": tags.get("truncationReason"),
                "sourceItemCount": tags.get("sourceItemCount"),
                "returnedItemCount": tags.get("returnedItemCount"),
            }
            for run in runs
            for tags in [self.read_tags(run.path)]
        ]

    def scalars_for_runs(
        self,
        *,
        run_ids: list[str],
        tags: list[str],
        max_points: int | None = None,
        sampling: str = "tail",
    ) -> list[dict[str, Any]]:
        runs = self.scanner.resolve_runs(run_ids)
        requested_tags = list(dict.fromkeys(tags))
        if not requested_tags:
            return []

        series: list[dict[str, Any]] = []
        for run in runs:
            cached_tags = self._cached_tags_if_current(run.path)
            run_tags = set(
                cached_tags["scalars"]
                if cached_tags is not None
                else self.read_tags(run.path)["scalars"],
            )
            for tag in requested_tags:
                if tag not in run_tags:
                    continue
                scalar_series = self.read_scalar_series(
                    run.path,
                    tag,
                    max_points=max_points,
                    sampling=sampling,
                )
                if scalar_series["points"]:
                    series.append(
                        {
                            "runId": run.id,
                            "tag": tag,
                            **scalar_series,
                        }
                    )
        return series

    def media_for_runs(
        self,
        *,
        run_ids: list[str],
        image_tags: list[str],
        text_tags: list[str],
    ) -> dict[str, Any]:
        runs = self.scanner.resolve_runs(run_ids)
        requested_image_tags = list(dict.fromkeys(image_tags))
        requested_text_tags = list(dict.fromkeys(text_tags))
        images: list[dict[str, Any]] = []
        texts: list[dict[str, Any]] = []
        source_item_count = len(runs) * (
            len(requested_image_tags) + len(requested_text_tags)
        )
        skipped_event_files = 0
        event_bytes = 0
        skipped_reasons: list[str] = []

        for run in runs:
            run_tags = self.read_tags(run.path)
            if run_tags.get("truncated"):
                skipped_event_files += int(run_tags.get("skippedEventFiles") or 0)
                event_bytes += int(run_tags.get("eventBytes") or 0)
                reason = run_tags.get("truncationReason")
                if isinstance(reason, str) and reason:
                    skipped_reasons.append(reason)
            image_tag_set = set(run_tags["images"])
            text_tag_set = set(run_tags["texts"])
            for tag in requested_image_tags:
                if tag not in image_tag_set:
                    continue
                summary = self.read_image_summary(run.path, tag)
                if summary is not None:
                    images.append({"runId": run.id, **summary})
            for tag in requested_text_tags:
                if tag not in text_tag_set:
                    continue
                summary = self.read_text_summary(run.path, tag)
                if summary is not None:
                    texts.append({"runId": run.id, **summary})

        returned_item_count = len(images) + len(texts)
        truncated_items = [
            item
            for item in [*images, *texts]
            if bool(item.get("truncated"))
        ]
        truncated = skipped_event_files > 0 or bool(truncated_items)
        reason = None
        if skipped_reasons:
            reason = skipped_reasons[0]
        elif truncated_items:
            reason = str(
                truncated_items[0].get("truncationReason") or "media truncated"
            )

        return {
            "sourceItemCount": source_item_count,
            "returnedItemCount": returned_item_count,
            "truncated": truncated,
            "truncationReason": reason,
            "eventBytes": event_bytes or None,
            "skippedEventFiles": skipped_event_files or None,
            "images": images,
            "texts": texts,
        }

    def monitor_data_for_run(self, run_id: str, node_path: str) -> dict[str, Any]:
        run = self.scanner.resolve_runs([run_id])[0]
        return self.monitor_reader.read(
            job_id=run.id,
            node_path=node_path,
            dataset=run.dataset,
            log_dir=str(run.path),
        )

    def parameter_status_for_runs(self, run_ids: list[str]) -> list[dict[str, Any]]:
        runs = self.scanner.resolve_runs(run_ids)
        return [
            self.parameter_status_reader.read(
                source_id=run.id,
                preset=run.preset,
                dataset=run.dataset,
                log_dir=str(run.path),
            )
            for run in runs
        ]

    def checkpoints_for_runs(self, run_ids: list[str]) -> list[dict[str, Any]]:
        runs = self.scanner.resolve_runs(run_ids)
        return [
            checkpoint.to_response()
            for run in runs
            for checkpoint in self.read_checkpoints(run)
        ]

    def artifacts_for_run(self, run_id: str) -> dict[str, Any]:
        run = self.scanner.resolve_runs([run_id])[0]
        result_path = self.scanner.artifact_path(run, "result.json")
        hparams_path = self.scanner.artifact_path(run, "hparams.yaml")
        result_params = _read_result_params(result_path) if result_path else {}
        hparams = _read_hparams_flat(hparams_path) if hparams_path else {}
        metrics = _read_result_metrics(result_path) if result_path else {}
        checkpoints = self.read_checkpoints(run)
        artifacts: list[LogRunArtifact] = []

        artifacts.extend(
            self.artifact_metadata(
                run,
                path,
                kind="event_file",
                label=_run_relative_file_label(run.path, path),
            )
            for path in self.scanner.artifact_files(run, "events.out.tfevents.*")
        )
        for filename, kind in (
            ("hparams.yaml", "hparams"),
            ("result.json", "result"),
        ):
            path = self.scanner.artifact_path(run, filename)
            if path is not None:
                artifacts.append(
                    self.artifact_metadata(
                        run,
                        path,
                        kind=kind,
                        label=filename,
                    )
                )
        artifacts.extend(
            self.artifact_metadata(
                run,
                self.scanner.resolved_root() / checkpoint.relativePath,
                kind="checkpoint",
                label=_run_relative_file_label(
                    run.path,
                    self.scanner.resolved_root() / checkpoint.relativePath,
                ),
            )
            for checkpoint in checkpoints
        )

        source_item_count = len(artifacts) + len(checkpoints)
        returned_artifacts = artifacts[:LOG_RESPONSE_ITEM_LIMIT]
        remaining_budget = max(0, LOG_RESPONSE_ITEM_LIMIT - len(returned_artifacts))
        returned_checkpoints = checkpoints[:remaining_budget]
        returned_item_count = len(returned_artifacts) + len(returned_checkpoints)
        truncated = source_item_count > returned_item_count
        response = LogRunArtifacts(
            runId=run.id,
            params={**hparams, **result_params},
            metrics=metrics,
            artifacts=returned_artifacts,
            checkpoints=returned_checkpoints,
        ).to_response()
        response.update(
            {
                "sourceItemCount": source_item_count,
                "returnedItemCount": returned_item_count,
                "truncated": truncated,
                "truncationReason": (
                    f"artifact metadata capped at {LOG_RESPONSE_ITEM_LIMIT} rows"
                    if truncated
                    else None
                ),
            }
        )
        return response

    def read_checkpoints(self, run: LogRun) -> list[LogCheckpoint]:
        checkpoints = [
            self.checkpoint_metadata(run, path)
            for path in self.scanner.artifact_files(run, "*.ckpt")
        ]
        return sorted(
            checkpoints,
            key=lambda checkpoint: (
                checkpoint.step is None,
                checkpoint.step if checkpoint.step is not None else -1,
                checkpoint.epoch is None,
                checkpoint.epoch if checkpoint.epoch is not None else -1,
                checkpoint.filename,
                checkpoint.relativePath,
            ),
        )

    def checkpoint_metadata(self, run: LogRun, path: Path) -> LogCheckpoint:
        root = self.scanner.resolved_root()
        relative_path = _relative_file_path(root, path)
        return LogCheckpoint(
            id=_file_id(run.id, relative_path),
            runId=run.id,
            filename=path.name,
            relativePath=relative_path,
            epoch=_parse_checkpoint_epoch(path.name),
            step=_parse_checkpoint_step(path.name),
            sizeBytes=path.stat().st_size,
            modifiedAt=_file_modified_at(path),
        )

    def artifact_metadata(
        self,
        run: LogRun,
        path: Path,
        *,
        kind: str,
        label: str,
    ) -> LogRunArtifact:
        root = self.scanner.resolved_root()
        relative_path = _relative_file_path(root, path)
        return LogRunArtifact(
            id=_file_id(run.id, relative_path),
            kind=kind,
            label=label,
            relativePath=relative_path,
            sizeBytes=path.stat().st_size,
            modifiedAt=_file_modified_at(path),
        )

    def read_tags(self, run_dir: Path) -> dict[str, Any]:
        cache_key = self._tags_cache_key(run_dir)
        cached = self._cache_get(self._tags_cache, cache_key)
        if cached is not None:
            return self._copy_tags_payload(cached)

        tags = {"scalars": set(), "histograms": set(), "images": set(), "texts": set()}
        if (
            self.max_tag_event_bytes > 0
            and event_file_total_size(run_dir) > self.max_tag_event_bytes
        ):
            index = event_file_index(run_dir)
            result = {
                key: sorted(value)
                for key, value in tags.items()
            } | {
                "eventBytes": index.total_size,
                "skippedEventFiles": len(index.fingerprint),
                "truncated": True,
                "truncationReason": (
                    "event files skipped: "
                    f"{index.total_size} bytes exceeds "
                    f"{self.max_tag_event_bytes} byte tag-read cap"
                ),
                "sourceItemCount": len(index.fingerprint),
                "returnedItemCount": 0,
            }
            self._cache_set(self._tags_cache, cache_key, result)
            return self._copy_tags_payload(result)
        for event_dir in event_dirs(run_dir):
            accumulator = load_event_accumulator(
                event_dir,
                size_guidance=TENSORBOARD_TAG_SIZE_GUIDANCE,
            )
            if accumulator is None:
                continue
            try:
                accumulator_tags = accumulator.Tags()
            except Exception:
                continue
            tags["scalars"].update(accumulator_tags.get("scalars", []))
            tags["histograms"].update(accumulator_tags.get("histograms", []))
            tags["images"].update(accumulator_tags.get("images", []))
            tags["texts"].update(
                tag
                for tag in accumulator_tags.get("tensors", [])
                if tag.endswith("/text_summary")
            )
        returned_item_count = sum(len(value) for value in tags.values())
        result = {
            key: sorted(value)
            for key, value in tags.items()
        } | {
            "truncated": False,
            "sourceItemCount": returned_item_count,
            "returnedItemCount": returned_item_count,
        }
        self._cache_set(self._tags_cache, cache_key, result)
        return self._copy_tags_payload(result)

    def read_scalar_series(
        self,
        run_dir: Path,
        tag: str,
        *,
        max_points: int | None = None,
        sampling: str = "tail",
    ) -> dict[str, Any]:
        point_limit = max_points if max_points is not None else self.scalar_point_limit
        cache_key = self._scalar_cache_key(
            run_dir,
            tag=tag,
            max_points=point_limit,
            sampling=sampling,
        )
        cached = self._cache_get(self._scalar_cache, cache_key)
        if cached is not None:
            return {
                "points": [dict(point) for point in cached["points"]],
                "sourcePointCount": cached["sourcePointCount"],
                "truncated": cached["truncated"],
            }

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
        source_point_count = len(points)
        if sampling == "tail":
            sampled_points = points[-point_limit:]
        else:
            sampled_points = points[-point_limit:]
        result = {
            "points": sampled_points,
            "sourcePointCount": source_point_count,
            "truncated": source_point_count > len(sampled_points),
        }
        self._cache_set(self._scalar_cache, cache_key, result)
        return {
            "points": [dict(point) for point in sampled_points],
            "sourcePointCount": result["sourcePointCount"],
            "truncated": result["truncated"],
        }

    def read_scalar_points(
        self,
        run_dir: Path,
        tag: str,
        *,
        max_points: int | None = None,
        sampling: str = "tail",
    ) -> list[dict[str, Any]]:
        return self.read_scalar_series(
            run_dir,
            tag,
            max_points=max_points,
            sampling=sampling,
        )["points"]

    def read_image_summary(self, run_dir: Path, tag: str) -> dict[str, Any] | None:
        return self._read_latest_summary(run_dir, tag, image_summary)

    def read_text_summary(self, run_dir: Path, tag: str) -> dict[str, Any] | None:
        return self._read_latest_summary(run_dir, tag, text_summary)

    def _read_latest_summary(
        self,
        run_dir: Path,
        tag: str,
        summary_reader: Callable[[Any, str], dict[str, Any] | None],
    ) -> dict[str, Any] | None:
        summaries: list[dict[str, Any]] = []
        for event_dir in event_dirs(run_dir):
            accumulator = load_event_accumulator(event_dir)
            if accumulator is None:
                continue
            try:
                summary = summary_reader(accumulator, tag)
            except Exception:
                continue
            if summary is not None:
                summaries.append(summary)
        if not summaries:
            return None
        summaries.sort(key=lambda item: (item["step"], item["wallTime"]))
        return summaries[-1]


class LogRunDeletionPlanner:
    def __init__(self, *, scanner: LogRunScanner) -> None:
        self.scanner = scanner

    def create_delete_plan(
        self,
        filters: LogRunDeleteFilters,
        *,
        active_jobs: list[dict[str, Any]],
    ) -> LogRunDeletePlan:
        candidates = [
            LogRunDeleteCandidate.from_run(run)
            for run in self.filtered_runs(filters)
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
            for run in self.scanner.list_runs()
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
            run for run in self.scanner.list_runs() if run.experiment == experiment
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
            delete_dir = self.validated_delete_candidate_path(candidate, root)
            shutil.rmtree(delete_dir)
            deleted_run_ids.append(candidate.id)
            deleted_relative_paths.append(candidate.relativePath)
            self.prune_empty_run_parents(
                start=delete_dir.parent,
                experiment_dir=root / candidate.experiment,
                root=root,
            )

        return LogRunDeleteResult(
            candidates=plan.candidates,
            deletedRunIds=deleted_run_ids,
            deletedRelativePaths=deleted_relative_paths,
        )

    def validated_delete_candidate_path(
        self,
        candidate: LogRunDeleteCandidate,
        root: Path,
    ) -> Path:
        return _validated_log_run_delete_candidate_path(candidate, root)

    def prune_empty_run_parents(
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

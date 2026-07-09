"""Log Run catalog scanning and run parsing."""

from __future__ import annotations

import hashlib
import re
import time
from collections import Counter
from pathlib import Path
from threading import RLock
from typing import Any

from models.catalog import MODEL_CATALOG, public_id_for_flat_name

from workbench.backend.inspector.errors import InspectorError
from workbench.backend.log_run_artifacts import (
    _read_result_experiment_task,
    _read_result_metrics,
    _safe_artifact_file,
    _safe_artifact_files,
)
from workbench.backend.log_run_models import LogExperiment, LogRun
from workbench.backend.log_run_names import is_valid_log_experiment_name

RUN_TIMESTAMP_RE = re.compile(r"(?P<timestamp>\d{8}_\d{6})$")
LogRunCatalogFingerprint = tuple[tuple[Any, ...], ...]


def _resolved_logs_root(logs_root: Path) -> Path:
    return logs_root.resolve()


def _resolve_log_path_under_root(path: Path, root: Path) -> Path | None:
    try:
        resolved = path.resolve()
        resolved.relative_to(root)
    except (OSError, ValueError):
        return None
    return resolved


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


class LogRunScanner:
    def __init__(
        self,
        *,
        logs_root: Path | str = "logs",
        cache_ttl_seconds: float = 30.0,
    ) -> None:
        self.logs_root = Path(logs_root)
        self.cache_ttl_seconds = max(0.0, float(cache_ttl_seconds))
        self._runs_cache: list[LogRun] | None = None
        self._runs_cache_fingerprint: LogRunCatalogFingerprint | None = None
        self._runs_cache_deadline = 0.0
        self._cache_lock = RLock()

    def list_runs(self) -> list[LogRun]:
        now = time.monotonic()
        with self._cache_lock:
            if self._runs_cache is not None and now < self._runs_cache_deadline:
                return list(self._runs_cache)

        root = self.resolved_root()
        if not root.exists():
            with self._cache_lock:
                self._runs_cache = []
                self._runs_cache_fingerprint = ()
                self._runs_cache_deadline = now + self.cache_ttl_seconds
            return []

        version_dirs, fingerprint = self._version_dirs_and_fingerprint(root)
        with self._cache_lock:
            if (
                self._runs_cache is not None
                and self._runs_cache_fingerprint == fingerprint
            ):
                self._runs_cache_deadline = now + self.cache_ttl_seconds
                return list(self._runs_cache)

        runs: list[LogRun] = []
        for version_dir in version_dirs:
            run = self.parse_run(root, version_dir)
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
        with self._cache_lock:
            self._runs_cache = list(sorted_runs)
            self._runs_cache_fingerprint = fingerprint
            self._runs_cache_deadline = now + self.cache_ttl_seconds
        return sorted_runs

    def clear_cache(self) -> None:
        with self._cache_lock:
            self._runs_cache = None
            self._runs_cache_fingerprint = None
            self._runs_cache_deadline = 0.0

    def _version_dirs_and_fingerprint(
        self,
        root: Path,
    ) -> tuple[list[Path], LogRunCatalogFingerprint]:
        version_dirs: list[Path] = []
        fingerprint: list[tuple[Any, ...]] = []
        for version_dir in sorted(root.rglob("version_*")):
            if not version_dir.is_dir():
                continue
            resolved = self.resolve_under_root(version_dir, root)
            if resolved is None:
                continue
            version_dirs.append(resolved)
            try:
                relative_path = resolved.relative_to(root).as_posix()
            except ValueError:
                continue
            fingerprint.append(("dir", relative_path, *self._path_stat(resolved)))
            for pattern in (
                "result.json",
                "hparams.yaml",
                "events.out.tfevents.*",
                "*.ckpt",
            ):
                for artifact in sorted(resolved.glob(pattern)):
                    artifact_resolved = self.resolve_under_root(artifact, root)
                    if artifact_resolved is None or not artifact_resolved.is_file():
                        continue
                    try:
                        artifact_relative = artifact_resolved.relative_to(
                            root
                        ).as_posix()
                    except ValueError:
                        continue
                    fingerprint.append(
                        ("file", artifact_relative, *self._path_stat(artifact_resolved))
                    )
        return version_dirs, tuple(fingerprint)

    def _path_stat(self, path: Path) -> tuple[int, int]:
        try:
            stat = path.stat()
        except OSError:
            return (0, 0)
        return (int(stat.st_mtime_ns), int(stat.st_size))

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
            experimentTask=(
                _read_result_experiment_task(result_path) if result_path else None
            ),
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


__all__ = [
    "LogRunCatalogFingerprint",
    "LogRunScanner",
]

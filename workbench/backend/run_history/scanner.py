"""Log Run catalog scanning and run parsing."""

from __future__ import annotations

import hashlib
import re
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from threading import RLock
from typing import Any, Literal

from emperor.model_packages import MODEL_CATALOG, model_id_from_payload

from workbench.backend.inspector.errors import InspectorError
from workbench.backend.log_experiments import is_valid_log_experiment_name
from workbench.backend.run_history.artifacts import (
    RunArtifactObservation,
    observe_run_artifacts,
)
from workbench.backend.run_history.paths import resolved_under_root
from workbench.backend.run_history.records import LogExperiment, LogRun

RUN_TIMESTAMP_RE = re.compile(r"(?P<timestamp>\d{8}_\d{6})$")
LogRunCatalogFingerprint = tuple[tuple[Any, ...], ...]
RunResultProjection = Literal["full", "summary", "none"]


def _resolved_logs_root(logs_root: Path) -> Path:
    return logs_root.resolve()


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
            public_id = model_id_from_payload({"model": candidate})
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
        self._artifact_observations: dict[str, RunArtifactObservation] = {}
        self._runs_cache_deadline = 0.0
        self._cache_generation = 0
        self._cache_lock = RLock()

    def list_runs(
        self,
        *,
        result_projection: RunResultProjection = "full",
    ) -> list[LogRun]:
        now = time.monotonic()
        cached_runs: list[LogRun] | None = None
        cached_observations: dict[str, RunArtifactObservation] = {}
        with self._cache_lock:
            generation = self._cache_generation
            if self._runs_cache is not None and now < self._runs_cache_deadline:
                cached_runs = list(self._runs_cache)
                cached_observations = dict(self._artifact_observations)
        if cached_runs is not None:
            return self._project_runs(
                cached_runs,
                cached_observations,
                result_projection=result_projection,
            )

        root = self.resolved_root()
        if not root.exists():
            with self._cache_lock:
                if generation == self._cache_generation:
                    self._runs_cache = []
                    self._runs_cache_fingerprint = ()
                    self._artifact_observations = {}
                    self._runs_cache_deadline = now + self.cache_ttl_seconds
            return []

        observations, fingerprint = self._version_dirs_and_fingerprint(root)
        with self._cache_lock:
            if (
                generation == self._cache_generation
                and self._runs_cache is not None
                and self._runs_cache_fingerprint == fingerprint
            ):
                self._runs_cache_deadline = now + self.cache_ttl_seconds
                cached_runs = list(self._runs_cache)
                cached_observations = dict(self._artifact_observations)
        if cached_runs is not None:
            return self._project_runs(
                cached_runs,
                cached_observations,
                result_projection=result_projection,
            )

        runs: list[LogRun] = []
        observations_by_path = {
            observation.run_dir.as_posix(): observation
            for observation in observations
        }
        for observation in observations:
            run = self.parse_run(
                root,
                observation.run_dir,
                artifacts=observation,
                result_projection="none",
            )
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
            if generation == self._cache_generation:
                self._runs_cache = list(sorted_runs)
                self._runs_cache_fingerprint = fingerprint
                self._artifact_observations = observations_by_path
                self._runs_cache_deadline = now + self.cache_ttl_seconds
        return self._project_runs(
            sorted_runs,
            observations_by_path,
            result_projection=result_projection,
        )

    def _project_runs(
        self,
        runs: list[LogRun],
        observations: dict[str, RunArtifactObservation],
        *,
        result_projection: RunResultProjection,
    ) -> list[LogRun]:
        if result_projection == "none":
            return list(runs)
        return [
            self._project_run(
                run,
                observations[run.path.as_posix()],
                result_projection=result_projection,
            )
            for run in runs
        ]

    def _project_run(
        self,
        run: LogRun,
        artifacts: RunArtifactObservation,
        *,
        result_projection: RunResultProjection,
    ) -> LogRun:
        if result_projection == "none":
            return run
        return replace(
            run,
            experimentTask=artifacts.experiment_task(),
            metrics=(
                artifacts.metrics() if result_projection == "full" else {}
            ),
        )

    def clear_cache(self) -> None:
        with self._cache_lock:
            self._cache_generation += 1
            self._runs_cache = None
            self._runs_cache_fingerprint = None
            self._artifact_observations = {}
            self._runs_cache_deadline = 0.0

    def _version_dirs_and_fingerprint(
        self,
        root: Path,
    ) -> tuple[list[RunArtifactObservation], LogRunCatalogFingerprint]:
        observations: list[RunArtifactObservation] = []
        fingerprint: list[tuple[Any, ...]] = []
        for version_dir in sorted(root.rglob("version_*")):
            if not version_dir.is_dir():
                continue
            resolved = self.resolve_under_root(version_dir, root)
            if resolved is None:
                continue
            try:
                relative_path = resolved.relative_to(root).as_posix()
            except ValueError:
                continue
            observation = observe_run_artifacts(resolved, root)
            observations.append(observation)
            fingerprint.append(
                (
                    "run",
                    relative_path,
                    observation.fingerprint,
                    observation.truncation_reasons,
                )
            )
        return observations, tuple(fingerprint)

    def list_experiments(self) -> list[LogExperiment]:
        root = self.resolved_root()
        if not root.exists():
            return []

        run_counts = Counter(
            run.experiment
            for run in self.list_runs(result_projection="none")
        )
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
        return resolved_under_root(path, root)

    def parse_run(
        self,
        root: Path,
        version_dir: Path,
        *,
        artifacts: RunArtifactObservation | None = None,
        result_projection: RunResultProjection = "full",
    ) -> LogRun | None:
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
        artifacts = artifacts or observe_run_artifacts(version_dir, root)

        run = LogRun(
            id=_run_id(relative_path),
            group=group,
            experiment=experiment,
            model=model,
            preset=preset,
            experimentTask=None,
            dataset=dataset,
            runName=run_name,
            timestamp=_display_timestamp(run_name),
            version=version,
            relativePath=relative_path,
            hasResult=artifacts.result is not None,
            eventFileCount=len(artifacts.event_artifacts),
            checkpointCount=len(artifacts.checkpoints),
            hasHparams=artifacts.hparams is not None,
            metrics={},
            path=version_dir,
            artifacts=artifacts,
        )
        return self._project_run(
            run,
            artifacts,
            result_projection=result_projection,
        )

    def project_run(
        self,
        run: LogRun,
        *,
        include_metrics: bool,
    ) -> LogRun:
        return self._project_run(
            run,
            self.artifact_observation(run),
            result_projection="full" if include_metrics else "summary",
        )

    def artifact_observation(self, run: LogRun) -> RunArtifactObservation:
        if isinstance(run.artifacts, RunArtifactObservation):
            return run.artifacts
        cache_key = run.path.as_posix()
        with self._cache_lock:
            cached = self._artifact_observations.get(cache_key)
        if cached is not None:
            return cached
        observation = observe_run_artifacts(run.path, self.resolved_root())
        with self._cache_lock:
            return self._artifact_observations.setdefault(
                cache_key,
                observation,
            )

    def resolve_runs(self, run_ids: list[str]) -> list[LogRun]:
        if not run_ids:
            return []
        runs_by_id = {
            run.id: run
            for run in self.list_runs(result_projection="none")
        }
        unknown = [run_id for run_id in run_ids if run_id not in runs_by_id]
        if unknown:
            raise InspectorError(f"Unknown log run id: {unknown[0]}")
        return [runs_by_id[run_id] for run_id in dict.fromkeys(run_ids)]

__all__ = [
    "LogRunCatalogFingerprint",
    "LogRunScanner",
]

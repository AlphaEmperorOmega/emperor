from __future__ import annotations

import hashlib
import re
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from threading import RLock
from typing import Any, Literal

from emperor_workbench.filesystem import PersistentJsonCatalog
from emperor_workbench.log_experiments import is_valid_log_experiment_name
from emperor_workbench.run_history._artifacts import (
    RunArtifactObservation,
    observe_run_artifacts,
)
from emperor_workbench.run_history._contracts import (
    KnownModelPackageIdentityResolver,
)
from emperor_workbench.run_history._errors import RunHistoryFailure
from emperor_workbench.run_history._paths import resolved_under_root
from emperor_workbench.run_history._records import LogExperiment, LogRun

RUN_TIMESTAMP_RE = re.compile(r"(?P<timestamp>\d{8}_\d{6})$")
LogRunCatalogFingerprint = tuple[tuple[Any, ...], ...]
LogRunCatalogGeneration = tuple[tuple[str, int, int, int], ...]
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


class LogRunScanner:
    def __init__(
        self,
        *,
        logs_root: Path | str = "logs",
        cache_ttl_seconds: float = 30.0,
        state_root: Path | None = None,
        model_identity_resolver: KnownModelPackageIdentityResolver,
    ) -> None:
        self.logs_root = Path(logs_root)
        self.cache_ttl_seconds = max(0.0, float(cache_ttl_seconds))
        self._model_identity_resolver = model_identity_resolver
        self._persistent_catalog = (
            PersistentJsonCatalog(
                state_root=state_root,
                name="run-history",
                authority_root=self.logs_root,
            )
            if state_root is not None
            else None
        )
        self._persistent_generation = 0
        self._runs_cache: list[LogRun] | None = None
        self._runs_cache_fingerprint: LogRunCatalogFingerprint | None = None
        self._runs_cache_catalog_generation: LogRunCatalogGeneration | None = None
        self._artifact_observations: dict[str, RunArtifactObservation] = {}
        self._runs_cache_deadline = 0.0
        self._cache_generation = 0
        self._cache_lock = RLock()

    def _split_log_model_prefix(
        self,
        prefix_parts: tuple[str, ...],
    ) -> tuple[tuple[str, ...], str] | None:
        for index in range(len(prefix_parts)):
            candidate = "/".join(prefix_parts[index:])
            model_id = self._model_identity_resolver(candidate)
            if model_id is not None:
                return prefix_parts[:index], model_id
        return None

    def list_runs(
        self,
        *,
        result_projection: RunResultProjection = "full",
    ) -> list[LogRun]:
        now = time.monotonic()
        root = self.resolved_root()
        catalog_generation = self._catalog_generation(root)
        cached_runs: list[LogRun] | None = None
        cached_observations: dict[str, RunArtifactObservation] = {}
        with self._cache_lock:
            generation = self._cache_generation
            if (
                self._runs_cache is not None
                and now < self._runs_cache_deadline
                and self._runs_cache_catalog_generation == catalog_generation
            ):
                cached_runs = list(self._runs_cache)
                cached_observations = dict(self._artifact_observations)
        if cached_runs is not None:
            return self._project_runs(
                cached_runs,
                cached_observations,
                result_projection=result_projection,
            )

        if self._runs_cache is None:
            persisted = self._load_persistent_catalog(root)
            if persisted is not None:
                persisted_runs, persisted_generation = persisted
                with self._cache_lock:
                    if generation == self._cache_generation:
                        self._runs_cache = persisted_runs
                        self._runs_cache_fingerprint = None
                        self._runs_cache_catalog_generation = catalog_generation
                        self._artifact_observations = {}
                        self._runs_cache_deadline = now + self.cache_ttl_seconds
                        self._persistent_generation = persisted_generation
                return self._project_runs(
                    persisted_runs,
                    {},
                    result_projection=result_projection,
                )

        if not root.exists():
            with self._cache_lock:
                if generation == self._cache_generation:
                    self._runs_cache = []
                    self._runs_cache_fingerprint = ()
                    self._runs_cache_catalog_generation = catalog_generation
                    self._artifact_observations = {}
                    self._runs_cache_deadline = now + self.cache_ttl_seconds
                    self._publish_persistent_catalog([], generation=1)
            return []

        observations, fingerprint = self._version_dirs_and_fingerprint(root)
        with self._cache_lock:
            if (
                generation == self._cache_generation
                and self._runs_cache is not None
                and self._runs_cache_fingerprint == fingerprint
            ):
                self._runs_cache_deadline = now + self.cache_ttl_seconds
                self._runs_cache_catalog_generation = catalog_generation
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
            observation.run_dir.as_posix(): observation for observation in observations
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
                run.run_name,
                run.version,
            ),
            reverse=True,
        )
        with self._cache_lock:
            if generation == self._cache_generation:
                self._runs_cache = list(sorted_runs)
                self._runs_cache_fingerprint = fingerprint
                self._runs_cache_catalog_generation = catalog_generation
                self._artifact_observations = observations_by_path
                self._runs_cache_deadline = now + self.cache_ttl_seconds
                self._persistent_generation += 1
                persistent_generation = self._persistent_generation
            else:
                persistent_generation = None
        if persistent_generation is not None:
            self._publish_persistent_catalog(
                sorted_runs,
                generation=persistent_generation,
            )
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
                observations.get(run.path.as_posix()) or self.artifact_observation(run),
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
            experiment_task=artifacts.experiment_task(),
            metrics=(artifacts.metrics() if result_projection == "full" else {}),
        )

    def clear_cache(self) -> None:
        with self._cache_lock:
            self._cache_generation += 1
            self._runs_cache = None
            self._runs_cache_fingerprint = None
            self._runs_cache_catalog_generation = None
            self._artifact_observations = {}
            self._runs_cache_deadline = 0.0
        if self._persistent_catalog is not None:
            self._persistent_catalog.invalidate()

    def reconcile_catalog(self) -> None:
        """Force one bounded reconciliation before an authoritative mutation."""

        with self._cache_lock:
            self._runs_cache_deadline = 0.0
            self._runs_cache_catalog_generation = None
        self.list_runs(result_projection="none")

    def invalidate_experiment(self, experiment: str) -> list[Path]:
        """Invalidate cached catalog state scoped by one Log Experiment."""

        with self._cache_lock:
            affected_paths = [
                run.path
                for run in (self._runs_cache or [])
                if run.experiment == experiment
            ]
            affected_keys = {path.as_posix() for path in affected_paths}
            for key in affected_keys:
                self._artifact_observations.pop(key, None)
            self._cache_generation += 1
            self._runs_cache = None
            self._runs_cache_fingerprint = None
            self._runs_cache_catalog_generation = None
            self._runs_cache_deadline = 0.0
        if self._persistent_catalog is not None:
            self._persistent_catalog.invalidate()
        return affected_paths

    def _catalog_generation(self, root: Path) -> LogRunCatalogGeneration:
        if not root.exists():
            return ()
        generation: list[tuple[str, int, int, int]] = []
        try:
            candidates = (root, *sorted(root.iterdir(), key=lambda path: path.name))
        except OSError:
            candidates = (root,)
        for catalog_path in candidates:
            try:
                path_stat = catalog_path.stat()
                relative = (
                    "."
                    if catalog_path == root
                    else catalog_path.relative_to(root).as_posix()
                )
            except (OSError, ValueError):
                continue
            if catalog_path.is_dir():
                generation.append(
                    (
                        relative,
                        int(path_stat.st_dev),
                        int(path_stat.st_ino),
                        int(path_stat.st_mtime_ns),
                    )
                )
        return tuple(sorted(generation))

    def _load_persistent_catalog(
        self,
        root: Path,
    ) -> tuple[list[LogRun], int] | None:
        catalog = self._persistent_catalog
        if catalog is None:
            return None
        payload = catalog.load(kind="run-history")
        if payload is None:
            return None
        entries = payload.get("entries")
        if not isinstance(entries, list):
            return None
        runs: list[LogRun] = []
        for entry in entries:
            if not isinstance(entry, dict):
                return None
            run = self._run_from_catalog_entry(root, entry)
            if run is None:
                return None
            runs.append(run)
        generation = payload["generation"]
        assert isinstance(generation, int)
        return runs, generation

    def _publish_persistent_catalog(
        self,
        runs: list[LogRun],
        *,
        generation: int,
    ) -> None:
        catalog = self._persistent_catalog
        if catalog is None:
            return
        catalog.publish(
            kind="run-history",
            generation=generation,
            entries=[self._run_catalog_entry(run) for run in runs],
        )

    @staticmethod
    def _run_catalog_entry(run: LogRun) -> dict[str, Any]:
        return {
            "id": run.id,
            "group": run.group,
            "experiment": run.experiment,
            "model": run.model,
            "preset": run.preset,
            "dataset": run.dataset,
            "runName": run.run_name,
            "timestamp": run.timestamp,
            "version": run.version,
            "relativePath": run.relative_path,
            "hasResult": run.has_result,
            "eventFileCount": run.event_file_count,
            "checkpointCount": run.checkpoint_count,
            "hasHparams": run.has_hparams,
        }

    def _run_from_catalog_entry(
        self,
        root: Path,
        entry: dict[str, Any],
    ) -> LogRun | None:
        try:
            relative_path = str(entry["relativePath"])
            candidate = root / relative_path
            resolved = self.resolve_under_root(candidate, root)
            if resolved is None or not resolved.is_dir():
                return None
            group = entry.get("group")
            timestamp = entry.get("timestamp")
            if group is not None and not isinstance(group, str):
                return None
            if timestamp is not None and not isinstance(timestamp, str):
                return None
            return LogRun(
                id=str(entry["id"]),
                group=group,
                experiment=str(entry["experiment"]),
                model=str(entry["model"]),
                preset=str(entry["preset"]),
                dataset=str(entry["dataset"]),
                run_name=str(entry["runName"]),
                timestamp=timestamp,
                version=str(entry["version"]),
                relative_path=relative_path,
                has_result=bool(entry["hasResult"]),
                event_file_count=int(entry["eventFileCount"]),
                checkpoint_count=int(entry["checkpointCount"]),
                has_hparams=bool(entry["hasHparams"]),
                path=resolved,
            )
        except (KeyError, TypeError, ValueError):
            return None

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
            run.experiment for run in self.list_runs(result_projection="none")
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
                    run_count=run_counts[child.name],
                    relative_path=child.name,
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
        model_prefix = self._split_log_model_prefix(parts[:-4])
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
            experiment_task=None,
            dataset=dataset,
            run_name=run_name,
            timestamp=_display_timestamp(run_name),
            version=version,
            relative_path=relative_path,
            has_result=artifacts.result is not None,
            event_file_count=len(artifacts.event_artifacts),
            checkpoint_count=len(artifacts.checkpoints),
            has_hparams=artifacts.hparams is not None,
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
        runs_by_id = {run.id: run for run in self.list_runs(result_projection="none")}
        unknown = [run_id for run_id in run_ids if run_id not in runs_by_id]
        if unknown:
            raise RunHistoryFailure(f"Unknown log run id: {unknown[0]}")
        return [runs_by_id[run_id] for run_id in dict.fromkeys(run_ids)]


__all__ = [
    "LogRunCatalogFingerprint",
    "LogRunCatalogGeneration",
    "LogRunScanner",
]

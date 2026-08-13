from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, cast, runtime_checkable

from filelock import FileLock
from filelock import Timeout as FileLockTimeout

from emperor.experiments import ExperimentTask
from model_runtime.packages.identity import ModelIdentity, model_key
from model_runtime.runs._metrics import sanitize_metric_payload
from model_runtime.runs.json_values import require_finite_json
from model_runtime.task_behavior import (
    CORE_RESULT_METRIC_KEYS,
    experiment_task_behavior,
)

DEFAULT_RESULT_METRIC_KEY_LIMIT = 512
DEFAULT_RESULT_STRING_VALUE_LIMIT = 20_000
_RUN_ARTIFACT_SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _validate_namespace(namespace: str | None) -> str | None:
    if namespace is None:
        return None
    if type(namespace) is not str:
        raise TypeError("log_folder must be a string or None")
    path = Path(namespace)
    if (
        not namespace
        or namespace in {".", ".."}
        or "\\" in namespace
        or path.is_absolute()
        or len(path.parts) != 1
    ):
        raise ValueError(
            "log_folder must be a single relative folder name without path separators"
        )
    return namespace


def _validate_path_segment(value: object, field_name: str) -> str:
    if (
        type(value) is not str
        or value in {"", ".", ".."}
        or value.strip() != value
        or _RUN_ARTIFACT_SEGMENT_RE.fullmatch(value) is None
    ):
        raise ValueError(f"{field_name} must be one safe relative path segment.")
    return value


def _model_id(identity: object) -> str:
    if not isinstance(identity, ModelIdentity):
        raise TypeError("Run Artifact paths require a ModelIdentity.")
    return model_key(identity.model_type, identity.model)


def _resolved_contained_path(root: Path, path: Path) -> Path:
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    if not resolved_path.is_relative_to(resolved_root):
        raise ValueError(
            f"Run Artifact path '{path}' is outside artifact root '{root}'."
        )
    return resolved_path


def _best_results_path(root: Path, model_root: Path) -> Path:
    return _resolved_contained_path(root, model_root / "best_results.json")


def _result_path(root: Path, log_dir: str | Path) -> Path:
    contained_log_dir = _resolved_contained_path(root, Path(log_dir))
    return _resolved_contained_path(root, contained_log_dir / "result.json")


def _best_results_lock_path(root: Path, summary_path: Path) -> Path:
    return _resolved_contained_path(
        root,
        summary_path.with_suffix(summary_path.suffix + ".lock"),
    )


@runtime_checkable
class RunArtifacts(Protocol):
    """Portable Run Artifact lifecycle consumed by generic execution."""

    @property
    def root(self) -> Path: ...

    def run_name(
        self,
        identity: ModelIdentity,
        preset_key: str,
        dataset: str,
        parameters: Mapping[str, Any],
    ) -> str: ...

    def result_metrics_payload(
        self,
        metrics: Mapping[Any, Any],
    ) -> dict[str, Any]: ...

    def write_result(
        self,
        log_dir: str | Path,
        result: Mapping[str, Any],
    ) -> Path:
        """Commit the terminal receipt for one completed artifact attempt."""
        ...

    def read_best_results(
        self,
        identity: ModelIdentity,
    ) -> dict[str, Any]: ...

    def update_best_results(
        self,
        identity: ModelIdentity,
        experiment_task: ExperimentTask | None,
        result: Mapping[str, Any],
    ) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class RunArtifactReservation:
    """One exclusively claimed filesystem Run Artifact directory."""

    name: str
    version: int
    log_dir: Path


@dataclass(frozen=True, slots=True)
class FilesystemRunArtifacts:
    """Atomic filesystem Implementation of the Run Artifact Interface.

    Final ``version_N`` directory allocation is exclusive across processes.
    Other Run Artifact Adapters retain responsibility for their own namespace
    allocation policy.

    Resolved-path checks contain existing symlinks. They assume an actor cannot
    concurrently replace checked path components between validation and I/O.
    """

    root: Path = Path("logs")
    namespace: str | None = None
    clock: Callable[[], datetime] = datetime.now
    best_results_lock_timeout_seconds: float = field(default=30.0, kw_only=True)

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))
        object.__setattr__(self, "namespace", _validate_namespace(self.namespace))
        timeout = self.best_results_lock_timeout_seconds
        if (
            type(timeout) not in (int, float)
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError(
                "best_results_lock_timeout_seconds must be a finite positive number."
            )
        object.__setattr__(self, "best_results_lock_timeout_seconds", float(timeout))

    def model_root(self, identity: ModelIdentity) -> Path:
        root = self.root
        if self.namespace is not None:
            root = root / self.namespace
        return _resolved_contained_path(self.root, root / _model_id(identity))

    def best_results_path(self, identity: ModelIdentity) -> Path:
        return _best_results_path(self.root, self.model_root(identity))

    def run_name(
        self,
        identity: ModelIdentity,
        preset_key: str,
        dataset: str,
        parameters: Mapping[str, Any],
    ) -> str:
        preset_key = _validate_path_segment(preset_key, "preset_key")
        dataset = _validate_path_segment(dataset, "dataset")
        param_string = "_".join(f"{key}={value}" for key, value in parameters.items())
        parameter_id = (
            hashlib.md5(
                param_string.encode(),
                usedforsecurity=False,
            ).hexdigest()[:8]
            if param_string
            else "default"
        )
        timestamp = self.clock().strftime("%Y%m%d_%H%M%S")
        model_id = _model_id(identity)
        prefix = (
            f"{self.namespace}/{model_id}" if self.namespace is not None else model_id
        )
        return f"{prefix}/{preset_key}/{dataset}/{parameter_id}_{timestamp}"

    def reserve_run(
        self,
        identity: ModelIdentity,
        preset_key: str,
        dataset: str,
        parameters: Mapping[str, Any],
    ) -> RunArtifactReservation:
        """Atomically claim a final ``version_N`` directory for one Run.

        Creating the directory is the reservation. A process failure may leave
        an empty failed-attempt directory, which is intentionally never reused.
        """

        name = self.run_name(identity, preset_key, dataset, parameters)
        root = self.root
        run_dir = _resolved_contained_path(root, root / name)
        run_dir.mkdir(parents=True, exist_ok=True)
        run_dir = _resolved_contained_path(root, run_dir)

        version = 0
        while True:
            log_dir = _resolved_contained_path(root, run_dir / f"version_{version}")
            try:
                log_dir.mkdir()
            except FileExistsError:
                version += 1
                continue
            return RunArtifactReservation(
                name=name,
                version=version,
                log_dir=_resolved_contained_path(root, log_dir),
            )

    def result_metrics_payload(
        self,
        metrics: Mapping[Any, Any],
    ) -> dict[str, Any]:
        sanitized, original_count, dropped_count = sanitize_metric_payload(
            metrics,
            metric_key_limit=DEFAULT_RESULT_METRIC_KEY_LIMIT,
            string_value_limit=DEFAULT_RESULT_STRING_VALUE_LIMIT,
            protected_metric_keys=CORE_RESULT_METRIC_KEYS,
            deterministic_selection=True,
        )
        payload: dict[str, Any] = {"metrics": sanitized}
        if dropped_count > 0:
            payload["metricsOriginalCount"] = original_count
            payload["metricsDroppedCount"] = dropped_count
        return payload

    def write_result(
        self,
        log_dir: str | Path,
        result: Mapping[str, Any],
    ) -> Path:
        """Atomically commit the authoritative completed-Run receipt."""

        result_path = _result_path(self.root, log_dir)
        self._write_json_atomic(result_path, result, trailing_newline=False)
        return result_path

    def read_result(self, log_dir: str | Path) -> dict[str, Any]:
        """Read one contained terminal receipt for explicit retry admission."""

        result_path = _result_path(self.root, log_dir)
        if not result_path.is_file():
            raise ValueError(f"Run Artifact '{result_path}' has no result receipt.")
        try:
            payload: object = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, RecursionError, ValueError) as exc:
            raise ValueError(
                f"Run Artifact '{result_path}' has an invalid result receipt."
            ) from exc
        if not isinstance(payload, dict):
            raise ValueError(
                f"Run Artifact '{result_path}' has an invalid result receipt."
            )
        return cast(dict[str, Any], payload)

    def read_best_results(self, identity: ModelIdentity) -> dict[str, Any]:
        return self._read_json_object(self.best_results_path(identity))

    def update_best_results(
        self,
        identity: ModelIdentity,
        experiment_task: ExperimentTask | None,
        result: Mapping[str, Any],
    ) -> dict[str, Any]:
        summary_path = self.best_results_path(identity)
        with self._best_results_lock(summary_path):
            merged = self._read_json_object(summary_path)
            dataset = str(result["dataset"])
            runs = list(merged.get(dataset, []))
            artifact_id = self._artifact_id(result)
            if artifact_id is not None:
                matching = self._artifact_matches(merged, artifact_id)
                if matching:
                    normalized = self._normalized_result(result)
                    if any(
                        candidate_dataset != dataset
                        or self._normalized_result(candidate) != normalized
                        for candidate_dataset, candidate in matching
                    ):
                        raise ValueError(
                            "Best-results artifactId "
                            f"'{artifact_id}' already has different content."
                        )
                    return merged
            new_score = self._ranking_score(experiment_task, result)
            worst_score = min(
                (self._ranking_score(experiment_task, candidate) for candidate in runs),
                default=(float("-inf"), float("-inf")),
            )
            if len(runs) < 5 or new_score > worst_score:
                runs.append(dict(result))
                merged[dataset] = [
                    {**candidate, "rank": rank}
                    for rank, candidate in enumerate(
                        sorted(
                            runs,
                            key=lambda candidate: self._ranking_score(
                                experiment_task,
                                candidate,
                            ),
                            reverse=True,
                        )[:5],
                        start=1,
                    )
                ]
                self._write_json_atomic(summary_path, merged, trailing_newline=True)
        return merged

    @staticmethod
    def _artifact_id(result: Mapping[str, Any]) -> str | None:
        if "artifactId" not in result:
            return None
        artifact_id = result["artifactId"]
        if type(artifact_id) is not str or not artifact_id:
            raise ValueError("Best-results artifactId must be a non-empty string.")
        return artifact_id

    @staticmethod
    def _artifact_matches(
        summary: Mapping[str, Any],
        artifact_id: str,
    ) -> tuple[tuple[str, Mapping[str, Any]], ...]:
        matches: list[tuple[str, Mapping[str, Any]]] = []
        for dataset, candidates in summary.items():
            if not isinstance(candidates, list):
                continue
            for value in cast(Sequence[object], candidates):
                if not isinstance(value, dict):
                    continue
                candidate = cast(dict[str, Any], value)
                if candidate.get("artifactId") == artifact_id:
                    matches.append((dataset, candidate))
        return tuple(matches)

    @staticmethod
    def _normalized_result(result: Mapping[str, Any]) -> object:
        payload = {key: value for key, value in result.items() if key != "rank"}
        require_finite_json(payload)
        return json.loads(json.dumps(payload, default=str, sort_keys=True))

    @staticmethod
    def _ranking_score(
        experiment_task: ExperimentTask | None,
        result: Mapping[str, Any],
    ) -> tuple[float, float]:
        task = experiment_task or ExperimentTask.IMAGE_CLASSIFICATION
        return experiment_task_behavior(task).ranking_score(result)

    @staticmethod
    def _read_json_object(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        payload: object = json.loads(path.read_text(encoding="utf-8"))
        return cast(dict[str, Any], payload) if isinstance(payload, dict) else {}

    @staticmethod
    def _write_json_atomic(
        path: Path,
        payload: Mapping[str, Any],
        *,
        trailing_newline: bool,
    ) -> None:
        require_finite_json(payload)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                delete=False,
                dir=path.parent,
                encoding="utf-8",
                prefix=f".{path.name}.",
                suffix=".tmp",
            ) as temp_file:
                temp_path = Path(temp_file.name)
                json.dump(payload, temp_file, indent=2, default=str)
                if trailing_newline:
                    temp_file.write("\n")
            os.replace(temp_path, path)
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()

    @contextmanager
    def _best_results_lock(self, summary_path: Path) -> Generator[None]:
        lock_path = _best_results_lock_path(self.root, summary_path)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with FileLock(
                str(lock_path),
                timeout=self.best_results_lock_timeout_seconds,
            ):
                yield
        except FileLockTimeout as exc:
            raise TimeoutError("Timed out acquiring the best-results lock.") from exc


__all__ = [
    "FilesystemRunArtifacts",
    "RunArtifactReservation",
    "RunArtifacts",
]

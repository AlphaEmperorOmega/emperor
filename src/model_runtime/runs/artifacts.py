from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from filelock import FileLock

from emperor.experiments import ExperimentTask
from model_runtime.packages.identity import ModelIdentity
from model_runtime.runs._metrics import sanitize_metric_payload
from model_runtime.runs.json_values import require_finite_json
from model_runtime.task_behavior import experiment_task_behavior

DEFAULT_RESULT_METRIC_KEY_LIMIT = 512
DEFAULT_RESULT_STRING_VALUE_LIMIT = 20_000


@runtime_checkable
class RunArtifacts(Protocol):
    """Portable Run Artifact lifecycle consumed by generic execution."""

    @property
    def root(self) -> Path: ...

    def run_name(
        self,
        identity: ModelIdentity | str,
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
    ) -> Path: ...

    def read_best_results(
        self,
        identity: ModelIdentity | str,
    ) -> dict[str, Any]: ...

    def update_best_results(
        self,
        identity: ModelIdentity | str,
        experiment_task: ExperimentTask | None,
        result: Mapping[str, Any],
    ) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class FilesystemRunArtifacts:
    """Atomic filesystem Implementation of the Run Artifact Interface."""

    root: Path = Path("logs")
    namespace: str | None = None
    clock: Callable[[], datetime] = datetime.now

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))
        object.__setattr__(self, "namespace", self._validate_namespace(self.namespace))

    @staticmethod
    def _validate_namespace(namespace: str | None) -> str | None:
        if namespace is None:
            return None
        folder = str(namespace)
        path = Path(folder)
        if (
            not folder
            or folder in {".", ".."}
            or "\\" in folder
            or path.is_absolute()
            or len(path.parts) != 1
        ):
            raise ValueError(
                "log_folder must be a single relative folder name without path "
                "separators"
            )
        return folder

    @staticmethod
    def _model_id(identity: ModelIdentity | str) -> str:
        return identity.catalog_key if isinstance(identity, ModelIdentity) else identity

    def model_root(self, identity: ModelIdentity | str) -> Path:
        root = self.root
        if self.namespace is not None:
            root = root / self.namespace
        return root / self._model_id(identity)

    def best_results_path(self, identity: ModelIdentity | str) -> Path:
        return self.model_root(identity) / "best_results.json"

    def run_name(
        self,
        identity: ModelIdentity | str,
        preset_key: str,
        dataset: str,
        parameters: Mapping[str, Any],
    ) -> str:
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
        model_id = self._model_id(identity)
        prefix = (
            f"{self.namespace}/{model_id}" if self.namespace is not None else model_id
        )
        return f"{prefix}/{preset_key}/{dataset}/{parameter_id}_{timestamp}"

    def result_metrics_payload(
        self,
        metrics: Mapping[Any, Any],
    ) -> dict[str, Any]:
        sanitized, original_count, dropped_count = sanitize_metric_payload(
            metrics,
            metric_key_limit=DEFAULT_RESULT_METRIC_KEY_LIMIT,
            string_value_limit=DEFAULT_RESULT_STRING_VALUE_LIMIT,
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
        result_path = Path(log_dir) / "result.json"
        self._write_json_atomic(result_path, result, trailing_newline=False)
        return result_path

    def read_best_results(self, identity: ModelIdentity | str) -> dict[str, Any]:
        return self._read_json_object(self.best_results_path(identity))

    def update_best_results(
        self,
        identity: ModelIdentity | str,
        experiment_task: ExperimentTask | None,
        result: Mapping[str, Any],
    ) -> dict[str, Any]:
        summary_path = self.best_results_path(identity)
        with self._best_results_lock(summary_path):
            merged = self._read_json_object(summary_path)
            dataset = str(result["dataset"])
            runs = list(merged.get(dataset, []))
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
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}

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
    def _best_results_lock(self, summary_path: Path) -> Iterator[None]:
        lock_path = summary_path.with_suffix(summary_path.suffix + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(lock_path)):
            yield


__all__ = ["FilesystemRunArtifacts", "RunArtifacts"]

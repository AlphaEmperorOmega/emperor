from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from emperor.experiments import ExperimentTask
from model_runtime.packages.identity import ModelIdentity
from model_runtime.runs.json_values import require_finite_json
from model_runtime.runs.locking import exclusive_file_lock
from model_runtime.runs.progress import sanitize_metric_payload
from model_runtime.task_behavior import experiment_task_behavior

DEFAULT_RESULT_METRIC_KEY_LIMIT = 512
DEFAULT_RESULT_STRING_VALUE_LIMIT = 20_000


def _model_id(identity: ModelIdentity | str) -> str:
    return identity.catalog_key if isinstance(identity, ModelIdentity) else identity


def validate_artifact_namespace(namespace: str | None) -> str | None:
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
            "log_folder must be a single relative folder name without path separators"
        )
    return folder


def result_metrics_payload(metrics: dict[Any, Any]) -> dict[str, Any]:
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


def result_ranking_score(
    experiment_task: ExperimentTask | None,
    result: Mapping[str, Any],
) -> tuple[float, float]:
    task = experiment_task or ExperimentTask.IMAGE_CLASSIFICATION
    return experiment_task_behavior(task).ranking_score(result)


@contextmanager
def _best_results_lock(summary_path: Path):
    lock_path = summary_path.with_suffix(summary_path.suffix + ".lock")
    with exclusive_file_lock(lock_path):
        yield


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
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
            temp_file.write("\n")
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def write_run_result(log_dir: str | Path, result: Mapping[str, Any]) -> Path:
    require_finite_json(result)
    result_path = Path(log_dir) / "result.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, default=str))
    return result_path


@dataclass(frozen=True, slots=True)
class FilesystemRunArtifacts:
    root: Path = Path("logs")
    namespace: str | None = None
    clock: Callable[[], datetime] = datetime.now

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))
        object.__setattr__(
            self,
            "namespace",
            validate_artifact_namespace(self.namespace),
        )

    def model_root(self, identity: ModelIdentity | str) -> Path:
        root = self.root
        if self.namespace is not None:
            root = root / self.namespace
        return root / _model_id(identity)

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
        prefix = (
            f"{self.namespace}/{_model_id(identity)}"
            if self.namespace is not None
            else _model_id(identity)
        )
        return f"{prefix}/{preset_key}/{dataset}/{parameter_id}_{timestamp}"

    def read_best_results(self, identity: ModelIdentity | str) -> dict[str, Any]:
        return _read_json_object(self.best_results_path(identity))

    def update_best_results(
        self,
        identity: ModelIdentity | str,
        experiment_task: ExperimentTask | None,
        result: Mapping[str, Any],
        current: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        summary_path = self.best_results_path(identity)
        with _best_results_lock(summary_path):
            merged = _read_json_object(summary_path)
            dataset = str(result["dataset"])
            runs = list(merged.get(dataset, []))
            new_score = result_ranking_score(experiment_task, result)
            worst_score = min(
                (
                    result_ranking_score(experiment_task, candidate)
                    for candidate in runs
                ),
                default=(float("-inf"), float("-inf")),
            )
            if len(runs) < 5 or new_score > worst_score:
                runs.append(dict(result))
                merged[dataset] = [
                    {**candidate, "rank": rank}
                    for rank, candidate in enumerate(
                        sorted(
                            runs,
                            key=lambda candidate: result_ranking_score(
                                experiment_task,
                                candidate,
                            ),
                            reverse=True,
                        )[:5],
                        start=1,
                    )
                ]
                _write_json_atomic(summary_path, merged)
        if current is not None:
            current.clear()
            current.update(merged)
        return merged


__all__ = [
    "DEFAULT_RESULT_METRIC_KEY_LIMIT",
    "DEFAULT_RESULT_STRING_VALUE_LIMIT",
    "FilesystemRunArtifacts",
    "result_metrics_payload",
    "result_ranking_score",
    "validate_artifact_namespace",
    "write_run_result",
]

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, cast, runtime_checkable

from model_runtime.runs._metrics import sanitize_metric_payload, truncate_string
from model_runtime.runs._progress_events import (
    MODEL_RUNTIME_PROGRESS_CONTEXT_FIELDS,
    MODEL_RUNTIME_PROGRESS_EVENT_FIELDS,
    MODEL_RUNTIME_PROGRESS_EVENT_TYPES,
    MODEL_RUNTIME_PROGRESS_OPTIONAL_FIELDS,
    RunProgressEventInput,
    project_run_progress_context,
    project_run_progress_event,
)
from model_runtime.runs.json_values import require_finite_json

DEFAULT_PROGRESS_METRIC_KEY_LIMIT = 512
DEFAULT_PROGRESS_EVENT_BYTE_LIMIT = 128_000
DEFAULT_PROGRESS_STRING_VALUE_LIMIT = 20_000


@runtime_checkable
class RunProgress(Protocol):
    """Destination accepting typed Run events or explicit legacy wire mappings."""

    def write_event(self, event: RunProgressEventInput) -> None: ...


@dataclass(frozen=True, slots=True)
class RunProgressContext:
    experiment_task: str | None
    dataset: str
    preset: str
    preset_key: str
    log_dir: str | None
    run_id: str | None
    run_index: int | None
    run_total: int | None
    total_epochs: int

    def with_log_dir(self, log_dir: str) -> RunProgressContext:
        return replace(self, log_dir=log_dir)

    def event_fields(self) -> dict[str, Any]:
        return project_run_progress_context(self)


@dataclass(frozen=True, slots=True)
class ContextualRunProgress:
    destination: RunProgress
    context: RunProgressContext

    def with_log_dir(self, log_dir: str) -> ContextualRunProgress:
        return replace(self, context=self.context.with_log_dir(log_dir))

    def write_event(self, event: RunProgressEventInput) -> None:
        self.destination.write_event(
            project_run_progress_event(event, context=self.context)
        )


def require_run_progress(progress: object) -> RunProgress:
    if not isinstance(progress, RunProgress):
        raise TypeError("Runs progress must define write_event(event).")
    return progress


def contextual_run_progress(
    progress: RunProgress | None,
    context: RunProgressContext,
) -> ContextualRunProgress | None:
    return (
        ContextualRunProgress(destination=progress, context=context)
        if progress is not None
        else None
    )


def _encoded_size(payload: dict[str, Any]) -> int:
    return len(json.dumps(payload, default=str).encode("utf-8"))


@dataclass(frozen=True, slots=True)
class JsonlRunProgress:
    path: Path
    metric_key_limit: int = DEFAULT_PROGRESS_METRIC_KEY_LIMIT
    event_byte_limit: int = DEFAULT_PROGRESS_EVENT_BYTE_LIMIT
    string_value_limit: int = DEFAULT_PROGRESS_STRING_VALUE_LIMIT

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(
            self,
            "metric_key_limit",
            max(0, int(self.metric_key_limit)),
        )
        object.__setattr__(
            self,
            "event_byte_limit",
            max(0, int(self.event_byte_limit)),
        )
        object.__setattr__(
            self,
            "string_value_limit",
            max(0, int(self.string_value_limit)),
        )

    def write_event(self, event: RunProgressEventInput) -> None:
        event_payload = self._sanitize_event(event)
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            **event_payload,
        }
        require_finite_json(payload)
        self._enforce_event_byte_limit(payload)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, default=str) + "\n")

    def _sanitize_event(self, event: RunProgressEventInput) -> dict[str, Any]:
        payload = project_run_progress_event(event)
        for key in ("error", "traceback"):
            value = payload.get(key)
            if isinstance(value, str):
                payload[key] = truncate_string(value, self.string_value_limit)
        metrics = payload.get("metrics")
        if isinstance(metrics, dict):
            typed_metrics = cast(dict[str, Any], metrics)
            sanitized, original_count, dropped_count = sanitize_metric_payload(
                typed_metrics,
                metric_key_limit=self.metric_key_limit,
                string_value_limit=self.string_value_limit,
            )
            payload["metrics"] = sanitized
            if dropped_count > 0:
                payload["metricsOriginalCount"] = original_count
                payload["metricsDroppedCount"] = dropped_count
        return payload

    def _enforce_event_byte_limit(self, payload: dict[str, Any]) -> None:
        if self.event_byte_limit <= 0:
            return
        metrics = payload.get("metrics")
        typed_metrics = (
            cast(dict[str, Any], metrics) if isinstance(metrics, dict) else None
        )
        while (
            typed_metrics is not None
            and typed_metrics
            and _encoded_size(payload) > self.event_byte_limit
        ):
            key = next(reversed(typed_metrics))
            del typed_metrics[key]
            payload["metricsDroppedCount"] = (
                int(payload.get("metricsDroppedCount") or 0) + 1
            )
            payload["metricsOriginalCount"] = int(
                payload.get("metricsOriginalCount") or len(typed_metrics) + 1
            )
        encoded_size = _encoded_size(payload)
        if encoded_size > self.event_byte_limit:
            raise ValueError(
                "Training progress event exceeds the "
                f"{self.event_byte_limit} byte record limit."
            )


__all__ = [
    "ContextualRunProgress",
    "DEFAULT_PROGRESS_EVENT_BYTE_LIMIT",
    "DEFAULT_PROGRESS_METRIC_KEY_LIMIT",
    "DEFAULT_PROGRESS_STRING_VALUE_LIMIT",
    "JsonlRunProgress",
    "MODEL_RUNTIME_PROGRESS_CONTEXT_FIELDS",
    "MODEL_RUNTIME_PROGRESS_EVENT_FIELDS",
    "MODEL_RUNTIME_PROGRESS_EVENT_TYPES",
    "MODEL_RUNTIME_PROGRESS_OPTIONAL_FIELDS",
    "RunProgress",
    "RunProgressContext",
    "contextual_run_progress",
    "require_run_progress",
]

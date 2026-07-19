from __future__ import annotations

import json
import math
import traceback
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lightning.pytorch.callbacks import Callback

from model_runtime.runs.json_values import require_finite_json

DEFAULT_PROGRESS_METRIC_KEY_LIMIT = 512
DEFAULT_PROGRESS_EVENT_BYTE_LIMIT = 128_000
DEFAULT_PROGRESS_STRING_VALUE_LIMIT = 20_000
_DROPPED_METRIC_TOKENS = ("confusion_matrix", "per_class")
CLUSTER_COORDINATE_SAMPLE_LIMIT = 100
NEURON_ADDED_BURST_LIMIT = 100


def _truncate_string(value: str, limit: int) -> str:
    if limit <= 0 or len(value) <= limit:
        return value
    return f"{value[:limit]}...[truncated {len(value) - limit} chars]"


def _json_value(value: Any, *, string_value_limit: int | None = None) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, str) and string_value_limit is not None:
            return _truncate_string(value, string_value_limit)
        if isinstance(value, float) and not math.isfinite(value):
            require_finite_json(value)
        return value
    rendered = str(value)
    if string_value_limit is not None:
        return _truncate_string(rendered, string_value_limit)
    return rendered


def _metric_key_is_dropped(key: str) -> bool:
    normalized = key.replace("\\", "/").lower()
    return any(token in normalized for token in _DROPPED_METRIC_TOKENS)


def sanitize_metric_payload(
    metrics: dict[Any, Any],
    *,
    metric_key_limit: int = DEFAULT_PROGRESS_METRIC_KEY_LIMIT,
    string_value_limit: int = DEFAULT_PROGRESS_STRING_VALUE_LIMIT,
) -> tuple[dict[str, Any], int, int]:
    sanitized: dict[str, Any] = {}
    dropped_count = 0
    safe_metric_key_limit = max(0, int(metric_key_limit))
    for raw_key, value in metrics.items():
        key = str(raw_key)
        if _metric_key_is_dropped(key):
            dropped_count += 1
            continue
        if len(sanitized) >= safe_metric_key_limit:
            dropped_count += 1
            continue
        sanitized[key] = _json_value(
            value,
            string_value_limit=string_value_limit,
        )
    return sanitized, len(metrics), dropped_count


def _encoded_size(payload: dict[str, Any]) -> int:
    return len(json.dumps(payload, default=str).encode("utf-8"))


def _coordinate_from_neuron_name(name: str) -> list[int] | None:
    parts = name.split("_")
    if len(parts) != 4 or parts[0] != "neuron":
        return None
    try:
        return [int(parts[1]), int(parts[2]), int(parts[3])]
    except ValueError:
        return None


class NeuronClusterGrowthCallback(Callback):
    """Emit portable progress events for dynamic neuron-cluster growth."""

    def __init__(self, write_event: Callable[[dict[str, Any]], None]) -> None:
        super().__init__()
        self._write_event = write_event
        self._clusters: list[tuple[str, Any]] = []
        self._known_names: dict[str, set[str]] = {}

    def on_fit_start(self, trainer, pl_module) -> None:
        from emperor.neuron import NeuronCluster

        self._clusters = [
            (name, module)
            for name, module in pl_module.named_modules()
            if isinstance(module, NeuronCluster)
        ]
        for name, cluster in self._clusters:
            names = set(cluster.cluster.keys())
            self._known_names[name] = names
            self._write_event(
                {
                    "type": "cluster_initialized",
                    "node": name,
                    "count": len(names),
                    "capacity": self._capacity(cluster),
                    **self._coordinate_sample_payload(names),
                }
            )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        for name, cluster in self._clusters:
            current = set(cluster.cluster.keys())
            new_names = current - self._known_names.get(name, set())
            if not new_names:
                continue
            self._known_names[name] = current
            coordinates = [
                coordinate
                for coordinate in (
                    _coordinate_from_neuron_name(new_name)
                    for new_name in sorted(new_names)
                )
                if coordinate is not None
            ]
            if not coordinates:
                continue
            epoch = int(getattr(trainer, "current_epoch", 0))
            step = int(getattr(trainer, "global_step", 0))
            if len(coordinates) > NEURON_ADDED_BURST_LIMIT:
                self._write_event(
                    {
                        "type": "neurons_added",
                        "node": name,
                        "coordinates": coordinates[:CLUSTER_COORDINATE_SAMPLE_LIMIT],
                        "coordinateCount": len(coordinates),
                        "coordinatesTruncated": (
                            len(coordinates) > CLUSTER_COORDINATE_SAMPLE_LIMIT
                        ),
                        "count": len(current),
                        "capacity": self._capacity(cluster),
                        "epoch": epoch,
                        "step": step,
                    }
                )
                continue
            for coordinate in coordinates:
                self._write_event(
                    {
                        "type": "neuron_added",
                        "node": name,
                        "coord": coordinate,
                        "count": len(current),
                        "capacity": self._capacity(cluster),
                        "epoch": epoch,
                        "step": step,
                    }
                )

    def _capacity(self, cluster) -> list[int]:
        return [
            cluster.x_axis_total_neurons,
            cluster.y_axis_total_neurons,
            cluster.z_axis_total_neurons,
        ]

    def _coordinates(self, names: set[str]) -> list[list[int]]:
        coordinates = (_coordinate_from_neuron_name(name) for name in names)
        return sorted(
            coordinate for coordinate in coordinates if coordinate is not None
        )

    def _coordinate_sample_payload(self, names: set[str]) -> dict[str, Any]:
        coordinates = self._coordinates(names)
        sampled = coordinates[:CLUSTER_COORDINATE_SAMPLE_LIMIT]
        return {
            "coordinates": sampled,
            "coordinateCount": len(coordinates),
            "coordinatesTruncated": len(coordinates) > len(sampled),
        }

    def on_fit_end(self, trainer, pl_module) -> None:
        self._clusters = []
        self._known_names.clear()


class JsonlTrainingProgressCallback(Callback):
    def __init__(
        self,
        path: str | Path,
        step_interval: int = 1,
        *,
        metric_key_limit: int = DEFAULT_PROGRESS_METRIC_KEY_LIMIT,
        event_byte_limit: int = DEFAULT_PROGRESS_EVENT_BYTE_LIMIT,
        string_value_limit: int = DEFAULT_PROGRESS_STRING_VALUE_LIMIT,
    ) -> None:
        super().__init__()
        self.path = Path(path)
        self.step_interval = max(1, int(step_interval))
        self.metric_key_limit = max(0, int(metric_key_limit))
        self.event_byte_limit = max(0, int(event_byte_limit))
        self.string_value_limit = max(0, int(string_value_limit))
        self.dataset: str | None = None
        self.preset: str | None = None
        self.preset_key: str | None = None
        self.log_dir: str | None = None
        self.run_id: str | None = None
        self.run_index: int | None = None
        self.run_total: int | None = None
        self.total_epochs: int | None = None

    def set_run_context(
        self,
        dataset: str,
        log_dir: str | None = None,
        preset: str | None = None,
        preset_key: str | None = None,
        run_id: str | None = None,
        run_index: int | None = None,
        run_total: int | None = None,
        total_epochs: int | None = None,
    ) -> None:
        self.dataset = dataset
        self.log_dir = log_dir
        self.preset = preset
        self.preset_key = preset_key
        self.run_id = run_id
        self.run_index = run_index
        self.run_total = run_total
        self.total_epochs = total_epochs

    def write_event(self, event: dict[str, Any]) -> None:
        event_payload = self._sanitize_event(event)
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "dataset": self.dataset,
            "preset": self.preset,
            "presetKey": self.preset_key,
            "logDir": self.log_dir,
            "runId": self.run_id,
            "runIndex": self.run_index,
            "runTotal": self.run_total,
            "totalEpochs": self.total_epochs,
            **event_payload,
        }
        require_finite_json(payload)
        self._enforce_event_byte_limit(payload)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, default=str) + "\n")

    def _metrics(self, trainer) -> dict[str, Any]:
        return {
            key: _json_value(value)
            for key, value in getattr(trainer, "callback_metrics", {}).items()
        }

    def _sanitize_event(self, event: dict[str, Any]) -> dict[str, Any]:
        payload = dict(event)
        for key in ("error", "traceback"):
            value = payload.get(key)
            if isinstance(value, str):
                payload[key] = _truncate_string(value, self.string_value_limit)
        metrics = payload.get("metrics")
        if isinstance(metrics, dict):
            sanitized, original_count, dropped_count = self._sanitize_metrics(metrics)
            payload["metrics"] = sanitized
            if dropped_count > 0:
                payload["metricsOriginalCount"] = original_count
                payload["metricsDroppedCount"] = dropped_count
        return payload

    def _sanitize_metrics(
        self,
        metrics: dict[Any, Any],
    ) -> tuple[dict[str, Any], int, int]:
        return sanitize_metric_payload(
            metrics,
            metric_key_limit=self.metric_key_limit,
            string_value_limit=self.string_value_limit,
        )

    def _enforce_event_byte_limit(self, payload: dict[str, Any]) -> None:
        if self.event_byte_limit <= 0:
            return
        metrics = payload.get("metrics")
        while (
            isinstance(metrics, dict)
            and metrics
            and _encoded_size(payload) > self.event_byte_limit
        ):
            key = next(reversed(metrics))
            del metrics[key]
            payload["metricsDroppedCount"] = (
                int(payload.get("metricsDroppedCount") or 0) + 1
            )
            payload["metricsOriginalCount"] = int(
                payload.get("metricsOriginalCount") or len(metrics) + 1
            )
        encoded_size = _encoded_size(payload)
        if encoded_size > self.event_byte_limit:
            raise ValueError(
                "Training progress event exceeds the "
                f"{self.event_byte_limit} byte record limit."
            )

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        self.write_event(
            {
                "type": "epoch_started",
                "status": "running",
                "epoch": int(trainer.current_epoch),
                "step": int(trainer.global_step),
            }
        )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        global_step = int(trainer.global_step)
        if self.step_interval > 1 and global_step % self.step_interval != 0:
            return
        self.write_event(
            {
                "type": "step",
                "status": "running",
                "epoch": int(trainer.current_epoch),
                "step": global_step,
                "batch": int(batch_idx),
                "metrics": self._metrics(trainer),
            }
        )

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        self.write_event(
            {
                "type": "validation",
                "status": "running",
                "epoch": int(trainer.current_epoch),
                "step": int(trainer.global_step),
                "metrics": self._metrics(trainer),
            }
        )

    def on_fit_end(self, trainer, pl_module) -> None:
        self.write_event(
            {
                "type": "fit_completed",
                "status": "running",
                "epoch": int(trainer.current_epoch),
                "step": int(trainer.global_step),
                "metrics": self._metrics(trainer),
            }
        )

    def on_test_end(self, trainer, pl_module) -> None:
        self.write_event(
            {
                "type": "test_completed",
                "status": "running",
                "epoch": int(trainer.current_epoch),
                "step": int(trainer.global_step),
                "metrics": self._metrics(trainer),
            }
        )

    def on_exception(self, trainer, pl_module, exception: BaseException) -> None:
        formatted_traceback = "".join(
            traceback.format_exception(
                type(exception),
                exception,
                exception.__traceback__,
            )
        )
        self.write_event(
            {
                "type": "error",
                "status": "failed",
                "epoch": int(getattr(trainer, "current_epoch", 0)),
                "step": int(getattr(trainer, "global_step", 0)),
                "error": str(exception),
                "traceback": formatted_traceback,
            }
        )


__all__ = [
    "DEFAULT_PROGRESS_EVENT_BYTE_LIMIT",
    "DEFAULT_PROGRESS_METRIC_KEY_LIMIT",
    "DEFAULT_PROGRESS_STRING_VALUE_LIMIT",
    "JsonlTrainingProgressCallback",
    "NeuronClusterGrowthCallback",
    "sanitize_metric_payload",
]

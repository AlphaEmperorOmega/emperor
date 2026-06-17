from __future__ import annotations

import json
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lightning.pytorch.callbacks import Callback

DEFAULT_PROGRESS_METRIC_KEY_LIMIT = 512
DEFAULT_PROGRESS_EVENT_BYTE_LIMIT = 128_000
DEFAULT_PROGRESS_STRING_VALUE_LIMIT = 20_000
_DROPPED_METRIC_TOKENS = ("confusion_matrix", "per_class")


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
            return value.item()
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, str) and string_value_limit is not None:
            return _truncate_string(value, string_value_limit)
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
        self.option: str | None = None
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
        option: str | None = None,
        run_id: str | None = None,
        run_index: int | None = None,
        run_total: int | None = None,
        total_epochs: int | None = None,
    ) -> None:
        self.dataset = dataset
        self.log_dir = log_dir
        self.preset = preset
        self.option = option
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
            "option": self.option,
            "logDir": self.log_dir,
            "runId": self.run_id,
            "runIndex": self.run_index,
            "runTotal": self.run_total,
            "totalEpochs": self.total_epochs,
            **event_payload,
        }
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
        if not isinstance(metrics, dict):
            return

        while metrics and _encoded_size(payload) > self.event_byte_limit:
            key = next(reversed(metrics))
            del metrics[key]
            payload["metricsDroppedCount"] = (
                int(payload.get("metricsDroppedCount") or 0) + 1
            )
            payload["metricsOriginalCount"] = int(
                payload.get("metricsOriginalCount") or len(metrics) + 1
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

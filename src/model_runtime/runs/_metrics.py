from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from model_runtime.runs.json_values import require_finite_json

_DROPPED_METRIC_TOKENS = ("confusion_matrix", "per_class")


def truncate_string(value: str, limit: int) -> str:
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
            return truncate_string(value, string_value_limit)
        if isinstance(value, float) and not math.isfinite(value):
            require_finite_json(value)
        return value
    rendered = str(value)
    if string_value_limit is not None:
        return truncate_string(rendered, string_value_limit)
    return rendered


def portable_metric_values(metrics: Mapping[Any, Any]) -> dict[str, Any]:
    return {str(key): _json_value(value) for key, value in metrics.items()}


def _metric_key_is_dropped(key: str) -> bool:
    normalized = key.replace("\\", "/").lower()
    return any(token in normalized for token in _DROPPED_METRIC_TOKENS)


def sanitize_metric_payload(
    metrics: Mapping[Any, Any],
    *,
    metric_key_limit: int,
    string_value_limit: int,
) -> tuple[dict[str, Any], int, int]:
    sanitized: dict[str, Any] = {}
    dropped_count = 0
    safe_metric_key_limit = max(0, int(metric_key_limit))
    for raw_key, value in metrics.items():
        key = str(raw_key)
        if _metric_key_is_dropped(key) or len(sanitized) >= safe_metric_key_limit:
            dropped_count += 1
            continue
        sanitized[key] = _json_value(
            value,
            string_value_limit=string_value_limit,
        )
    return sanitized, len(metrics), dropped_count


__all__ = ["portable_metric_values", "sanitize_metric_payload", "truncate_string"]

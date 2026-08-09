from __future__ import annotations

import math
from collections.abc import Collection, Mapping
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
    protected_metric_keys: Collection[str] = (),
    deterministic_selection: bool = False,
) -> tuple[dict[str, Any], int, int]:
    eligible_metrics: dict[str, Any] = {}
    filtered_count = 0
    protected_keys = frozenset(str(key) for key in protected_metric_keys)
    for raw_key, value in metrics.items():
        key = str(raw_key)
        if _metric_key_is_dropped(key):
            filtered_count += 1
            continue
        eligible_metrics[key] = value

    safe_metric_key_limit = max(0, int(metric_key_limit))
    present_protected_keys = protected_keys.intersection(eligible_metrics)
    optional_keys = [
        key for key in eligible_metrics if key not in present_protected_keys
    ]
    optional_key_limit = safe_metric_key_limit
    optional_truncated_count = max(0, len(optional_keys) - optional_key_limit)
    if optional_truncated_count > 0 and deterministic_selection:
        retained_keys = sorted(
            present_protected_keys.union(
                sorted(optional_keys)[:optional_key_limit],
            )
        )
    elif optional_truncated_count > 0:
        retained_optional_keys = frozenset(optional_keys[:optional_key_limit])
        retained_keys = [
            key
            for key in eligible_metrics
            if key in present_protected_keys or key in retained_optional_keys
        ]
    else:
        retained_keys = list(eligible_metrics)

    sanitized = {
        key: _json_value(
            eligible_metrics[key],
            string_value_limit=string_value_limit,
        )
        for key in retained_keys
    }
    dropped_count = filtered_count + optional_truncated_count
    return sanitized, len(metrics), dropped_count


__all__ = ["portable_metric_values", "sanitize_metric_payload", "truncate_string"]

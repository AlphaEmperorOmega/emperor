"""Log Run artifact, result, and checkpoint helpers."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from workbench.backend.run_history.paths import resolved_under_root
from workbench.backend.tensorboard.events import event_file_fingerprint

CHECKPOINT_EPOCH_RE = re.compile(r"(?:^|[-_])epoch=(?P<value>\d+)(?:[-_]|$)")
CHECKPOINT_STEP_RE = re.compile(r"(?:^|[-_])step=(?P<value>\d+)(?:[-_]|$)")
HPARAM_INT_RE = re.compile(r"^[+-]?\d+$")
HPARAM_FLOAT_RE = re.compile(
    r"^[+-]?(?:(?:\d+\.\d*)|(?:\.\d+)|(?:\d+))(?:[eE][+-]?\d+)$"
    r"|^[+-]?(?:(?:\d+\.\d*)|(?:\.\d+))$"
)

EventFingerprint = tuple[tuple[str, int, int], ...]


def _read_result_payload(result_path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_result_object(result_path: Path, key: str) -> dict[str, Any]:
    value = _read_result_payload(result_path).get(key)
    return value if isinstance(value, dict) else {}


def _read_result_metrics(result_path: Path) -> dict[str, Any]:
    return _read_result_object(result_path, "metrics")


def _read_result_experiment_task(result_path: Path) -> str | None:
    value = _read_result_payload(result_path).get("experimentTask")
    return value if isinstance(value, str) and value else None


def _read_result_params(result_path: Path) -> dict[str, Any]:
    return _read_result_object(result_path, "params")


def _parse_hparam_value(raw_value: str) -> bool | int | float | str | None:
    value = raw_value.strip()
    if not value:
        return ""
    if " #" in value:
        value = value.split(" #", 1)[0].strip()
    normalized = value.lower()
    if normalized in {"null", "none", "~"}:
        return None
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    if (
        len(value) >= 2
        and value[0] == value[-1]
        and value[0] in {"'", '"'}
    ):
        return value[1:-1]
    if HPARAM_INT_RE.fullmatch(value):
        try:
            return int(value)
        except ValueError:
            return value
    if HPARAM_FLOAT_RE.fullmatch(value):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def _read_hparams_flat(hparams_path: Path) -> dict[str, Any]:
    try:
        lines = hparams_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return {}

    values: dict[str, Any] = {}
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        key = key.strip()
        value = raw_value.strip()
        if (
            not key
            or not value
            or value in {"|", ">"}
            or value.startswith(("[", "{", "- "))
        ):
            continue
        values[key] = _parse_hparam_value(value)
    return values


def _file_modified_at(path: Path) -> str:
    return (
        datetime.fromtimestamp(path.stat().st_mtime, UTC)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _relative_file_path(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _run_relative_file_label(run_dir: Path, path: Path) -> str:
    try:
        return path.relative_to(run_dir).as_posix()
    except ValueError:
        return path.name


def _safe_artifact_file(path: Path, root: Path) -> Path | None:
    if not path.is_file():
        return None
    resolved = resolved_under_root(path, root)
    if resolved is None or not resolved.is_file():
        return None
    return resolved


def _safe_artifact_files(run_dir: Path, root: Path, pattern: str) -> list[Path]:
    files: list[Path] = []
    for path in sorted(run_dir.rglob(pattern)):
        resolved = _safe_artifact_file(path, root)
        if resolved is not None:
            files.append(resolved)
    return files


def _event_file_fingerprint(run_dir: Path) -> EventFingerprint:
    return event_file_fingerprint(run_dir)


def _file_id(run_id: str, relative_path: str) -> str:
    return hashlib.sha256(f"{run_id}:{relative_path}".encode()).hexdigest()[:16]


def _parse_checkpoint_field(pattern: re.Pattern[str], filename: str) -> int | None:
    match = pattern.search(filename)
    if not match:
        return None
    try:
        return int(match.group("value"))
    except ValueError:
        return None


def _parse_checkpoint_epoch(filename: str) -> int | None:
    return _parse_checkpoint_field(CHECKPOINT_EPOCH_RE, Path(filename).stem)


def _parse_checkpoint_step(filename: str) -> int | None:
    return _parse_checkpoint_field(CHECKPOINT_STEP_RE, Path(filename).stem)


__all__ = [
    "EventFingerprint",
    "_event_file_fingerprint",
    "_file_id",
    "_file_modified_at",
    "_parse_checkpoint_epoch",
    "_parse_checkpoint_step",
    "_read_result_experiment_task",
    "_read_hparams_flat",
    "_read_result_metrics",
    "_read_result_params",
    "_relative_file_path",
    "_run_relative_file_label",
    "_safe_artifact_file",
    "_safe_artifact_files",
]

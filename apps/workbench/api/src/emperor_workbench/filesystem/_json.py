from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from model_runtime.runs import require_finite_json

from emperor_workbench.filesystem._paths import reject_link_like
from emperor_workbench.filesystem._permissions import apply_owner_only_permissions


def read_json_object(path: Path) -> dict[str, Any] | None:
    """Read a JSON object from ``path``.

    Missing files, read errors, invalid JSON, and non-object payloads return
    ``None`` so callers can preserve their own missing/corrupt file policy.
    """

    try:
        reject_link_like(Path(path), "JSON file")
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a JSON object to ``path`` using a same-directory replace."""

    require_finite_json(payload)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            delete=False,
            dir=target.parent,
            encoding="utf-8",
            prefix=f".{target.name}.",
            suffix=".tmp",
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(json.dumps(dict(payload), indent=2, sort_keys=True))
        apply_owner_only_permissions(temporary_path)
        temporary_path.replace(target)
        apply_owner_only_permissions(target)
    except Exception:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise

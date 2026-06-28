"""Small helpers for safe local filesystem access.

The Viewer backend is intentionally local-file based. These helpers keep common
path safety rules in one place without introducing a storage framework.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def resolve_root(root: Path) -> Path:
    """Return an absolute, resolved storage root."""

    return Path(root).resolve()


def resolve_under_root(root: Path, path: Path) -> Path:
    """Resolve ``path`` and require it to stay under ``root``.

    Relative paths are interpreted as children of ``root``. Absolute paths are
    checked directly against ``root``.
    """

    resolved_root = resolve_root(root)
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = resolved_root / candidate
    try:
        resolved_candidate = candidate.resolve()
        resolved_candidate.relative_to(resolved_root)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Path is outside allowed root: {path}") from exc
    return resolved_candidate


def safe_child_path(root: Path, relative_path: str | Path) -> Path:
    """Build a child path under ``root`` from user-controlled relative input."""

    path_text = str(relative_path)
    child = Path(path_text)
    if child.is_absolute():
        raise ValueError(f"Path must be relative: {relative_path}")
    if not path_text or any(part in {"", ".", ".."} for part in path_text.split("/")):
        raise ValueError(f"Path contains unsafe components: {relative_path}")
    if "\\" in path_text:
        raise ValueError(f"Path contains unsafe separators: {relative_path}")
    return resolve_under_root(root, child)


def require_safe_name(name: str, label: str = "name") -> str:
    """Validate a single filesystem name component."""

    path = Path(name)
    if not name:
        raise ValueError(f"{label} is required")
    if path.is_absolute() or len(path.parts) != 1:
        raise ValueError(f"{label} must be a single path component")
    if name in {".", ".."} or "\\" in name:
        raise ValueError(f"{label} contains unsafe path characters")
    return name


def reject_symlink(path: Path, label: str = "path") -> None:
    """Raise when ``path`` is a symlink."""

    if Path(path).is_symlink():
        raise ValueError(f"Refusing to use symlink {label}: {path}")


def read_json_object(path: Path) -> dict[str, Any] | None:
    """Read a JSON object from ``path``.

    Missing files, read errors, invalid JSON, and non-object payloads return
    ``None`` so callers can preserve their own missing/corrupt file policy.
    """

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a JSON object to ``path`` using a same-directory replace."""

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
        temporary_path.replace(target)
    except Exception:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise

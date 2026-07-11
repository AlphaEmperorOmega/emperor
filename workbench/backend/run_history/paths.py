"""Contained path primitive shared by Run History read and mutation policies."""

from __future__ import annotations

from pathlib import Path


def resolved_under_root(path: Path, root: Path) -> Path | None:
    """Resolve ``path`` under canonical ``root``, returning ``None`` on escape."""

    try:
        resolved = path.resolve()
        resolved.relative_to(root)
    except (OSError, ValueError):
        return None
    return resolved


__all__ = ["resolved_under_root"]

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

from workbench.backend.storage.local_files import (
    resolve_under_root,
    windows_regular_file_descriptor,
)


def resolved_under_root(path: Path, root: Path) -> Path | None:
    """Resolve ``path`` under canonical ``root``, returning ``None`` on escape."""

    try:
        resolved = path.resolve()
        resolved.relative_to(root)
    except (OSError, ValueError):
        return None
    return resolved


def read_regular_file_beneath(
    path: Path,
    *,
    boundary: Path,
    anchor: Path,
    max_bytes: int,
) -> bytes | None:
    """Descriptor-walk and read one regular file without following symlinks."""

    try:
        relative_boundary = boundary.relative_to(anchor)
        path.relative_to(boundary)
        relative_path = path.relative_to(anchor)
    except ValueError:
        return None
    if (
        not relative_path.parts
        or any(part in {"", ".", ".."} for part in relative_path.parts)
        or tuple(relative_path.parts[: len(relative_boundary.parts)])
        != relative_boundary.parts
    ):
        return None

    if sys.platform == "win32":
        try:
            resolve_under_root(anchor, boundary)
            with windows_regular_file_descriptor(
                path,
                trusted_root=boundary,
            ) as file_descriptor:
                metadata = os.fstat(file_descriptor)
                if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > max_bytes:
                    return None
                chunks: list[bytes] = []
                remaining = max_bytes + 1
                while remaining > 0:
                    chunk = os.read(file_descriptor, min(remaining, 64 * 1024))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    remaining -= len(chunk)
                payload = b"".join(chunks)
                return payload if len(payload) <= max_bytes else None
        except (OSError, ValueError):
            return None

    directory_flags = (
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    )
    file_flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
    opened: list[int] = []
    file_descriptor: int | None = None
    try:
        directory_fd = os.open(anchor, directory_flags)
        opened.append(directory_fd)
        for part in relative_path.parts[:-1]:
            directory_fd = os.open(part, directory_flags, dir_fd=directory_fd)
            opened.append(directory_fd)
        file_descriptor = os.open(
            relative_path.parts[-1],
            file_flags,
            dir_fd=directory_fd,
        )
        metadata = os.fstat(file_descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > max_bytes:
            return None
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining > 0:
            chunk = os.read(file_descriptor, min(remaining, 64 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        return payload if len(payload) <= max_bytes else None
    except OSError:
        return None
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
        for directory_fd in reversed(opened):
            os.close(directory_fd)


__all__ = ["read_regular_file_beneath", "resolved_under_root"]

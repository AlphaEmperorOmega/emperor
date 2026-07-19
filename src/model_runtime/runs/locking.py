from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from filelock import FileLock


@contextmanager
def exclusive_file_lock(path: str | Path, *, timeout: float = -1) -> Iterator[Path]:
    lock_path = Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(lock_path), timeout=timeout)
    with lock:
        yield lock_path


__all__ = ["exclusive_file_lock"]

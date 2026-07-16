from __future__ import annotations

from pathlib import Path
from typing import BinaryIO, Protocol

from emperor_workbench.filesystem import (
    apply_owner_only_permissions,
    reject_link_like,
)
from emperor_workbench.training_jobs._records import TrainingJobRecord

PRIVATE_DIRECTORY_MODE = 0o700
PRIVATE_FILE_MODE = 0o600


class TrainingJobStore(Protocol):
    def save(self, job: TrainingJobRecord) -> None: ...

    def get(self, job_id: str) -> TrainingJobRecord | None: ...

    def list(self) -> list[TrainingJobRecord]: ...


def ensure_private_directory(path: Path) -> Path:
    candidate = Path(path)
    reject_link_like(candidate, "private directory")
    candidate.mkdir(parents=True, exist_ok=True, mode=PRIVATE_DIRECTORY_MODE)
    reject_link_like(candidate, "private directory")
    if not candidate.is_dir():
        raise ValueError(f"Private directory is not canonical: {candidate}")
    apply_owner_only_permissions(candidate)
    return candidate


def open_private_binary(path: Path, flags: int) -> BinaryIO:
    import os

    if path.exists():
        reject_link_like(path, "private file")
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    close_on_exec = getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(
        path,
        flags | no_follow | close_on_exec,
        PRIVATE_FILE_MODE,
    )
    try:
        if hasattr(os, "fchmod"):
            os.fchmod(descriptor, PRIVATE_FILE_MODE)
        return os.fdopen(descriptor, "ab" if flags & os.O_APPEND else "wb")
    except Exception:
        os.close(descriptor)
        raise


def ensure_private_file(path: Path) -> Path:
    import os

    ensure_private_directory(path.parent)
    with open_private_binary(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND):
        pass
    apply_owner_only_permissions(path)
    return path


__all__ = [
    "PRIVATE_DIRECTORY_MODE",
    "PRIVATE_FILE_MODE",
    "TrainingJobStore",
    "ensure_private_directory",
    "ensure_private_file",
    "open_private_binary",
]

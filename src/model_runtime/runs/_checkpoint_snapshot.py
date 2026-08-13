from __future__ import annotations

import hashlib
import os
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from model_runtime.runs.errors import InvalidCheckpointContinuation

_COPY_CHUNK_BYTES = 1024 * 1024


@dataclass(slots=True)
class CheckpointSnapshot:
    """Private immutable bytes admitted from one opened source descriptor."""

    source_path: Path
    path: Path
    sha256: str
    size_bytes: int
    _temporary_root: Path
    _closed: bool = False

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def open_verified(self, maximum_bytes: int, display_path: str) -> BinaryIO:
        snapshot_file, metadata = _opened_regular_file(self.path, display_path)
        try:
            digest, size_bytes = _snapshot_digest(snapshot_file, maximum_bytes)
            if (
                int(metadata.st_size) != self.size_bytes
                or digest != self.sha256
                or size_bytes != self.size_bytes
            ):
                raise InvalidCheckpointContinuation(
                    f"Checkpoint '{display_path}' private snapshot changed after "
                    "admission."
                )
            return snapshot_file
        except BaseException:
            snapshot_file.close()
            raise

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.path.chmod(0o600)
        except OSError:
            pass
        try:
            shutil.rmtree(self._temporary_root)
        except OSError as exc:
            raise InvalidCheckpointContinuation(
                "Checkpoint private snapshot could not be removed."
            ) from exc
        self._closed = True


def admit_checkpoint_snapshot(
    source_path: Path,
    maximum_bytes: int,
    display_path: str,
) -> CheckpointSnapshot:
    source_file, initial_stat = _opened_regular_file(source_path, display_path)
    with source_file:
        _validate_initial_size(
            display_path,
            int(initial_stat.st_size),
            maximum_bytes,
        )
        return _copy_private_snapshot(
            source_file,
            initial_stat,
            source_path,
            display_path,
            maximum_bytes,
        )


def _opened_regular_file(
    path: Path,
    display_path: str,
) -> tuple[BinaryIO, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except (OSError, ValueError) as exc:
        raise InvalidCheckpointContinuation(
            f"Checkpoint '{display_path}' must be a readable regular file."
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise InvalidCheckpointContinuation(
                f"Checkpoint '{display_path}' must be a readable regular file."
            )
        return os.fdopen(descriptor, "rb", closefd=True), metadata
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def _validate_initial_size(
    display_path: str,
    initial_size: int,
    maximum_bytes: int,
) -> None:
    if initial_size > maximum_bytes:
        raise InvalidCheckpointContinuation(
            f"Checkpoint '{display_path}' is {initial_size} bytes, exceeding the "
            f"checkpoint continuation limit of {maximum_bytes} bytes."
        )


def _temporary_snapshot(display_path: str) -> tuple[Path, Path]:
    temporary_root: Path | None = None
    try:
        temporary_root = Path(tempfile.mkdtemp(prefix="emperor-checkpoint-"))
        temporary_root.chmod(0o700)
        return temporary_root, temporary_root / "admitted.ckpt"
    except BaseException as exc:
        if temporary_root is not None:
            _cleanup_partial_snapshot(temporary_root, exc)
        if not isinstance(exc, Exception):
            raise
        raise InvalidCheckpointContinuation(
            f"Checkpoint '{display_path}' could not create a private snapshot."
        ) from exc


def _copy_checkpoint_bytes(
    source_file: BinaryIO,
    snapshot_path: Path,
    display_path: str,
    maximum_bytes: int,
) -> tuple[str, int]:
    digest = hashlib.sha256()
    copied_bytes = 0
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(snapshot_path, flags, 0o600)
    with os.fdopen(descriptor, "wb", closefd=True) as snapshot_file:
        snapshot_path.chmod(0o600)
        while chunk := source_file.read(_COPY_CHUNK_BYTES):
            copied_bytes += len(chunk)
            if copied_bytes > maximum_bytes:
                raise InvalidCheckpointContinuation(
                    f"Checkpoint '{display_path}' exceeds the checkpoint "
                    f"continuation limit of {maximum_bytes} bytes while being read."
                )
            digest.update(chunk)
            snapshot_file.write(chunk)
        snapshot_file.flush()
        os.fsync(snapshot_file.fileno())
    return digest.hexdigest(), copied_bytes


def _validate_stable_source(
    initial_stat: os.stat_result,
    final_stat: os.stat_result,
    copied_bytes: int,
    display_path: str,
) -> None:
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    changed = any(
        getattr(initial_stat, name) != getattr(final_stat, name)
        for name in stable_fields
    )
    if changed or copied_bytes != int(final_stat.st_size):
        raise InvalidCheckpointContinuation(
            f"Checkpoint '{display_path}' changed while it was being admitted."
        )


def _copy_private_snapshot(
    source_file: BinaryIO,
    initial_stat: os.stat_result,
    source_path: Path,
    display_path: str,
    maximum_bytes: int,
) -> CheckpointSnapshot:
    temporary_root, snapshot_path = _temporary_snapshot(display_path)
    try:
        digest, copied_bytes = _copy_checkpoint_bytes(
            source_file,
            snapshot_path,
            display_path,
            maximum_bytes,
        )
        _validate_stable_source(
            initial_stat,
            os.fstat(source_file.fileno()),
            copied_bytes,
            display_path,
        )
        snapshot_path.chmod(0o400)
        return CheckpointSnapshot(
            source_path=source_path,
            path=snapshot_path,
            sha256=digest,
            size_bytes=copied_bytes,
            _temporary_root=temporary_root,
        )
    except BaseException as exception:
        _cleanup_partial_snapshot(temporary_root, exception)
        if not isinstance(exception, Exception):
            raise
        if isinstance(exception, InvalidCheckpointContinuation):
            raise
        raise InvalidCheckpointContinuation(
            f"Checkpoint '{display_path}' could not be copied into a private "
            "snapshot."
        ) from exception


def _cleanup_partial_snapshot(
    temporary_root: Path,
    primary: BaseException,
) -> None:
    try:
        shutil.rmtree(temporary_root)
    except OSError as cleanup_error:
        primary.add_note(
            "Partial checkpoint snapshot cleanup also failed: "
            f"{type(cleanup_error).__name__}."
        )


def _snapshot_digest(
    snapshot_file: BinaryIO,
    maximum_bytes: int,
) -> tuple[str, int]:
    digest = hashlib.sha256()
    size_bytes = 0
    snapshot_file.seek(0)
    while chunk := snapshot_file.read(_COPY_CHUNK_BYTES):
        size_bytes += len(chunk)
        if size_bytes > maximum_bytes:
            raise InvalidCheckpointContinuation(
                "Checkpoint private snapshot exceeds its admitted byte limit."
            )
        digest.update(chunk)
    snapshot_file.seek(0)
    return digest.hexdigest(), size_bytes


__all__: list[str] = []

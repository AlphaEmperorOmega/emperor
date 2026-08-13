from __future__ import annotations

import zipfile
from pathlib import Path
from typing import BinaryIO, NoReturn

from model_runtime.runs._checkpoint_receipt import CheckpointArchiveReceipt
from model_runtime.runs.checkpoint_admission import CheckpointAdmissionPolicy
from model_runtime.runs.errors import InvalidCheckpointContinuation


def _raise_limit(path: Path, detail: str) -> NoReturn:
    raise InvalidCheckpointContinuation(f"Checkpoint '{path}' {detail}")


def checkpoint_archive_receipt(
    source: BinaryIO,
    display_path: Path,
    policy: CheckpointAdmissionPolicy,
) -> CheckpointArchiveReceipt:
    """Bound a Torch archive's declared expansion before deserialization."""

    try:
        source.seek(0)
        if not zipfile.is_zipfile(source):
            return CheckpointArchiveReceipt(0, 0)
        source.seek(0)
        with zipfile.ZipFile(source) as archive:
            return _archive_members_receipt(archive, display_path, policy)
    except InvalidCheckpointContinuation:
        raise
    except (OSError, zipfile.BadZipFile) as exc:
        raise InvalidCheckpointContinuation(
            f"Checkpoint '{display_path}' has an invalid archive container."
        ) from exc
    finally:
        source.seek(0)


def _archive_members_receipt(
    archive: zipfile.ZipFile,
    display_path: Path,
    policy: CheckpointAdmissionPolicy,
) -> CheckpointArchiveReceipt:
    record_count = 0
    uncompressed_bytes = 0
    for record in archive.infolist():
        record_count += 1
        if record_count > policy.max_archive_records:
            _raise_limit(
                display_path,
                f"archive contains more than {policy.max_archive_records} records.",
            )
        record_bytes = int(record.file_size)
        if record_bytes > policy.max_archive_record_bytes:
            _raise_limit(
                display_path,
                f"archive record expands to {record_bytes} bytes, exceeding the "
                f"limit of {policy.max_archive_record_bytes} bytes.",
            )
        uncompressed_bytes += record_bytes
        if uncompressed_bytes > policy.max_archive_uncompressed_bytes:
            _raise_limit(
                display_path,
                f"archive expands to {uncompressed_bytes} bytes, exceeding the "
                f"limit of {policy.max_archive_uncompressed_bytes} bytes.",
            )
    return CheckpointArchiveReceipt(record_count, uncompressed_bytes)


__all__: list[str] = []

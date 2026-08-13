from __future__ import annotations

import hashlib
import os
import stat
import sys
from collections.abc import Mapping
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, Protocol, cast

if TYPE_CHECKING:
    from model_runtime.runs._checkpoint_receipt import CheckpointArchiveReceipt
    from model_runtime.runs._checkpoint_worker_protocol import CheckpointWorkerRequest


class _ResourceModule(Protocol):
    RLIM_INFINITY: int
    RLIMIT_AS: int
    RLIMIT_CORE: int
    RLIMIT_CPU: int
    RLIMIT_FSIZE: int

    def getrlimit(self, resource_id: int) -> tuple[int, int]: ...

    def setrlimit(self, resource_id: int, limits: tuple[int, int]) -> None: ...


def _resource_module() -> _ResourceModule | None:
    try:
        module = import_module("resource")
    except ModuleNotFoundError:  # pragma: no cover - unsupported platform
        return None
    return cast(_ResourceModule, cast(object, module))


_RESOURCE = _resource_module()


def _effective_limit(resource_id: int, requested: int) -> int:
    if _RESOURCE is None:  # pragma: no cover - guarded by caller
        raise RuntimeError("hard checkpoint decoder controls are unavailable")
    current_soft, current_hard = _RESOURCE.getrlimit(resource_id)
    candidates = [requested]
    for current in (current_soft, current_hard):
        if current != _RESOURCE.RLIM_INFINITY:
            candidates.append(max(0, int(current)))
    effective = min(candidates)
    _RESOURCE.setrlimit(resource_id, (effective, effective))
    return effective


def _apply_hard_limits(
    memory_bytes: int,
    cpu_seconds: int,
    receipt_bytes: int,
) -> Mapping[str, int]:
    if _RESOURCE is None or not sys.platform.startswith("linux"):
        raise RuntimeError("hard checkpoint decoder controls are unavailable")
    return {
        "coreBytes": _effective_limit(_RESOURCE.RLIMIT_CORE, 0),
        "fileSizeBytes": _effective_limit(_RESOURCE.RLIMIT_FSIZE, receipt_bytes),
        "cpuSeconds": _effective_limit(_RESOURCE.RLIMIT_CPU, cpu_seconds),
        "addressSpaceBytes": _effective_limit(_RESOURCE.RLIMIT_AS, memory_bytes),
    }


def _opened_snapshot(path: str):
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("Checkpoint snapshot is not a regular file.")
        return os.fdopen(descriptor, "rb", closefd=True), metadata
    except BaseException:
        os.close(descriptor)
        raise


def _write_response(path: str, response: bytes) -> None:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as output:
            output.write(response)
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def _read_request() -> CheckpointWorkerRequest | None:
    from model_runtime.runs._checkpoint_worker_protocol import (
        MAX_CHECKPOINT_WORKER_REQUEST_BYTES,
        decode_worker_request,
    )

    request = sys.stdin.buffer.read(MAX_CHECKPOINT_WORKER_REQUEST_BYTES + 1)
    if len(request) > MAX_CHECKPOINT_WORKER_REQUEST_BYTES:
        return None
    try:
        values = decode_worker_request(request)
    except Exception:
        return None
    return values


def _snapshot_digest(snapshot_file: BinaryIO) -> str:
    digest = hashlib.sha256()
    while chunk := snapshot_file.read(1024 * 1024):
        digest.update(chunk)
    snapshot_file.seek(0)
    return digest.hexdigest()


def _load_snapshot(
    request: CheckpointWorkerRequest,
) -> tuple[object, str, CheckpointArchiveReceipt]:
    import torch

    from model_runtime.runs._checkpoint_archive import checkpoint_archive_receipt

    snapshot_file, metadata = _opened_snapshot(str(request.snapshot_path))
    with snapshot_file:
        if int(metadata.st_size) != request.size_bytes:
            raise ValueError("Checkpoint snapshot size changed before decode.")
        actual_sha = _snapshot_digest(snapshot_file)
        if actual_sha != request.sha256:
            raise ValueError("Checkpoint snapshot digest changed before decode.")
        archive = checkpoint_archive_receipt(
            snapshot_file,
            Path(request.source_name),
            request.policy,
        )
        snapshot_file.seek(0)
        payload = torch.load(snapshot_file, map_location="cpu", weights_only=True)
    return payload, actual_sha, archive


def _decode_response(
    request: CheckpointWorkerRequest,
    controls: Mapping[str, int],
) -> bytes:
    from model_runtime.runs._checkpoint_payload import validate_checkpoint_payload
    from model_runtime.runs._checkpoint_worker_protocol import (
        encode_worker_error,
        encode_worker_receipt,
    )
    from model_runtime.runs.errors import InvalidCheckpointContinuation

    try:
        payload, actual_sha, archive = _load_snapshot(request)
        validated = validate_checkpoint_payload(
            payload,
            Path(request.source_name),
            request.policy,
            archive,
        )
        return encode_worker_receipt(
            actual_sha,
            request.size_bytes,
            validated.receipt,
            controls,
        )
    except InvalidCheckpointContinuation as exc:
        return encode_worker_error("invalid-payload", str(exc))
    except Exception:
        return encode_worker_error(
            "load-failed",
            "Checkpoint could not be loaded as a Lightning checkpoint.",
        )


def _control_failure_response() -> bytes:
    from model_runtime.runs._checkpoint_worker_protocol import encode_worker_error

    return encode_worker_error(
        "controls-unavailable",
        "Checkpoint isolated decoder could not install every hard control.",
    )


def main() -> int:
    request = _read_request()
    if request is None:
        return 70
    try:
        from model_runtime.runs._checkpoint_worker_protocol import (
            MAX_CHECKPOINT_WORKER_RECEIPT_BYTES,
        )

        controls = _apply_hard_limits(
            request.policy.worker_memory_bytes,
            request.policy.worker_cpu_seconds,
            MAX_CHECKPOINT_WORKER_RECEIPT_BYTES,
        )
    except Exception:
        response = _control_failure_response()
    else:
        response = _decode_response(request, controls)
    try:
        _write_response(str(request.receipt_path), response)
    except Exception:
        return 70
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]

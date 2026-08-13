from __future__ import annotations

import os
import stat
import subprocess
import sys
import unicodedata
from collections.abc import Mapping
from pathlib import Path

from model_runtime.runs._checkpoint_receipt import CheckpointPayloadReceipt
from model_runtime.runs._checkpoint_worker_protocol import (
    MAX_CHECKPOINT_WORKER_RECEIPT_BYTES,
    CheckpointWorkerRejected,
    CheckpointWorkerRequest,
    decode_worker_response,
    encode_worker_request,
)
from model_runtime.runs.checkpoint_admission import CheckpointAdmissionPolicy
from model_runtime.runs.errors import InvalidCheckpointContinuation

_WORKER_COMMAND = (
    sys.executable,
    "-P",
    "-m",
    "model_runtime.runs._checkpoint_worker",
)


def isolated_decode_available() -> bool:
    return os.name == "posix" and sys.platform.startswith("linux")


def _display_path(path: Path) -> str:
    return "".join(
        "?" if unicodedata.category(character).startswith("C") else character
        for character in str(path)
    )


def _semantic_worker_message(detail: str, source_path: Path) -> str:
    source_name = _display_path(Path(source_path.name))
    requested_path = _display_path(source_path)
    prefix = f"Checkpoint '{source_name}'"
    if detail.startswith(prefix):
        return f"Checkpoint '{requested_path}'{detail[len(prefix):]}"
    return f"Checkpoint '{requested_path}' failed checkpoint continuation validation."


def _read_bounded_receipt(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size > MAX_CHECKPOINT_WORKER_RECEIPT_BYTES
        ):
            raise ValueError("Invalid checkpoint decoder receipt.")
        with os.fdopen(descriptor, "rb", closefd=True) as receipt_file:
            raw = receipt_file.read(MAX_CHECKPOINT_WORKER_RECEIPT_BYTES + 1)
        if len(raw) > MAX_CHECKPOINT_WORKER_RECEIPT_BYTES:
            raise ValueError("Checkpoint decoder receipt exceeds its size limit.")
        return raw
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def decode_checkpoint_isolated(
    snapshot_path: Path,
    source_path: Path,
    sha256: str,
    size_bytes: int,
    policy: CheckpointAdmissionPolicy,
) -> CheckpointPayloadReceipt:
    if not isolated_decode_available():
        raise InvalidCheckpointContinuation(
            "Checkpoint continuation requires isolated decode, but hard decoder "
            "controls are unavailable on this platform."
        )
    receipt_path = snapshot_path.parent / "decoder-receipt.json"
    worker_request = CheckpointWorkerRequest(
        snapshot_path=snapshot_path,
        receipt_path=receipt_path,
        source_name=source_path.name,
        sha256=sha256,
        size_bytes=size_bytes,
        policy=policy,
    )
    request = _worker_request(worker_request, source_path)
    process = _start_worker()
    _communicate_with_worker(process, request, policy)
    return _validated_worker_receipt(
        receipt_path,
        source_path,
        sha256,
        size_bytes,
        policy,
    )


def _worker_request(
    request: CheckpointWorkerRequest,
    source_path: Path,
) -> bytes:
    try:
        return encode_worker_request(request)
    except (TypeError, ValueError) as exc:
        raise InvalidCheckpointContinuation(
            f"Checkpoint '{_display_path(source_path)}' could not start its "
            "isolated decoder."
        ) from exc


def _start_worker() -> subprocess.Popen[bytes]:
    try:
        return subprocess.Popen(  # noqa: S603 - fixed interpreter/module command
            _WORKER_COMMAND,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
        )
    except OSError as exc:
        raise InvalidCheckpointContinuation(
            "Checkpoint isolated decoder could not be started."
        ) from exc


def _communicate_with_worker(
    process: subprocess.Popen[bytes],
    request: bytes,
    policy: CheckpointAdmissionPolicy,
) -> None:
    try:
        try:
            process.communicate(
                input=request,
                timeout=policy.worker_wall_timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            try:
                _terminate_and_reap(process)
            except Exception as cleanup_error:
                exc.add_note(
                    "Checkpoint decoder termination also failed: "
                    f"{type(cleanup_error).__name__}."
                )
            raise InvalidCheckpointContinuation(
                "Checkpoint isolated decoder timed out after "
                f"{policy.worker_wall_timeout_seconds:g} seconds."
            ) from exc
        except BaseException as exc:
            try:
                _terminate_and_reap(process)
            except Exception as cleanup_error:
                exc.add_note(
                    "Checkpoint decoder cleanup also failed: "
                    f"{type(cleanup_error).__name__}."
                )
            raise
    finally:
        _close_process_streams(process)
    if process.returncode != 0:
        raise InvalidCheckpointContinuation(
            "Checkpoint isolated decoder terminated abnormally."
        )


def _validated_worker_receipt(
    receipt_path: Path,
    source_path: Path,
    sha256: str,
    size_bytes: int,
    policy: CheckpointAdmissionPolicy,
) -> CheckpointPayloadReceipt:
    try:
        raw_response = _read_bounded_receipt(receipt_path)
        response = decode_worker_response(raw_response)
        if response.sha256 != sha256 or response.size_bytes != size_bytes:
            raise ValueError("Checkpoint decoder receipt did not bind admitted bytes.")
        _validate_controls(response.controls, policy)
    except CheckpointWorkerRejected as exc:
        if exc.category == "invalid-payload":
            message = _semantic_worker_message(exc.detail, source_path)
        elif exc.category == "load-failed":
            message = (
                f"Checkpoint '{_display_path(source_path)}' could not be loaded "
                "as a Lightning checkpoint."
            )
        elif exc.category == "controls-unavailable":
            message = (
                "Checkpoint isolated decoder could not install every hard control."
            )
        else:
            message = "Checkpoint isolated decoder rejected its request."
        raise InvalidCheckpointContinuation(message) from exc
    except (OSError, TypeError, ValueError) as exc:
        raise InvalidCheckpointContinuation(
            f"Checkpoint '{_display_path(source_path)}' isolated decoder returned "
            "an invalid receipt."
        ) from exc
    finally:
        try:
            receipt_path.unlink(missing_ok=True)
        except OSError:
            pass
    return response.receipt


def _validate_controls(
    controls: Mapping[str, int],
    policy: CheckpointAdmissionPolicy,
) -> None:
    expected_maximums = {
        "addressSpaceBytes": policy.worker_memory_bytes,
        "coreBytes": 0,
        "cpuSeconds": policy.worker_cpu_seconds,
        "fileSizeBytes": MAX_CHECKPOINT_WORKER_RECEIPT_BYTES,
    }
    if set(controls) != set(expected_maximums) or any(
        value < 0 or value > expected_maximums[name]
        for name, value in controls.items()
    ):
        raise ValueError("Checkpoint decoder did not attest every hard control.")


def _terminate_and_reap(process: subprocess.Popen[bytes]) -> None:
    try:
        process.kill()
    except OSError:
        pass
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except OSError:
            pass
        process.wait(timeout=5.0)


def _close_process_streams(process: subprocess.Popen[bytes]) -> None:
    if process.stdin is None:
        return
    try:
        process.stdin.close()
    except OSError:
        pass


__all__: list[str] = []

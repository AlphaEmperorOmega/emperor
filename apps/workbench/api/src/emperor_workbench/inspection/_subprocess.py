from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from queue import Empty, Queue
from typing import TYPE_CHECKING

from emperor_workbench.failures import FailureKind
from emperor_workbench.inspection._errors import (
    InspectionFailure,
    inspection_failure,
)
from emperor_workbench.inspection._worker_protocol import (
    MAX_WORKER_RESULT_BYTES,
    decode_worker_response,
    encode_worker_request,
)
from emperor_workbench.model_packages import ModelPackageFailure

if TYPE_CHECKING:
    from model_runtime.inspection import InspectionRequest, InspectionResult

    from emperor_workbench.model_packages import SelectedModelPackage

_DEFAULT_WORKER_COMMAND = (
    sys.executable,
    "-P",
    "-m",
    "emperor_workbench.inspection.worker",
)


@dataclass(frozen=True, slots=True)
class InspectionWorkerLimits:
    memory_bytes: int = 4 * 1024**3
    cpu_count: int = 4
    timeout_seconds: float = 60.0

    def __post_init__(self) -> None:
        if self.memory_bytes < 1:
            raise ValueError("Inspection memory limit must be positive.")
        if self.cpu_count < 1:
            raise ValueError("Inspection CPU limit must be positive.")
        if self.timeout_seconds <= 0:
            raise ValueError("Inspection timeout must be positive.")


class SubprocessInspectionExecutor:
    """Execute one semantic Inspection request in a fresh contained process."""

    def __init__(
        self,
        limits: InspectionWorkerLimits | None = None,
        *,
        command: Sequence[str] | None = None,
    ) -> None:
        self._limits = limits or InspectionWorkerLimits()
        self._command = tuple(command or _DEFAULT_WORKER_COMMAND)

    def inspect(
        self,
        selected: SelectedModelPackage,
        request: InspectionRequest,
    ) -> InspectionResult:
        try:
            encoded_request = encode_worker_request(
                selected,
                request,
                self._limits,
            )
        except ModelPackageFailure as exc:
            raise inspection_failure(exc) from exc

        stdout, return_code = self._execute_worker(encoded_request)
        if return_code != 0:
            raise InspectionFailure(
                "Inspection worker crashed.",
                kind=FailureKind.UNAVAILABLE,
            )
        return decode_worker_response(stdout)

    def _execute_worker(self, encoded_request: bytes) -> tuple[bytes, int]:
        environment = dict(os.environ)
        if os.name == "nt":
            from emperor_workbench.inspection._windows_job import (
                WindowsInspectionOutputTooLarge,
                execute_windows_inspection_worker,
            )

            try:
                return execute_windows_inspection_worker(
                    self._command,
                    encoded_request=encoded_request,
                    environment=environment,
                    memory_bytes=self._limits.memory_bytes,
                    cpu_count=self._limits.cpu_count,
                    timeout_seconds=self._limits.timeout_seconds,
                    max_output_bytes=MAX_WORKER_RESULT_BYTES,
                )
            except subprocess.TimeoutExpired as exc:
                raise self._timeout_failure() from exc
            except WindowsInspectionOutputTooLarge as exc:
                raise self._oversize_failure() from exc

        started_at = time.monotonic()
        try:
            process = subprocess.Popen(  # noqa: S603 - fixed or injected command
                self._command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env=environment,
                start_new_session=True,
            )
        except OSError as exc:
            raise InspectionFailure(
                "Inspection worker crashed.",
                kind=FailureKind.UNAVAILABLE,
            ) from exc
        if process.stdin is None or process.stdout is None:
            _terminate_and_reap(process)
            raise InspectionFailure(
                "Inspection worker crashed.",
                kind=FailureKind.UNAVAILABLE,
            )

        result_queue: Queue[tuple[bytes | None, BaseException | None]] = Queue(
            maxsize=1
        )

        def exchange() -> None:
            try:
                process.stdin.write(encoded_request)
                process.stdin.close()
                result_queue.put(
                    (
                        process.stdout.read(MAX_WORKER_RESULT_BYTES + 1),
                        None,
                    )
                )
            except BaseException as exc:  # pragma: no cover - surfaced below
                result_queue.put((None, exc))

        exchange_thread = threading.Thread(
            target=exchange,
            name="inspection-worker-exchange",
            daemon=True,
        )
        exchange_thread.start()
        try:
            stdout, exchange_error = result_queue.get(
                timeout=_remaining_timeout(
                    started_at,
                    self._limits.timeout_seconds,
                )
            )
        except (Empty, subprocess.TimeoutExpired) as exc:
            _terminate_and_reap(process)
            exchange_thread.join(timeout=1.0)
            raise self._timeout_failure() from exc

        if exchange_error is not None or stdout is None:
            _terminate_and_reap(process)
            exchange_thread.join(timeout=1.0)
            raise InspectionFailure(
                "Inspection worker crashed.",
                kind=FailureKind.UNAVAILABLE,
            ) from exchange_error
        if len(stdout) > MAX_WORKER_RESULT_BYTES:
            _terminate_and_reap(process)
            exchange_thread.join(timeout=1.0)
            raise self._oversize_failure()

        try:
            process.wait(
                timeout=_remaining_timeout(
                    started_at,
                    self._limits.timeout_seconds,
                )
            )
        except subprocess.TimeoutExpired as exc:
            _terminate_and_reap(process)
            exchange_thread.join(timeout=1.0)
            raise self._timeout_failure() from exc
        finally:
            _close_process_streams(process)
        exchange_thread.join(timeout=1.0)
        return stdout, int(process.returncode or 0)

    def _timeout_failure(self) -> InspectionFailure:
        return InspectionFailure(
            "Inspection construction exceeded the "
            f"{self._limits.timeout_seconds:g} second limit.",
            kind=FailureKind.TIMEOUT,
        )

    @staticmethod
    def _oversize_failure() -> InspectionFailure:
        return InspectionFailure(
            "Inspection worker result exceeded its size limit.",
            kind=FailureKind.UNAVAILABLE,
        )


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    if os.name != "posix":
        try:
            process.kill()
        except OSError:
            pass
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return


def _terminate_and_reap(process: subprocess.Popen[bytes]) -> None:
    _terminate_process_group(process)
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except OSError:
            pass
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            pass
    finally:
        _close_process_streams(process)


def _close_process_streams(process: subprocess.Popen[bytes]) -> None:
    for stream in (process.stdin, process.stdout):
        if stream is None:
            continue
        try:
            stream.close()
        except OSError:
            pass


def _remaining_timeout(started_at: float, timeout_seconds: float) -> float:
    remaining = timeout_seconds - (time.monotonic() - started_at)
    if remaining <= 0:
        raise subprocess.TimeoutExpired(
            "emperor_workbench.inspection.worker",
            timeout_seconds,
        )
    return remaining


__all__ = ["InspectionWorkerLimits", "SubprocessInspectionExecutor"]

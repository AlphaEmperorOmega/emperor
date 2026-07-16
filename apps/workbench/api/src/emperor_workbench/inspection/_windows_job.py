from __future__ import annotations

import os
import subprocess
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, BinaryIO


class WindowsInspectionOutputTooLarge(Exception):
    """The contained worker exceeded its bounded stdout protocol."""


def _modules() -> tuple[Any, ...]:
    if os.name != "nt":
        raise OSError("Windows Job Objects are only available on Windows.")
    try:
        import msvcrt

        import pywintypes
        import win32api
        import win32con
        import win32event
        import win32job
        import win32process
    except ImportError as exc:  # pragma: no cover - Windows dependency contract
        raise OSError("Windows Job Objects require pywin32.") from exc
    return (
        msvcrt,
        pywintypes,
        win32api,
        win32con,
        win32event,
        win32job,
        win32process,
    )


def execute_windows_inspection_worker(
    command: Sequence[str],
    *,
    encoded_request: bytes,
    environment: dict[str, str],
    memory_bytes: int,
    cpu_count: int,
    timeout_seconds: float,
    max_output_bytes: int,
) -> tuple[bytes, int]:
    with tempfile.TemporaryDirectory(prefix="emperor-inspection-") as directory:
        root = Path(directory)
        request_path = root / "request.json"
        output_path = root / "result.json"
        error_path = root / "error.log"
        request_path.write_bytes(encoded_request)
        job = _create_job(
            memory_bytes=memory_bytes,
            cpu_count=cpu_count,
            process_count=max(8, cpu_count * 4),
        )
        process_handle = None
        try:
            with (
                request_path.open("rb") as request_file,
                output_path.open("wb") as output_file,
                error_path.open("wb") as error_file,
            ):
                process_handle = _start_suspended(
                    job,
                    list(command),
                    cwd=Path.cwd(),
                    env=environment,
                    stdin=request_file,
                    stdout=output_file,
                    stderr=error_file,
                )
                return_code = _wait_for_process_and_job(
                    process_handle,
                    job,
                    command=tuple(command),
                    timeout_seconds=timeout_seconds,
                    output_path=output_path,
                    max_output_bytes=max_output_bytes,
                )
            with output_path.open("rb") as output_file:
                stdout = output_file.read(max_output_bytes + 1)
            if len(stdout) > max_output_bytes:
                raise WindowsInspectionOutputTooLarge
            return stdout, return_code
        except (subprocess.TimeoutExpired, WindowsInspectionOutputTooLarge):
            _terminate_job(job)
            raise
        finally:
            if process_handle is not None:
                process_handle.Close()
            job.Close()


def _create_job(
    *,
    memory_bytes: int,
    cpu_count: int,
    process_count: int,
):
    (
        _msvcrt,
        pywintypes,
        _win32api,
        _win32con,
        _win32event,
        win32job,
        _win32process,
    ) = _modules()
    security = pywintypes.SECURITY_ATTRIBUTES()
    security.bInheritHandle = False
    handle = win32job.CreateJobObject(security, None)
    try:
        information = win32job.QueryInformationJobObject(
            handle,
            win32job.JobObjectExtendedLimitInformation,
        )
        basic = information["BasicLimitInformation"]
        basic["LimitFlags"] |= (
            getattr(win32job, "JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE", 0x2000)
            | getattr(win32job, "JOB_OBJECT_LIMIT_ACTIVE_PROCESS", 0x00000008)
            | getattr(win32job, "JOB_OBJECT_LIMIT_JOB_MEMORY", 0x00000200)
        )
        basic["ActiveProcessLimit"] = process_count
        information["JobMemoryLimit"] = memory_bytes
        win32job.SetInformationJobObject(
            handle,
            win32job.JobObjectExtendedLimitInformation,
            information,
        )
        available = os.cpu_count() or 1
        cpu_rate = min(10_000, max(1, int(10_000 * cpu_count / available)))
        if cpu_rate < 10_000:
            win32job.SetInformationJobObject(
                handle,
                win32job.JobObjectCpuRateControlInformation,
                {
                    "ControlFlags": getattr(
                        win32job,
                        "JOB_OBJECT_CPU_RATE_CONTROL_ENABLE",
                        0x1,
                    )
                    | getattr(
                        win32job,
                        "JOB_OBJECT_CPU_RATE_CONTROL_HARD_CAP",
                        0x4,
                    ),
                    "CpuRate": cpu_rate,
                },
            )
    except Exception:
        handle.Close()
        raise
    return handle


def _start_suspended(
    job,
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    stdin: BinaryIO,
    stdout: BinaryIO,
    stderr: BinaryIO,
):
    (
        msvcrt,
        _pywintypes,
        win32api,
        win32con,
        _win32event,
        win32job,
        win32process,
    ) = _modules()
    handles = [
        msvcrt.get_osfhandle(stdin.fileno()),
        msvcrt.get_osfhandle(stdout.fileno()),
        msvcrt.get_osfhandle(stderr.fileno()),
    ]
    previous_inheritance: list[bool] = []
    for handle in handles:
        previous_inheritance.append(os.get_handle_inheritable(handle))
        os.set_handle_inheritable(handle, True)
    startup = win32process.STARTUPINFO()
    startup.dwFlags |= win32con.STARTF_USESTDHANDLES
    startup.hStdInput, startup.hStdOutput, startup.hStdError = handles
    process_handle = None
    thread_handle = None
    try:
        process_handle, thread_handle, _pid, _thread_id = win32process.CreateProcess(
            None,
            subprocess.list2cmdline(command),
            None,
            None,
            True,
            (
                win32process.CREATE_SUSPENDED
                | win32process.CREATE_NEW_PROCESS_GROUP
                | win32process.CREATE_UNICODE_ENVIRONMENT
            ),
            env,
            str(cwd),
            startup,
        )
        win32job.AssignProcessToJobObject(job, process_handle)
        win32process.ResumeThread(thread_handle)
        return process_handle
    except Exception:
        if process_handle is not None:
            try:
                win32api.TerminateProcess(process_handle, 1)
            except OSError:
                pass
            process_handle.Close()
        raise
    finally:
        if thread_handle is not None:
            thread_handle.Close()
        for handle, inherited in zip(handles, previous_inheritance, strict=True):
            os.set_handle_inheritable(handle, inherited)


def _wait_for_process_and_job(
    process_handle,
    job,
    *,
    command: tuple[str, ...],
    timeout_seconds: float,
    output_path: Path,
    max_output_bytes: int,
) -> int:
    (
        _msvcrt,
        _pywintypes,
        _win32api,
        win32con,
        win32event,
        win32job,
        win32process,
    ) = _modules()
    deadline = time.monotonic() + timeout_seconds
    return_code: int | None = None
    while True:
        try:
            output_size = output_path.stat().st_size
        except OSError:
            output_size = 0
        if output_size > max_output_bytes:
            raise WindowsInspectionOutputTooLarge
        if time.monotonic() >= deadline:
            raise subprocess.TimeoutExpired(command, timeout_seconds)
        result = win32event.WaitForSingleObject(process_handle, 50)
        if result != win32event.WAIT_TIMEOUT and return_code is None:
            return_code = int(win32process.GetExitCodeProcess(process_handle))
        if return_code is not None and not _job_has_processes(job, win32job):
            return 0 if return_code == win32con.STILL_ACTIVE else return_code


def _job_has_processes(job, win32job) -> bool:
    information = win32job.QueryInformationJobObject(
        job,
        win32job.JobObjectBasicProcessIdList,
    )
    if isinstance(information, dict):
        values = information.get("ProcessIdList", ())
    else:
        values = information or ()
    return bool(values)


def _terminate_job(job) -> None:
    (
        _msvcrt,
        _pywintypes,
        _win32api,
        _win32con,
        _win32event,
        win32job,
        _win32process,
    ) = _modules()
    try:
        win32job.TerminateJobObject(job, 1)
    except OSError:
        pass


__all__: list[str] = []

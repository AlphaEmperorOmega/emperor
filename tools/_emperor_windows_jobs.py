from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO


def _modules():
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


def service_job_object_name(service: str, port: int) -> str:
    safe = "".join(
        character for character in service if character.isalnum() or character in "-_"
    )
    if not safe or safe != service or not 1 <= port <= 65535:
        raise ValueError("Unsafe Workbench service Job Object identity.")
    return f"Local\\EmperorWorkbenchService-{safe}-{port}"


@dataclass(frozen=True, slots=True)
class WindowsJobLimits:
    memory_bytes: int
    cpu_count: int
    process_count: int


class WindowsJob:
    def __init__(self, handle: Any, *, name: str | None) -> None:
        self.handle = handle
        self.name = name

    @classmethod
    def create(
        cls,
        *,
        name: str | None,
        limits: WindowsJobLimits,
        inheritable: bool = True,
    ) -> WindowsJob:
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
        security.bInheritHandle = bool(inheritable)
        handle = win32job.CreateJobObject(security, name)
        job = cls(handle, name=name)
        if job.has_processes():
            job.close()
            raise OSError(f"Windows Job Object is already active: {name}")
        job._apply_limits(limits)
        return job

    @classmethod
    def open(cls, name: str) -> WindowsJob | None:
        try:
            (
                _msvcrt,
                _pywintypes,
                _win32api,
                win32con,
                _win32event,
                win32job,
                _win32process,
            ) = _modules()
            access = getattr(
                win32job,
                "JOB_OBJECT_ALL_ACCESS",
                getattr(win32con, "JOB_OBJECT_ALL_ACCESS", 0x1F001F),
            )
            handle = win32job.OpenJobObject(
                access,
                False,
                name,
            )
        except OSError:
            return None
        return cls(handle, name=name)

    def _apply_limits(self, limits: WindowsJobLimits) -> None:
        (
            _msvcrt,
            _pywintypes,
            _win32api,
            _win32con,
            _win32event,
            win32job,
            _win32process,
        ) = _modules()
        information = win32job.QueryInformationJobObject(
            self.handle,
            win32job.JobObjectExtendedLimitInformation,
        )
        basic = information["BasicLimitInformation"]
        basic["LimitFlags"] |= (
            getattr(win32job, "JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE", 0x2000)
            | getattr(win32job, "JOB_OBJECT_LIMIT_ACTIVE_PROCESS", 0x00000008)
            | getattr(win32job, "JOB_OBJECT_LIMIT_JOB_MEMORY", 0x00000200)
        )
        basic["ActiveProcessLimit"] = limits.process_count
        information["JobMemoryLimit"] = limits.memory_bytes
        win32job.SetInformationJobObject(
            self.handle,
            win32job.JobObjectExtendedLimitInformation,
            information,
        )
        available = os.cpu_count() or 1
        cpu_rate = min(10_000, max(1, int(10_000 * limits.cpu_count / available)))
        if cpu_rate < 10_000:
            win32job.SetInformationJobObject(
                self.handle,
                win32job.JobObjectCpuRateControlInformation,
                {
                    "ControlFlags": getattr(
                        win32job, "JOB_OBJECT_CPU_RATE_CONTROL_ENABLE", 0x1
                    )
                    | getattr(win32job, "JOB_OBJECT_CPU_RATE_CONTROL_HARD_CAP", 0x4),
                    "CpuRate": cpu_rate,
                },
            )

    def start_suspended(
        self,
        command: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
        stdout: BinaryIO,
        stderr: BinaryIO,
        stdin: BinaryIO | None = None,
        detached: bool = False,
    ) -> WindowsJobProcessHandle:
        (
            msvcrt,
            _pywintypes,
            win32api,
            win32con,
            _win32event,
            win32job,
            win32process,
        ) = _modules()
        owned_stdin = None
        if stdin is None:
            owned_stdin = open(os.devnull, "rb")
            stdin = owned_stdin
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
        creation_flags = (
            win32process.CREATE_SUSPENDED
            | win32process.CREATE_NEW_PROCESS_GROUP
            | win32process.CREATE_UNICODE_ENVIRONMENT
        )
        if detached:
            creation_flags |= getattr(
                win32process,
                "DETACHED_PROCESS",
                subprocess.DETACHED_PROCESS,
            )
        process_handle = None
        thread_handle = None
        try:
            process_handle, thread_handle, pid, _thread_id = win32process.CreateProcess(
                None,
                subprocess.list2cmdline(command),
                None,
                None,
                True,
                creation_flags,
                env,
                str(cwd),
                startup,
            )
            win32job.AssignProcessToJobObject(self.handle, process_handle)
            win32process.ResumeThread(thread_handle)
            return WindowsJobProcessHandle(
                pid=pid,
                command=tuple(command),
                process_handle=process_handle,
                job=self,
            )
        except Exception:
            if process_handle is not None:
                try:
                    win32api.TerminateProcess(process_handle, 1)
                except OSError:
                    pass
                process_handle.Close()
            self.close()
            raise
        finally:
            if thread_handle is not None:
                thread_handle.Close()
            for handle, inherited in zip(handles, previous_inheritance, strict=True):
                os.set_handle_inheritable(handle, inherited)
            if owned_stdin is not None:
                owned_stdin.close()

    def process_ids(self) -> tuple[int, ...]:
        (
            _msvcrt,
            _pywintypes,
            _win32api,
            _win32con,
            _win32event,
            win32job,
            _win32process,
        ) = _modules()
        information = win32job.QueryInformationJobObject(
            self.handle,
            win32job.JobObjectBasicProcessIdList,
        )
        if isinstance(information, dict):
            values = information.get("ProcessIdList", ())
        else:
            values = information or ()
        return tuple(int(value) for value in values)

    def has_processes(self) -> bool:
        try:
            return bool(self.process_ids())
        except OSError:
            return False

    def terminate(self, exit_code: int = 1) -> None:
        (
            _msvcrt,
            _pywintypes,
            _win32api,
            _win32con,
            _win32event,
            win32job,
            _win32process,
        ) = _modules()
        win32job.TerminateJobObject(self.handle, exit_code)

    def wait_empty(self, timeout: float | None = None) -> None:
        deadline = None if timeout is None else time.monotonic() + timeout
        while self.has_processes():
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("Timed out waiting for Windows Job Object to empty.")
            time.sleep(0.05)

    def close(self) -> None:
        handle, self.handle = self.handle, None
        if handle is not None:
            handle.Close()

    def __del__(self) -> None:  # pragma: no cover - defensive handle cleanup
        try:
            self.close()
        except Exception:
            pass


@dataclass(slots=True)
class WindowsJobProcessHandle:
    pid: int
    command: tuple[str, ...]
    process_handle: Any
    job: WindowsJob
    _exit_code: int | None = None

    def _primary_exit_code(self) -> int | None:
        (
            _msvcrt,
            _pywintypes,
            _win32api,
            win32con,
            _win32event,
            _win32job,
            win32process,
        ) = _modules()
        code = int(win32process.GetExitCodeProcess(self.process_handle))
        return None if code == win32con.STILL_ACTIVE else code

    def poll(self) -> int | None:
        code = self._primary_exit_code()
        if code is None or self.job.has_processes():
            return None
        self._exit_code = code
        return code

    def terminate(self) -> None:
        try:
            os.kill(self.pid, signal.CTRL_BREAK_EVENT)
        except (AttributeError, OSError):
            return

    def kill(self) -> None:
        self.job.terminate(1)

    def wait(self, timeout: float | None = None) -> int:
        (
            _msvcrt,
            _pywintypes,
            _win32api,
            _win32con,
            win32event,
            _win32job,
            _win32process,
        ) = _modules()
        deadline = None if timeout is None else time.monotonic() + timeout
        milliseconds = (
            win32event.INFINITE if timeout is None else max(0, int(timeout * 1000))
        )
        result = win32event.WaitForSingleObject(self.process_handle, milliseconds)
        if result == win32event.WAIT_TIMEOUT:
            raise subprocess.TimeoutExpired(self.command, timeout or 0.0)
        code = self._primary_exit_code()
        remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
        try:
            self.job.wait_empty(remaining)
        except TimeoutError as exc:
            raise subprocess.TimeoutExpired(self.command, timeout or 0.0) from exc
        self._exit_code = 0 if code is None else code
        return self._exit_code

    def has_live_containment(self) -> bool:
        return self.job.has_processes()


__all__ = [
    "WindowsJob",
    "WindowsJobLimits",
    "WindowsJobProcessHandle",
    "service_job_object_name",
]

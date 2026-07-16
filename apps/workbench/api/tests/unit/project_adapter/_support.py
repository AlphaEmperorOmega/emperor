from __future__ import annotations

import json
import subprocess
from io import BytesIO
from unittest.mock import Mock

from model_runtime.cli import PROTOCOL_VERSION


def _response(result: object) -> bytes:
    return json.dumps(
        {
            "version": PROTOCOL_VERSION,
            "ok": True,
            "result": result,
        }
    ).encode("utf-8")


class _RecordingInput(BytesIO):
    def __init__(self) -> None:
        super().__init__()
        self.captured = bytearray()

    def write(self, value: bytes) -> int:
        self.captured.extend(value)
        return super().write(value)


class _FakeOneShotProcess:
    def __init__(self, stdout: bytes, *, return_code: int = 0) -> None:
        self.stdin = _RecordingInput()
        self.stdout = BytesIO(stdout)
        self.returncode: int | None = return_code
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        if self.returncode is None:
            raise subprocess.TimeoutExpired("project-adapter", timeout or 0)
        return self.returncode


class _FakeProcess:
    def __init__(
        self,
        *responses: bytes,
        running: bool = True,
        ignore_terminate: bool = False,
    ) -> None:
        self.stdin = BytesIO()
        self.stdout = Mock()
        self.stdout.readline.side_effect = [*responses]
        self.return_code = None if running else 1
        self.ignore_terminate = ignore_terminate
        self.terminated = False
        self.killed = False
        self.wait_calls = 0

    def poll(self) -> int | None:
        return self.return_code

    def terminate(self) -> None:
        self.terminated = True
        if not self.ignore_terminate:
            self.return_code = -15

    def kill(self) -> None:
        self.killed = True
        self.return_code = -9

    def wait(self, timeout: float | None = None) -> int:
        self.wait_calls += 1
        if self.return_code is None:
            raise subprocess.TimeoutExpired("project-adapter", timeout or 0)
        return self.return_code

from __future__ import annotations

import os
import signal
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProcessGroupHandle:
    process: subprocess.Popen
    process_group_id: int | None = None

    @property
    def pid(self) -> int:
        return self.process.pid

    def poll(self) -> int | None:
        exit_code = self.process.poll()
        if exit_code is not None and self._process_group_exists():
            return None
        return exit_code

    def wait(self, timeout: float | None = None) -> int:
        deadline = None if timeout is None else time.monotonic() + timeout
        exit_code = self.process.wait(timeout=timeout)
        self._wait_for_process_group(deadline=deadline, timeout=timeout)
        return exit_code

    def terminate(self) -> None:
        self._signal(signal.SIGTERM, self.process.terminate)

    def kill(self) -> None:
        self._signal(getattr(signal, "SIGKILL", signal.SIGTERM), self.process.kill)

    def _signal(self, signum: int, fallback: Callable[[], None]) -> None:
        if self.process_group_id is None:
            fallback()
            return
        try:
            os.killpg(self.process_group_id, signum)
        except ProcessLookupError:
            return
        except OSError:
            fallback()

    def _wait_for_process_group(
        self,
        *,
        deadline: float | None,
        timeout: float | None,
    ) -> None:
        if self.process_group_id is None:
            return
        while self._process_group_exists():
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(
                        cmd=self.process.args,
                        timeout=timeout or 0.0,
                    )
                time.sleep(min(0.05, remaining))
            else:
                time.sleep(0.05)

    def _process_group_exists(self) -> bool:
        if self.process_group_id is None:
            return False
        try:
            os.killpg(self.process_group_id, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True


__all__ = ["ProcessGroupHandle"]

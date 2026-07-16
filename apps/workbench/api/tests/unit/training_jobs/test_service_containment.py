from __future__ import annotations

import json
import os
import signal
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from emperor_workbench.training_jobs import TrainingJobFailure
from emperor_workbench.training_jobs._containment._cgroup_v2 import (
    CgroupV2Manager,
    StrictCancellationUnavailable,
)
from emperor_workbench.training_jobs._containment._launcher import (
    TrainingWorkerLauncher,
)
from tests.support.training_jobs import (
    FakeRunner,
)
from tests.unit.training_jobs._support import (
    FakeCgroup,
    FakeCgroupManager,
    TrainingJobServiceHarness,
    create_restart_limitation_job,
)


class TrainingJobContainmentTests(unittest.TestCase):
    _create_restart_limitation_job = staticmethod(create_restart_limitation_job)

    def _wait_for_pid_file(self, path: Path, *, timeout: float = 5.0) -> int:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if path.is_file():
                text = path.read_text(encoding="utf-8").strip()
                if text:
                    return int(text)
            time.sleep(0.05)
        raise AssertionError(f"Timed out waiting for pid file: {path}")

    def _wait_for_pid_list(self, path: Path, *, timeout: float = 5.0) -> list[int]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if path.is_file():
                text = path.read_text(encoding="utf-8").strip()
                if text:
                    return [int(pid) for pid in json.loads(text)]
            time.sleep(0.05)
        raise AssertionError(f"Timed out waiting for pid list: {path}")

    def _process_is_alive(self, pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        stat_path = Path(f"/proc/{pid}/stat")
        if stat_path.is_file():
            try:
                stat = stat_path.read_text(encoding="utf-8")
            except OSError:
                return True
            fields = stat.split()
            if len(fields) > 2 and fields[2] == "Z":
                return False
        return True

    def _wait_for_process_exit(self, pid: int, *, timeout: float = 5.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self._process_is_alive(pid):
                return True
            time.sleep(0.05)
        return not self._process_is_alive(pid)

    def _kill_leftover_process(self, pid: int) -> None:
        if not self._process_is_alive(pid):
            return
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            return

    def test_strict_cgroup_unavailable_fails_training_start(self) -> None:
        class UnavailableCgroupManager:
            def __init__(self) -> None:
                self.probe_count = 0
                self.create_count = 0

            def is_available(self) -> bool:
                self.probe_count += 1
                return False

            def create_job_cgroup(self, job_id: str):
                self.create_count += 1
                raise StrictCancellationUnavailable(
                    "Strict training cancellation requires a writable cgroup."
                )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cgroups = UnavailableCgroupManager()
            manager = TrainingJobServiceHarness(
                root=root / "jobs",
                logs_root=root / "logs",
                worker_launcher=TrainingWorkerLauncher(
                    cwd=Path.cwd(),
                    runner=FakeRunner(),
                    cancellation_mode="strict-cgroup",
                    cgroup_manager=cgroups,
                ),
            )

            self.assertEqual(
                manager.service.cancellation_capability(),
                "unsupported",
            )
            with self.assertRaisesRegex(
                TrainingJobFailure,
                "requires a writable cgroup",
            ):
                manager.create_job_payload(
                    model="linears/linear",
                    preset="baseline",
                    datasets=["Mnist"],
                    overrides={},
                    log_folder="strict_unavailable",
                    monitors=[],
                )

        self.assertEqual(cgroups.probe_count, 1)
        self.assertEqual(cgroups.create_count, 1)

    def test_restart_can_cancel_persisted_strict_cgroup_job(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = TrainingJobServiceHarness(
                root=root / "jobs",
                logs_root=root / "logs",
                runner=FakeRunner(),
            )
            payload = manager.create_job_payload(
                model="linears/linear",
                preset="baseline",
                datasets=["Mnist"],
                overrides={},
                log_folder="restart_strict_cancel",
                monitors=[],
            )
            job = manager.jobs[str(payload["id"])]
            job.cancellation_mode = "strict-cgroup"
            job.worker_pid = job.pid
            job.process_group_id = job.pid
            job.cgroup_path = "/forged/victim/cgroup"
            manager.runtime.job_store.save(job)

            cgroup = FakeCgroup(ignores_terminate=True)
            cgroup_manager = FakeCgroupManager(cgroup)
            restarted = TrainingJobServiceHarness(
                root=root / "jobs",
                logs_root=root / "logs",
                runner=FakeRunner(),
                cgroup_manager=cgroup_manager,
            )

            with patch(
                "emperor_workbench.training_jobs._containment._launcher.os.killpg"
            ) as kill_process_group:
                cancelled = restarted.cancel_job_payload(str(payload["id"]))

        self.assertEqual(cancelled["status"], "cancelled")
        kill_process_group.assert_not_called()
        self.assertTrue(cgroup_manager.requested_job_ids)
        self.assertEqual(
            set(cgroup_manager.requested_job_ids),
            {str(payload["id"])},
        )
        self.assertTrue(cgroup.terminated)
        self.assertTrue(cgroup.killed)
        self.assertTrue(cgroup.cleaned)

    def test_restarted_strict_cgroup_empty_without_terminal_event_stays_unknown(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original, created, _ = self._create_restart_limitation_job(
                root,
                log_folder="restart_strict_unknown",
            )
            job_id = str(created["id"])
            job = original.jobs[job_id]
            job.cancellation_mode = "strict-cgroup"
            job.cgroup_path = "/persisted/path/is/not/authority"
            original.runtime.job_store.save(job)
            cgroup = FakeCgroup()
            restarted = TrainingJobServiceHarness(
                root=root / "jobs",
                logs_root=root / "logs",
                runner=FakeRunner(),
                cgroup_manager=FakeCgroupManager(cgroup),
            )

            observed_live = restarted.get_job_payload(job_id)
            cgroup.processes = False
            observed_empty = restarted.get_job_payload(job_id)
            active = restarted.active_job_payloads()

        self.assertEqual(observed_live["status"], "running")
        self.assertEqual(observed_empty["status"], "unknown")
        self.assertIsNone(observed_empty["exitCode"])
        self.assertEqual(
            active,
            [
                {
                    "id": job_id,
                    "status": "unknown",
                    "logFolder": "restart_strict_unknown",
                }
            ],
        )

    def test_restarted_empty_strict_cgroup_uses_terminal_event_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original, created, _ = self._create_restart_limitation_job(
                root,
                log_folder="restart_strict_terminal",
            )
            job_id = str(created["id"])
            job = original.jobs[job_id]
            job.cancellation_mode = "strict-cgroup"
            job.cgroup_path = "/persisted/path/is/not/authority"
            original.runtime._write_event(
                job,
                {"type": "completed", "status": "completed"},
            )
            original.runtime.job_store.save(job)
            cgroup = FakeCgroup()
            cgroup.processes = False
            restarted = TrainingJobServiceHarness(
                root=root / "jobs",
                logs_root=root / "logs",
                runner=FakeRunner(),
                cgroup_manager=FakeCgroupManager(cgroup),
            )

            recovered = restarted.get_job_payload(job_id)

        self.assertEqual(recovered["status"], "completed")
        self.assertEqual(recovered["exitCode"], 0)

    def test_cancel_job_terminates_real_worker_child_and_grandchild(self) -> None:
        parent_script = """
import subprocess
import sys

child_command = (
    "import json, os, pathlib, signal, subprocess, sys, time\\n"
    "grandchild = subprocess.Popen([sys.executable, '-c', "
    "'import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
    "time.sleep(60)'])\\n"
    "pathlib.Path(sys.argv[1]).write_text("
    "json.dumps([os.getpid(), grandchild.pid]), encoding='utf-8')\\n"
    "signal.signal(signal.SIGTERM, signal.SIG_IGN)\\n"
    "while True: time.sleep(1)"
)
child = subprocess.Popen([
    sys.executable,
    "-c",
    child_command,
    sys.argv[1],
])
"""

        class ChildSpawningLauncher(TrainingWorkerLauncher):
            def build_command(
                self,
                payload_path: Path,
                progress_path: Path,
            ) -> list[str]:
                child_pid_path = payload_path.parent / "child.pid"
                return [sys.executable, "-c", parent_script, str(child_pid_path)]

        descendant_pids: list[int] = []
        with tempfile.TemporaryDirectory() as tmp:
            try:
                root = Path(tmp)
                manager = TrainingJobServiceHarness(
                    root=root / "jobs",
                    logs_root=root / "logs",
                    worker_launcher=ChildSpawningLauncher(
                        cwd=Path.cwd(),
                        cancellation_mode="process-group",
                    ),
                )
                manager.runtime._cancel_reap_grace_seconds = 0.2
                payload = manager.create_job_payload(
                    model="linears/linear",
                    preset="baseline",
                    datasets=["Mnist"],
                    overrides={},
                    log_folder="cancel_process_group",
                    monitors=[],
                )
                child_pid_path = root / "jobs" / str(payload["id"]) / "child.pid"
                descendant_pids = self._wait_for_pid_list(child_pid_path)
                self.assertEqual(len(descendant_pids), 2)
                raw_process = getattr(
                    manager.runtime._processes[str(payload["id"])],
                    "process",
                    None,
                )
                self.assertIsNotNone(raw_process)
                self.assertEqual(raw_process.wait(timeout=1), 0)

                self.assertTrue(
                    all(self._process_is_alive(pid) for pid in descendant_pids)
                )
                cancelled = manager.cancel_job_payload(str(payload["id"]))
                self.assertEqual(cancelled["status"], "cancelled")
                self.assertTrue(
                    all(self._wait_for_process_exit(pid) for pid in descendant_pids),
                    f"descendant processes survived cancellation: {descendant_pids}",
                )
            finally:
                for pid in descendant_pids:
                    self._kill_leftover_process(pid)

    def test_cancel_job_terminates_escaped_session_child_with_cgroup(self) -> None:
        if not CgroupV2Manager().is_available():
            self.skipTest("writable/delegated cgroup v2 is unavailable")

        parent_script = """
import subprocess
import sys
import time
from pathlib import Path

child_command = (
    "import signal, time\\n"
    "signal.signal(signal.SIGTERM, signal.SIG_IGN)\\n"
    "while True: time.sleep(1)"
)
child = subprocess.Popen([
    sys.executable,
    "-c",
    child_command,
], start_new_session=True)
Path(sys.argv[1]).write_text(str(child.pid), encoding="utf-8")
while True:
    time.sleep(1)
"""

        class EscapedChildLauncher(TrainingWorkerLauncher):
            def build_command(
                self,
                payload_path: Path,
                progress_path: Path,
            ) -> list[str]:
                child_pid_path = payload_path.parent / "escaped-child.pid"
                return [sys.executable, "-c", parent_script, str(child_pid_path)]

        child_pid: int | None = None
        with tempfile.TemporaryDirectory() as tmp:
            try:
                root = Path(tmp)
                manager = TrainingJobServiceHarness(
                    root=root / "jobs",
                    logs_root=root / "logs",
                    worker_launcher=EscapedChildLauncher(
                        cwd=Path.cwd(),
                        cancellation_mode="strict-cgroup",
                    ),
                )
                manager.runtime._cancel_reap_grace_seconds = 0.2
                payload = manager.create_job_payload(
                    model="linears/linear",
                    preset="baseline",
                    datasets=["Mnist"],
                    overrides={},
                    log_folder="cancel_cgroup_escaped_child",
                    monitors=[],
                )
                child_pid_path = (
                    root / "jobs" / str(payload["id"]) / "escaped-child.pid"
                )
                child_pid = self._wait_for_pid_file(child_pid_path)

                self.assertTrue(self._process_is_alive(child_pid))
                cancelled = manager.cancel_job_payload(str(payload["id"]))
                self.assertEqual(cancelled["status"], "cancelled")
                self.assertTrue(
                    self._wait_for_process_exit(child_pid),
                    f"escaped child process {child_pid} survived cancellation",
                )
            finally:
                if child_pid is not None:
                    self._kill_leftover_process(child_pid)


if __name__ == "__main__":
    unittest.main()

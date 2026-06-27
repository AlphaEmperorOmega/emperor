from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import tempfile
import time
import unittest
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import viewer.backend.training_jobs as training_jobs
from viewer.backend.inspector.errors import InspectorError
from viewer.backend.job_store import InMemoryTrainingJobStore
from viewer.backend.tests.helpers import (
    FakeProcess,
    FakeRunner,
    create_progress_test_job,
)
from viewer.backend.training_cgroups import (
    CgroupV2Manager,
    StrictCancellationUnavailable,
)
from viewer.backend.training_jobs import TrainingJobManager
from viewer.backend.training_worker_launcher import TrainingWorkerLauncher


class FailingTrainingJobStore(InMemoryTrainingJobStore):
    def save(self, job) -> None:
        raise RuntimeError("job store failed")


class FakeCgroup:
    cgroup_path = "/sys/fs/cgroup/emperor-viewer-training/job-test"

    def __init__(self, *, ignores_terminate: bool = False) -> None:
        self.processes = True
        self.terminated = False
        self.killed = False
        self.cleaned = False
        self.ignores_terminate = ignores_terminate

    def has_processes(self) -> bool:
        return self.processes

    def terminate(self) -> None:
        self.terminated = True
        if not self.ignores_terminate:
            self.processes = False

    def kill(self) -> None:
        self.killed = True
        self.processes = False

    def wait_empty(self, timeout: float | None = None) -> None:
        if self.processes:
            raise TimeoutError("fake cgroup still has processes")

    def cleanup_empty(self) -> None:
        if not self.processes:
            self.cleaned = True


class FakeCgroupManager:
    def __init__(self, cgroup: FakeCgroup | None = None) -> None:
        self.cgroup = cgroup or FakeCgroup()

    def from_existing(self, cgroup_path: str | None):
        return self.cgroup if cgroup_path else None


class TrainingJobTests(unittest.TestCase):
    def _wait_for_pid_file(self, path: Path, *, timeout: float = 5.0) -> int:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if path.is_file():
                text = path.read_text(encoding="utf-8").strip()
                if text:
                    return int(text)
            time.sleep(0.05)
        raise AssertionError(f"Timed out waiting for pid file: {path}")

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

    def _create_progress_projection_job(
        self,
        root: Path,
        *,
        process: FakeProcess | None = None,
        run_count: int = 3,
        total_epochs: int = 4,
    ):
        datasets = ["Mnist", "Cifar10", "Cifar100"][:run_count]
        run_plan = {
            "runs": [
                {
                    "id": f"{dataset.lower()}-row",
                    "index": index,
                    "preset": "baseline",
                    "dataset": dataset,
                    "overrides": {},
                    "totalEpochs": total_epochs,
                }
                for index, dataset in enumerate(datasets, start=1)
            ]
        }
        manager = TrainingJobManager(
            root=root / "jobs",
            logs_root=root / "logs",
            runner=FakeRunner(process),
        )
        payload = manager.create_job(
            model="linears/linear",
            preset="baseline",
            datasets=datasets,
            overrides={},
            log_folder="progress_projection",
            run_plan=run_plan,
        )
        return manager, payload, manager.jobs[str(payload["id"])]

    def _create_restart_limitation_job(
        self,
        root: Path,
        *,
        process: FakeProcess | None = None,
        log_folder: str = "restart_limitation",
    ):
        process = process or FakeProcess()
        manager = TrainingJobManager(
            root=root / "jobs",
            logs_root=root / "logs",
            runner=FakeRunner(process),
        )
        payload = manager.create_job(
            model="linears/linear",
            preset="baseline",
            datasets=["Mnist"],
            overrides={},
            log_folder=log_folder,
            monitors=[],
        )
        return manager, payload, process

    def test_training_api_response_does_not_expose_manager_internals(self) -> None:
        import httpx

        from viewer.backend.api import ViewerApiSettings, create_app

        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            manager = TrainingJobManager(
                root=Path(tmp) / "jobs",
                logs_root=logs_root,
                runner=FakeRunner(),
            )

            async def call_api() -> httpx.Response:
                transport = httpx.ASGITransport(
                    app=create_app(
                        ViewerApiSettings(
                            logs_root=str(logs_root),
                            allow_unsafe_local_mutations=True,
                        ),
                        training_manager=manager,
                    )
                )
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                ) as client:
                    return await client.post(
                        "/training/jobs",
                        json={
                            "modelType": "linears",
                            "model": "linear",
                            "preset": "baseline",
                            "presets": ["baseline", "gating"],
                            "datasets": ["Mnist"],
                            "overrides": {"stack_hidden_dim": "128"},
                            "logFolder": "test_model",
                            "monitors": ["linear"],
                        },
                    )

            response = asyncio.run(call_api())

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "running")
        self.assertEqual(payload["preset"], "baseline")
        self.assertEqual(payload["presets"], ["baseline", "gating"])
        self.assertEqual(payload["pid"], 1234)
        for internal_key in ("command", "root", "process"):
            self.assertNotIn(internal_key, payload)

    def test_create_job_reaps_worker_if_registration_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            process = FakeProcess()
            manager = TrainingJobManager(
                root=root / "jobs",
                logs_root=root / "logs",
                runner=FakeRunner(process),
                job_store=FailingTrainingJobStore(),
            )

            with self.assertRaises(RuntimeError):
                manager.create_job(
                    model="linears/linear",
                    preset="baseline",
                    datasets=["Mnist"],
                    overrides={},
                    log_folder="registration_failure",
                    monitors=[],
                )

        self.assertTrue(process.terminated)
        self.assertFalse(process.killed)
        self.assertEqual(manager._processes, {})

    def test_cancel_job_reaps_process_after_terminate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            process = FakeProcess()
            manager = TrainingJobManager(
                root=root / "jobs",
                logs_root=root / "logs",
                runner=FakeRunner(process),
            )
            payload = manager.create_job(
                model="linears/linear",
                preset="baseline",
                datasets=["Mnist"],
                overrides={},
                log_folder="cancel_reap",
                monitors=[],
            )
            cancelled = manager.cancel_job(str(payload["id"]))

        self.assertEqual(cancelled["status"], "cancelled")
        self.assertTrue(process.terminated)
        self.assertFalse(process.killed)
        self.assertEqual(cancelled["exitCode"], -15)

    def test_cancel_job_kills_process_that_ignores_terminate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            process = FakeProcess(ignores_terminate=True)
            manager = TrainingJobManager(
                root=root / "jobs",
                logs_root=root / "logs",
                runner=FakeRunner(process),
            )
            payload = manager.create_job(
                model="linears/linear",
                preset="baseline",
                datasets=["Mnist"],
                overrides={},
                log_folder="cancel_kill",
                monitors=[],
            )
            cancelled = manager.cancel_job(str(payload["id"]))

        self.assertEqual(cancelled["status"], "cancelled")
        self.assertTrue(process.terminated)
        self.assertTrue(process.killed)
        self.assertEqual(cancelled["exitCode"], -9)

    def test_cancel_job_failure_keeps_job_running(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            process = FakeProcess(ignores_terminate=True, ignores_kill=True)
            manager = TrainingJobManager(
                root=root / "jobs",
                logs_root=root / "logs",
                runner=FakeRunner(process),
            )
            payload = manager.create_job(
                model="linears/linear",
                preset="baseline",
                datasets=["Mnist"],
                overrides={},
                log_folder="cancel_failure",
                monitors=[],
            )
            job_id = str(payload["id"])

            with self.assertRaisesRegex(
                InspectorError,
                "process survived terminate and kill",
            ):
                manager.cancel_job(job_id)
            current = manager.get_job(job_id)

        self.assertTrue(process.terminated)
        self.assertTrue(process.killed)
        self.assertEqual(current["status"], "running")
        self.assertIsNone(current["exitCode"])
        self.assertFalse(
            any(event.get("type") == "cancelled" for event in current["events"])
        )

    def test_strict_cgroup_unavailable_fails_training_start(self) -> None:
        class UnavailableCgroupManager:
            def create_job_cgroup(self, job_id: str):
                raise StrictCancellationUnavailable(
                    "Strict training cancellation requires a writable cgroup."
                )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = TrainingJobManager(
                root=root / "jobs",
                logs_root=root / "logs",
                worker_launcher=TrainingWorkerLauncher(
                    cwd=Path.cwd(),
                    runner=FakeRunner(),
                    cancellation_mode="strict-cgroup",
                    cgroup_manager=UnavailableCgroupManager(),
                ),
            )

            with self.assertRaisesRegex(
                InspectorError,
                "requires a writable cgroup",
            ):
                manager.create_job(
                    model="linears/linear",
                    preset="baseline",
                    datasets=["Mnist"],
                    overrides={},
                    log_folder="strict_unavailable",
                    monitors=[],
                )

    def test_terminal_progress_event_does_not_finish_live_process_scope(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            process = FakeProcess(exit_code=None)
            manager = TrainingJobManager(
                root=root / "jobs",
                logs_root=root / "logs",
                runner=FakeRunner(process),
            )
            payload = manager.create_job(
                model="linears/linear",
                preset="baseline",
                datasets=["Mnist"],
                overrides={},
                log_folder="terminal_event_live_scope",
                monitors=[],
            )
            job = manager.jobs[str(payload["id"])]
            manager._write_event(job, {"type": "completed", "status": "completed"})

            current = manager.get_job(str(payload["id"]))

        self.assertEqual(current["status"], "running")
        self.assertIsNone(current["exitCode"])

    def test_restart_can_cancel_persisted_strict_cgroup_job(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = TrainingJobManager(
                root=root / "jobs",
                logs_root=root / "logs",
                runner=FakeRunner(),
            )
            payload = manager.create_job(
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
            job.cgroup_path = FakeCgroup.cgroup_path
            manager.job_store.save(job)

            cgroup = FakeCgroup(ignores_terminate=True)
            restarted = TrainingJobManager(
                root=root / "jobs",
                logs_root=root / "logs",
                runner=FakeRunner(),
                cgroup_manager=FakeCgroupManager(cgroup),
            )

            cancelled = restarted.cancel_job(str(payload["id"]))

        self.assertEqual(cancelled["status"], "cancelled")
        self.assertTrue(cgroup.terminated)
        self.assertTrue(cgroup.killed)
        self.assertTrue(cgroup.cleaned)

    @unittest.skipIf(os.name != "posix", "process-group cancellation is POSIX-only")
    def test_cancel_job_terminates_real_worker_child_processes(self) -> None:
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
])
Path(sys.argv[1]).write_text(str(child.pid), encoding="utf-8")
"""

        class ChildSpawningLauncher(TrainingWorkerLauncher):
            def build_command(
                self,
                payload_path: Path,
                progress_path: Path,
            ) -> list[str]:
                child_pid_path = payload_path.parent / "child.pid"
                return [sys.executable, "-c", parent_script, str(child_pid_path)]

        child_pid: int | None = None
        original_grace_seconds = training_jobs.CANCEL_REAP_GRACE_SECONDS
        training_jobs.CANCEL_REAP_GRACE_SECONDS = 0.2
        with tempfile.TemporaryDirectory() as tmp:
            try:
                root = Path(tmp)
                manager = TrainingJobManager(
                    root=root / "jobs",
                    logs_root=root / "logs",
                    worker_launcher=ChildSpawningLauncher(
                        cwd=Path.cwd(),
                        cancellation_mode="process-group",
                    ),
                )
                payload = manager.create_job(
                    model="linears/linear",
                    preset="baseline",
                    datasets=["Mnist"],
                    overrides={},
                    log_folder="cancel_process_group",
                    monitors=[],
                )
                child_pid_path = root / "jobs" / str(payload["id"]) / "child.pid"
                child_pid = self._wait_for_pid_file(child_pid_path)
                raw_process = getattr(
                    manager._processes[str(payload["id"])],
                    "process",
                    None,
                )
                self.assertIsNotNone(raw_process)
                self.assertEqual(raw_process.wait(timeout=1), 0)

                self.assertTrue(self._process_is_alive(child_pid))
                cancelled = manager.cancel_job(str(payload["id"]))
                self.assertEqual(cancelled["status"], "cancelled")
                self.assertTrue(
                    self._wait_for_process_exit(child_pid),
                    f"child process {child_pid} survived cancellation",
                )
            finally:
                training_jobs.CANCEL_REAP_GRACE_SECONDS = original_grace_seconds
                if child_pid is not None:
                    self._kill_leftover_process(child_pid)

    @unittest.skipIf(os.name != "posix", "cgroup cancellation is POSIX-only")
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
        original_grace_seconds = training_jobs.CANCEL_REAP_GRACE_SECONDS
        training_jobs.CANCEL_REAP_GRACE_SECONDS = 0.2
        with tempfile.TemporaryDirectory() as tmp:
            try:
                root = Path(tmp)
                manager = TrainingJobManager(
                    root=root / "jobs",
                    logs_root=root / "logs",
                    worker_launcher=EscapedChildLauncher(
                        cwd=Path.cwd(),
                        cancellation_mode="strict-cgroup",
                    ),
                )
                payload = manager.create_job(
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
                cancelled = manager.cancel_job(str(payload["id"]))
                self.assertEqual(cancelled["status"], "cancelled")
                self.assertTrue(
                    self._wait_for_process_exit(child_pid),
                    f"escaped child process {child_pid} survived cancellation",
                )
            finally:
                training_jobs.CANCEL_REAP_GRACE_SECONDS = original_grace_seconds
                if child_pid is not None:
                    self._kill_leftover_process(child_pid)

    def test_training_api_cancel_job_terminates_process(self) -> None:
        import httpx

        from viewer.backend.api import ViewerApiSettings, create_app

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs_root = root / "logs"
            process = FakeProcess()
            manager = TrainingJobManager(
                root=root / "jobs",
                logs_root=logs_root,
                runner=FakeRunner(process),
            )
            test_app = create_app(
                ViewerApiSettings(
                    logs_root=str(logs_root),
                    allow_unsafe_local_mutations=True,
                ),
                training_manager=manager,
            )

            async def call_api() -> tuple[
                httpx.Response, httpx.Response, httpx.Response
            ]:
                transport = httpx.ASGITransport(app=test_app)
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                ) as client:
                    create_response = await client.post(
                        "/training/jobs",
                        json={
                            "modelType": "linears",
                            "model": "linear",
                            "preset": "baseline",
                            "datasets": ["Mnist"],
                            "overrides": {},
                            "logFolder": "cancel_api",
                            "monitors": [],
                        },
                    )
                    job_id = create_response.json()["id"]
                    cancel_response = await client.post(
                        f"/training/jobs/{job_id}/cancel"
                    )
                    unknown_response = await client.post(
                        "/training/jobs/missing/cancel"
                    )
                    return create_response, cancel_response, unknown_response

            create_response, cancel_response, unknown_response = asyncio.run(call_api())

        self.assertEqual(create_response.status_code, 200, create_response.text)
        job_id = create_response.json()["id"]
        self.assertEqual(cancel_response.status_code, 200, cancel_response.text)
        payload = cancel_response.json()
        self.assertEqual(payload["id"], job_id)
        self.assertEqual(payload["status"], "cancelled")
        self.assertTrue(process.terminated)
        self.assertEqual(payload["events"][-1]["type"], "cancelled")
        self.assertEqual(payload["events"][-1]["status"], "cancelled")
        self.assertEqual(payload["events"][-1]["jobId"], job_id)
        self.assertEqual(
            [run["status"] for run in payload["runPlan"]["runs"]],
            ["Skipped"],
        )
        self.assertEqual(payload["runPlan"]["summary"]["pendingRuns"], 0)
        self.assertEqual(payload["runPlan"]["summary"]["cancelledRuns"], 0)
        self.assertEqual(payload["runPlan"]["summary"]["skippedRuns"], 1)
        for internal_key in ("command", "root", "process"):
            self.assertNotIn(internal_key, payload)

        self.assertEqual(unknown_response.status_code, 400)
        self.assertEqual(
            unknown_response.json(),
            {"detail": "Unknown training job 'missing'."},
        )

    def test_training_api_created_job_uses_safe_worker_command_paths(self) -> None:
        import httpx

        from viewer.backend.api import ViewerApiSettings, create_app

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jobs_root = root / "jobs"
            logs_root = root / "logs"
            runner = FakeRunner()
            manager = TrainingJobManager(
                root=jobs_root,
                logs_root=logs_root,
                runner=runner,
            )
            test_app = create_app(
                ViewerApiSettings(
                    logs_root=str(logs_root),
                    allow_unsafe_local_mutations=True,
                ),
                training_manager=manager,
            )

            async def call_api() -> httpx.Response:
                transport = httpx.ASGITransport(app=test_app)
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                ) as client:
                    return await client.post(
                        "/training/jobs",
                        json={
                            "modelType": "linears",
                            "model": "linear",
                            "preset": "baseline",
                            "presets": ["baseline"],
                            "datasets": ["Mnist"],
                            "overrides": {},
                            "logFolder": "path_safety",
                            "monitors": [],
                        },
                    )

            response = asyncio.run(call_api())

            self.assertEqual(response.status_code, 200, response.text)
            job_id = response.json()["id"]
            self.assertEqual(len(job_id), 32)
            int(job_id, 16)
            job_root = jobs_root / job_id
            expected_payload = job_root / "payload.json"
            expected_progress = job_root / "progress.jsonl"
            expected_log = job_root / "training.log"

            self.assertEqual(len(runner.commands), 1)
            command = runner.commands[0]
            self.assertIsInstance(command, list)
            self.assertTrue(all(isinstance(part, str) for part in command))
            self.assertEqual(
                command,
                [
                    sys.executable,
                    "-m",
                    "viewer.backend.training_worker",
                    "--payload",
                    str(expected_payload),
                    "--progress",
                    str(expected_progress),
                ],
            )
            self.assertEqual(runner.log_paths, [expected_log])

            resolved_job_root = job_root.resolve()
            for path in (expected_payload, expected_progress, expected_log):
                with self.subTest(path=path):
                    self.assertTrue(
                        path.resolve().is_relative_to(resolved_job_root),
                        f"{path} should stay under {resolved_job_root}",
                    )

    def test_training_job_creation_uses_fake_process_runner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runner = FakeRunner()
            logs_root = Path(tmp) / "logs"
            manager = TrainingJobManager(
                root=Path(tmp), logs_root=logs_root, runner=runner
            )

            payload = manager.create_job(
                model="linears/linear",
                preset="baseline",
                datasets=["Mnist"],
                overrides={"stack_hidden_dim": "128"},
                log_folder="test_model",
                monitors=["linear"],
            )
            payload_path = Path(tmp) / payload["id"] / "payload.json"
            worker_payload = json.loads(payload_path.read_text())

            self.assertEqual(payload["status"], "running")
            self.assertEqual(payload["preset"], "baseline")
            self.assertEqual(payload["presets"], ["baseline"])
            self.assertEqual(payload["datasets"], ["Mnist"])
            self.assertEqual(payload["monitors"], ["linear"])
            self.assertEqual(payload["logFolder"], "test_model")
            self.assertEqual(worker_payload["monitors"], ["linear"])
            self.assertEqual(worker_payload["preset"], "baseline")
            self.assertEqual(worker_payload["presets"], ["baseline"])
            self.assertEqual(worker_payload["logFolder"], "test_model")
            self.assertEqual(payload["pid"], 1234)
            self.assertTrue((logs_root / "test_model").is_dir())
            self.assertTrue(runner.commands)
            self.assertIn("viewer.backend.training_worker", runner.commands[0])

    def test_training_job_manager_saves_created_job_to_injected_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            process = FakeProcess()
            store = InMemoryTrainingJobStore()
            manager = TrainingJobManager(
                root=root / "jobs",
                logs_root=root / "logs",
                runner=FakeRunner(process),
                job_store=store,
            )

            payload = manager.create_job(
                model="linears/linear",
                preset="baseline",
                datasets=["Mnist"],
                overrides={},
                log_folder="stored_job",
                monitors=[],
            )
            record = store.get(str(payload["id"]))
            self.assertIsNotNone(record)
            assert record is not None
            self.assertEqual(record.model, "linears/linear")
            self.assertEqual(record.preset, "baseline")
            self.assertEqual(record.datasets, ["Mnist"])
            self.assertEqual(record.log_folder, "stored_job")
            self.assertEqual(record.pid, 1234)
            self.assertFalse(hasattr(record, "process"))
            self.assertIs(manager.jobs[str(payload["id"])], record)

            cancelled = manager.cancel_job(str(payload["id"]))

            self.assertTrue(process.terminated)
            self.assertEqual(cancelled["status"], "cancelled")
            self.assertEqual(record.status, "cancelled")
            self.assertEqual(store.get(str(payload["id"])).status, "cancelled")

    def test_training_job_get_refreshes_process_completion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            process = FakeProcess()
            manager = TrainingJobManager(
                root=root / "jobs",
                logs_root=root / "logs",
                runner=FakeRunner(process),
            )
            created_payload = manager.create_job(
                model="linears/linear",
                preset="baseline",
                datasets=["Mnist"],
                overrides={},
                log_folder="completed_refresh",
                monitors=[],
            )
            job = manager.jobs[str(created_payload["id"])]
            manager._write_event(
                job,
                {
                    "type": "dataset_started",
                    "dataset": "Mnist",
                    "preset": "baseline",
                    "runIndex": 1,
                    "status": "running",
                },
            )

            process.exit_code = 0
            payload = manager.get_job(str(created_payload["id"]))

        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["exitCode"], 0)
        self.assertEqual(payload["id"], created_payload["id"])
        self.assertEqual(payload["pid"], 1234)
        self.assertEqual(payload["runPlan"]["runs"][0]["status"], "Completed")
        self.assertEqual(payload["runPlan"]["summary"]["completedRuns"], 1)

    def test_training_job_get_refreshes_process_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            process = FakeProcess()
            manager = TrainingJobManager(
                root=root / "jobs",
                logs_root=root / "logs",
                runner=FakeRunner(process),
            )
            created_payload = manager.create_job(
                model="linears/linear",
                preset="baseline",
                datasets=["Mnist"],
                overrides={},
                log_folder="failed_refresh",
                monitors=[],
            )

            process.exit_code = 2
            payload = manager.get_job(str(created_payload["id"]))

        failed_run = payload["runPlan"]["runs"][0]
        self.assertEqual(payload["status"], "failed")
        self.assertEqual(payload["exitCode"], 2)
        self.assertEqual(failed_run["status"], "Failed")
        self.assertEqual(failed_run["error"], "Training failed")
        self.assertEqual(payload["runPlan"]["summary"]["failedRuns"], 1)

    def test_training_job_missing_progress_jsonl_returns_empty_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager, created_payload, progress_path = create_progress_test_job(
                Path(tmp)
            )
            progress_path.unlink()

            payload = manager.get_job(str(created_payload["id"]))

        self.assertEqual(set(payload), set(created_payload))
        self.assertEqual(payload["events"], [])
        self.assertEqual(payload["logTail"], ["fake training log"])
        self.assertEqual(payload["currentPreset"], None)
        self.assertEqual(payload["currentDataset"], None)
        self.assertEqual(payload["metrics"], {})
        self.assertEqual(payload["resultLinks"], [])

    def test_training_job_progress_jsonl_ignores_blank_and_malformed_lines(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager, created_payload, progress_path = create_progress_test_job(
                Path(tmp)
            )
            valid_events = [
                {
                    "type": "job_started",
                    "status": "running",
                    "jobId": created_payload["id"],
                    "runTotal": 1,
                },
                {
                    "type": "dataset_started",
                    "dataset": "Mnist",
                    "preset": "baseline",
                    "runIndex": 1,
                    "status": "running",
                },
                {
                    "type": "validation",
                    "dataset": "Mnist",
                    "preset": "baseline",
                    "runIndex": 1,
                    "epoch": 0,
                    "step": 7,
                    "metrics": {"val/loss": 0.25},
                    "logDir": "logs/linear/baseline/Mnist/version_0",
                },
            ]
            progress_path.write_text(
                "\n".join(
                    [
                        json.dumps(valid_events[0]),
                        "",
                        "  ",
                        "{not json",
                        json.dumps(valid_events[1]),
                        "[",
                        json.dumps(valid_events[2]),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            payload = manager.get_job(str(created_payload["id"]))

        self.assertEqual(payload["events"], valid_events)
        self.assertEqual(
            [event["type"] for event in payload["events"]],
            ["job_started", "dataset_started", "validation"],
        )
        self.assertEqual(payload["currentPreset"], "baseline")
        self.assertEqual(payload["currentDataset"], "Mnist")
        self.assertEqual(payload["step"], 7)
        self.assertEqual(payload["metrics"], {"val/loss": 0.25})
        self.assertEqual(payload["logTail"], ["fake training log"])

    def test_training_job_invalid_utf8_progress_jsonl_raises_decode_error(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager, created_payload, progress_path = create_progress_test_job(
                Path(tmp)
            )
            progress_path.write_bytes(b'{"type": "job_started"}\n\xff\n')

            with self.assertRaises(UnicodeDecodeError):
                manager.get_job(str(created_payload["id"]))

    def test_training_job_progress_jsonl_read_failure_raises_os_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager, created_payload, progress_path = create_progress_test_job(
                Path(tmp)
            )
            progress_path.unlink()
            progress_path.mkdir()

            with self.assertRaises(IsADirectoryError):
                manager.get_job(str(created_payload["id"]))

    def test_training_job_large_progress_jsonl_is_bounded_with_paginated_history(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager, created_payload, progress_path = create_progress_test_job(
                Path(tmp)
            )
            events = [
                {
                    "type": "step",
                    "dataset": "Mnist",
                    "preset": "baseline",
                    "runIndex": 1,
                    "step": step,
                }
                for step in range(125)
            ]
            progress_path.write_text(
                "\n".join(json.dumps(event) for event in events) + "\n",
                encoding="utf-8",
            )

            payload = manager.get_job(str(created_payload["id"]))
            history = manager.get_job_events(
                str(created_payload["id"]),
                offset=0,
                limit=200,
            )

        self.assertEqual(payload["eventCount"], 125)
        self.assertEqual(payload["eventCounts"], {"step": 125})
        self.assertTrue(payload["eventsTruncated"])
        self.assertEqual(len(payload["events"]), 100)
        self.assertEqual(
            [event["step"] for event in payload["events"]],
            list(range(25, 125)),
        )
        self.assertEqual(payload["step"], 124)
        self.assertEqual(payload["logTail"], ["fake training log"])
        self.assertEqual(history["totalCount"], 125)
        self.assertIsNone(history["nextOffset"])
        self.assertEqual(history["events"], events)

    def test_training_job_live_payload_stays_small_for_large_progress_history(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager, created_payload, progress_path = create_progress_test_job(
                Path(tmp)
            )
            events = [
                {
                    "type": "step",
                    "dataset": "Mnist",
                    "preset": "baseline",
                    "runIndex": 1,
                    "step": step,
                    "metrics": {"train/loss": 1.0},
                }
                for step in range(150_000)
            ]
            progress_path.write_text(
                "\n".join(json.dumps(event) for event in events) + "\n",
                encoding="utf-8",
            )

            payload = manager.get_job(str(created_payload["id"]))
            encoded = json.dumps(payload)

        self.assertEqual(payload["eventCount"], 150_000)
        self.assertEqual(payload["eventCounts"], {"step": 150_000})
        self.assertTrue(payload["eventsTruncated"])
        self.assertEqual(len(payload["events"]), 100)
        self.assertLess(len(encoded), 250_000)

    def test_training_job_serializes_result_links_and_log_tail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager, created_payload, progress_path = create_progress_test_job(
                Path(tmp)
            )
            job = manager.jobs[str(created_payload["id"])]
            log_dir = "logs/progress_jsonl/linear/baseline/Mnist/version_0"
            events = [
                {
                    "type": "job_started",
                    "status": "running",
                    "jobId": created_payload["id"],
                    "runTotal": 1,
                },
                {
                    "type": "dataset_completed",
                    "dataset": "Mnist",
                    "preset": "baseline",
                    "runIndex": 1,
                    "logDir": log_dir,
                },
            ]
            progress_path.write_text(
                "\n".join(json.dumps(event) for event in events) + "\n",
                encoding="utf-8",
            )
            job.log_path.write_text(
                "\n".join(f"log line {index}" for index in range(85)) + "\n",
                encoding="utf-8",
            )

            payload = manager.get_job(str(created_payload["id"]))

        self.assertEqual(
            payload["resultLinks"],
            [
                {
                    "preset": "baseline",
                    "dataset": "Mnist",
                    "logDir": log_dir,
                }
            ],
        )
        self.assertEqual(
            payload["logTail"],
            [f"log line {index}" for index in range(5, 85)],
        )
        self.assertEqual(payload["logDir"], log_dir)

    def test_training_job_projects_cluster_growth_without_full_event_history(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager, created_payload, progress_path = create_progress_test_job(
                Path(tmp)
            )
            events = [
                {
                    "type": "cluster_initialized",
                    "node": "root.cluster",
                    "count": 1,
                    "capacity": [4, 1, 1],
                    "coordinates": [[1, 1, 1]],
                },
                *[
                    {
                        "type": "neuron_added",
                        "node": "root.cluster",
                        "coord": [index, 1, 1],
                        "count": index,
                        "capacity": [4, 1, 1],
                        "step": index * 10,
                    }
                    for index in range(2, 5)
                ],
            ]
            progress_path.write_text(
                "\n".join(json.dumps(event) for event in events) + "\n",
                encoding="utf-8",
            )

            payload = manager.get_job(str(created_payload["id"]))

        self.assertEqual(
            payload["clusterGrowth"],
            [
                {
                    "node": "root.cluster",
                    "count": 4,
                    "capacityTotal": 4,
                    "additionCount": 3,
                    "additions": [
                        {"coord": [2, 1, 1], "step": 20, "epoch": None},
                        {"coord": [3, 1, 1], "step": 30, "epoch": None},
                        {"coord": [4, 1, 1], "step": 40, "epoch": None},
                    ],
                }
            ],
        )

    def test_training_job_rejects_symlink_top_level_log_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs_root = root / "logs"
            logs_root.mkdir()
            outside_logs = root / "outside_logs"
            outside_logs.mkdir()
            logs_root.joinpath("linked").symlink_to(
                outside_logs,
                target_is_directory=True,
            )
            runner = FakeRunner()
            manager = TrainingJobManager(
                root=root / "jobs",
                logs_root=logs_root,
                runner=runner,
            )

            with self.assertRaises(InspectorError) as context:
                manager.create_job(
                    model="linears/linear",
                    preset="baseline",
                    datasets=["Mnist"],
                    overrides={},
                    log_folder="linked",
                    monitors=[],
                )

        self.assertEqual(
            str(context.exception),
            "Refusing to write symlink log experiment: linked",
        )
        self.assertEqual(runner.commands, [])
        self.assertEqual(runner.log_paths, [])
        self.assertEqual(manager.jobs, {})

    def test_training_job_get_unknown_id_raises_inspector_error(self) -> None:
        manager = TrainingJobManager(runner=FakeRunner())

        with self.assertRaises(InspectorError) as context:
            manager.get_job("missing")

        self.assertEqual(str(context.exception), "Unknown training job 'missing'.")

    def test_training_job_restart_behavior_fresh_manager_gets_persisted_disk_job(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, created_payload, _ = self._create_restart_limitation_job(root)
            job_id = str(created_payload["id"])
            job_root = root / "jobs" / job_id
            self.assertTrue(job_root.joinpath("payload.json").is_file())
            self.assertTrue(job_root.joinpath("progress.jsonl").is_file())
            fresh_manager = TrainingJobManager(
                root=root / "jobs",
                logs_root=root / "logs",
                runner=FakeRunner(),
            )

            payload = fresh_manager.get_job(job_id)

        self.assertEqual(payload["id"], job_id)
        self.assertEqual(payload["status"], "unknown")
        self.assertEqual(payload["modelType"], "linears")
        self.assertEqual(payload["model"], "linear")
        self.assertEqual(payload["preset"], "baseline")
        self.assertEqual(payload["presets"], ["baseline"])
        self.assertEqual(payload["datasets"], ["Mnist"])
        self.assertEqual(payload["logFolder"], "restart_limitation")
        self.assertEqual(payload["pid"], 1234)

    def test_restart_fresh_manager_cannot_cancel_disk_job_without_live_process(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            process = FakeProcess()
            _, created_payload, _ = self._create_restart_limitation_job(
                root,
                process=process,
                log_folder="restart_cancel_limitation",
            )
            job_id = str(created_payload["id"])
            fresh_manager = TrainingJobManager(
                root=root / "jobs",
                logs_root=root / "logs",
                runner=FakeRunner(),
            )

            with self.assertRaises(InspectorError) as context:
                fresh_manager.cancel_job(job_id)

            self.assertFalse(process.terminated)
            self.assertIsNone(process.exit_code)

        self.assertEqual(
            str(context.exception),
            f"Training job '{job_id}' has no live process handle.",
        )

    def test_restart_fresh_manager_preserves_unknown_disk_job_blocker(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original_manager, created_payload, _ = self._create_restart_limitation_job(
                root,
                log_folder="restart_active_limitation",
            )
            job_id = str(created_payload["id"])
            self.assertEqual(
                original_manager.active_jobs(),
                [
                    {
                        "id": job_id,
                        "status": "running",
                        "logFolder": "restart_active_limitation",
                    }
                ],
            )
            self.assertTrue((root / "jobs" / job_id / "payload.json").is_file())
            self.assertTrue((root / "jobs" / job_id / "progress.jsonl").is_file())
            fresh_manager = TrainingJobManager(
                root=root / "jobs",
                logs_root=root / "logs",
                runner=FakeRunner(),
            )

            self.assertEqual(
                fresh_manager.active_jobs(),
                [
                    {
                        "id": job_id,
                        "status": "unknown",
                        "logFolder": "restart_active_limitation",
                    }
                ],
            )

    def test_training_job_restart_behavior_fresh_manager_reconstructs_disk_job(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original_manager, created_payload, _ = self._create_restart_limitation_job(
                root,
                log_folder="restart_behavior",
            )
            job_id = str(created_payload["id"])
            job = original_manager.jobs[job_id]
            run = created_payload["runPlan"]["runs"][0]
            log_dir = "logs/restart_behavior/linear/baseline/Mnist/version_0"
            original_manager._write_event(
                job,
                {
                    "type": "dataset_started",
                    "status": "running",
                    "dataset": "Mnist",
                    "preset": "baseline",
                    "runId": run["id"],
                    "runIndex": 1,
                    "logDir": log_dir,
                },
            )
            original_manager._write_event(
                job,
                {
                    "type": "validation",
                    "status": "running",
                    "dataset": "Mnist",
                    "preset": "baseline",
                    "runId": run["id"],
                    "runIndex": 1,
                    "epoch": 1,
                    "step": 4,
                    "metrics": {"validation/accuracy": 0.75},
                    "logDir": log_dir,
                },
            )
            job_root = root / "jobs" / job_id
            self.assertTrue(job_root.joinpath("payload.json").is_file())
            self.assertTrue(job_root.joinpath("progress.jsonl").is_file())

            fresh_manager = TrainingJobManager(
                root=root / "jobs",
                logs_root=root / "logs",
                runner=FakeRunner(),
            )
            payload = fresh_manager.get_job(job_id)

        self.assertEqual(payload["id"], job_id)
        self.assertEqual(payload["status"], "unknown")
        self.assertEqual(payload["modelType"], "linears")
        self.assertEqual(payload["model"], "linear")
        self.assertEqual(payload["preset"], "baseline")
        self.assertEqual(payload["presets"], ["baseline"])
        self.assertEqual(payload["datasets"], ["Mnist"])
        self.assertEqual(payload["logFolder"], "restart_behavior")
        self.assertEqual(payload["plannedRunCount"], 1)
        self.assertEqual(
            [event["type"] for event in payload["events"]],
            ["job_started", "dataset_started", "validation"],
        )
        self.assertEqual(payload["currentPreset"], "baseline")
        self.assertEqual(payload["currentDataset"], "Mnist")
        self.assertEqual(payload["step"], 4)
        self.assertEqual(payload["metrics"], {"validation/accuracy": 0.75})
        self.assertEqual(payload["logDir"], log_dir)
        self.assertEqual(payload["runPlan"]["runs"][0]["status"], "Running")
        self.assertEqual(payload["runPlan"]["summary"]["runningRuns"], 1)

    def test_restart_fresh_manager_uses_completed_progress_event(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original_manager, created_payload, _ = self._create_restart_limitation_job(
                root,
                log_folder="restart_completed_behavior",
            )
            job_id = str(created_payload["id"])
            job = original_manager.jobs[job_id]
            run = created_payload["runPlan"]["runs"][0]
            log_dir = "logs/restart_completed_behavior/linear/baseline/Mnist/version_0"
            original_manager._write_event(
                job,
                {
                    "type": "dataset_completed",
                    "status": "running",
                    "dataset": "Mnist",
                    "preset": "baseline",
                    "runId": run["id"],
                    "runIndex": 1,
                    "metrics": {"validation/accuracy": 0.8},
                    "logDir": log_dir,
                },
            )
            original_manager._write_event(
                job,
                {
                    "type": "completed",
                    "status": "completed",
                    "jobId": job_id,
                    "preset": "baseline",
                    "presets": ["baseline"],
                },
            )

            fresh_manager = TrainingJobManager(
                root=root / "jobs",
                logs_root=root / "logs",
                runner=FakeRunner(),
            )
            payload = fresh_manager.get_job(job_id)

        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["exitCode"], 0)
        self.assertEqual(payload["eventCounts"]["completed"], 1)
        self.assertEqual(payload["runPlan"]["summary"]["completedRuns"], 1)
        self.assertEqual(fresh_manager.active_jobs(), [])
        self.assertEqual(fresh_manager.jobs[job_id].status, "completed")
        self.assertEqual(fresh_manager.jobs[job_id].exit_code, 0)

    def test_restart_fresh_manager_lists_unknown_disk_job_as_active(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original_manager, created_payload, _ = self._create_restart_limitation_job(
                root,
                log_folder="restart_active_behavior",
            )
            job_id = str(created_payload["id"])
            self.assertEqual(
                original_manager.active_jobs(),
                [
                    {
                        "id": job_id,
                        "status": "running",
                        "logFolder": "restart_active_behavior",
                    }
                ],
            )

            fresh_manager = TrainingJobManager(
                root=root / "jobs",
                logs_root=root / "logs",
                runner=FakeRunner(),
            )

            self.assertEqual(
                fresh_manager.active_jobs(),
                [
                    {
                        "id": job_id,
                        "status": "unknown",
                        "logFolder": "restart_active_behavior",
                    }
                ],
            )

    def test_training_job_restart_behavior_reconstructed_job_is_non_cancellable(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            process = FakeProcess()
            _, created_payload, _ = self._create_restart_limitation_job(
                root,
                process=process,
                log_folder="restart_read_only_behavior",
            )
            job_id = str(created_payload["id"])
            fresh_manager = TrainingJobManager(
                root=root / "jobs",
                logs_root=root / "logs",
                runner=FakeRunner(),
            )

            recovered = fresh_manager.get_job(job_id)
            self.assertEqual(recovered["status"], "unknown")
            with self.assertRaisesRegex(
                InspectorError,
                "live process handle|after restart",
            ):
                fresh_manager.cancel_job(job_id)

            self.assertFalse(process.terminated)
            self.assertIsNone(process.exit_code)

    def test_training_job_restart_behavior_unknown_ids_remain_current_errors(
        self,
    ) -> None:
        import httpx

        from viewer.backend.api import ViewerApiSettings, create_app

        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            manager = TrainingJobManager(
                root=Path(tmp) / "jobs",
                logs_root=logs_root,
                runner=FakeRunner(),
            )

            with self.assertRaises(InspectorError) as get_context:
                manager.get_job("missing")
            with self.assertRaises(InspectorError) as cancel_context:
                manager.cancel_job("missing")

            async def call_api() -> tuple[httpx.Response, httpx.Response]:
                transport = httpx.ASGITransport(
                    app=create_app(
                        ViewerApiSettings(
                            logs_root=str(logs_root),
                            allow_unsafe_local_mutations=True,
                        ),
                        training_manager=manager,
                    )
                )
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                ) as client:
                    get_response = await client.get("/training/jobs/missing")
                    cancel_response = await client.post("/training/jobs/missing/cancel")
                    return get_response, cancel_response

            get_response, cancel_response = asyncio.run(call_api())

        self.assertEqual(str(get_context.exception), "Unknown training job 'missing'.")
        self.assertEqual(
            str(cancel_context.exception),
            "Unknown training job 'missing'.",
        )
        self.assertEqual(get_response.status_code, 400)
        self.assertEqual(cancel_response.status_code, 400)
        self.assertEqual(
            get_response.json(),
            {"detail": "Unknown training job 'missing'."},
        )
        self.assertEqual(
            cancel_response.json(),
            {"detail": "Unknown training job 'missing'."},
        )

    def test_training_api_get_unknown_job_returns_http_400(self) -> None:
        import httpx

        from viewer.backend.api import ViewerApiSettings, create_app

        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            manager = TrainingJobManager(
                root=Path(tmp) / "jobs",
                logs_root=logs_root,
                runner=FakeRunner(),
            )

            async def call_api() -> httpx.Response:
                transport = httpx.ASGITransport(
                    app=create_app(
                        ViewerApiSettings(
                            logs_root=str(logs_root),
                            allow_unsafe_local_mutations=True,
                        ),
                        training_manager=manager,
                    )
                )
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                ) as client:
                    return await client.get("/training/jobs/missing")

            response = asyncio.run(call_api())

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.json(),
            {"detail": "Unknown training job 'missing'."},
        )

    def test_training_job_manager_active_jobs_excludes_terminal_statuses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            running_manager = TrainingJobManager(
                root=root / "running",
                logs_root=root / "logs-running",
                runner=FakeRunner(FakeProcess()),
            )
            running_job = running_manager.create_job(
                model="linears/linear",
                preset="baseline",
                datasets=["Mnist"],
                overrides={},
                log_folder="running_model",
                monitors=[],
            )
            self.assertEqual(
                running_manager.active_jobs(),
                [
                    {
                        "id": running_job["id"],
                        "status": "running",
                        "logFolder": "running_model",
                    }
                ],
            )

            completed_manager = TrainingJobManager(
                root=root / "completed",
                logs_root=root / "logs-completed",
                runner=FakeRunner(FakeProcess(exit_code=0)),
            )
            completed_manager.create_job(
                model="linears/linear",
                preset="baseline",
                datasets=["Mnist"],
                overrides={},
                log_folder="completed_model",
                monitors=[],
            )
            self.assertEqual(completed_manager.active_jobs(), [])

            failed_manager = TrainingJobManager(
                root=root / "failed",
                logs_root=root / "logs-failed",
                runner=FakeRunner(FakeProcess(exit_code=1)),
            )
            failed_manager.create_job(
                model="linears/linear",
                preset="baseline",
                datasets=["Mnist"],
                overrides={},
                log_folder="failed_model",
                monitors=[],
            )
            self.assertEqual(failed_manager.active_jobs(), [])

            cancelled_manager = TrainingJobManager(
                root=root / "cancelled",
                logs_root=root / "logs-cancelled",
                runner=FakeRunner(FakeProcess()),
            )
            cancelled_job = cancelled_manager.create_job(
                model="linears/linear",
                preset="baseline",
                datasets=["Mnist"],
                overrides={},
                log_folder="cancelled_model",
                monitors=[],
            )
            cancelled_manager.cancel_job(cancelled_job["id"])
            self.assertEqual(cancelled_manager.active_jobs(), [])

    def test_training_job_accepts_multiple_presets_and_multiplies_run_count(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = TrainingJobManager(
                root=Path(tmp) / "jobs",
                logs_root=Path(tmp) / "logs",
                runner=FakeRunner(),
            )

            payload = manager.create_job(
                model="linears/linear",
                preset="baseline",
                presets=["baseline", "gating", "baseline"],
                datasets=["Mnist", "Cifar10"],
                overrides={},
                log_folder="multi_preset",
            )
            payload_path = Path(tmp) / "jobs" / payload["id"] / "payload.json"
            worker_payload = json.loads(payload_path.read_text())

        self.assertEqual(payload["preset"], "baseline")
        self.assertEqual(payload["presets"], ["baseline", "gating"])
        self.assertEqual(payload["plannedRunCount"], 4)
        self.assertEqual(worker_payload["presets"], ["baseline", "gating"])

    def test_training_job_rejects_unknown_selected_preset(self) -> None:
        manager = TrainingJobManager(runner=FakeRunner())

        with self.assertRaises(InspectorError):
            manager.create_job(
                model="linears/linear",
                preset="baseline",
                presets=["baseline", "missing-preset"],
                datasets=["Mnist"],
                overrides={},
                log_folder="test_model",
            )

    def test_training_job_rejects_path_like_dataset_input_before_side_effects(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner = FakeRunner()
            manager = TrainingJobManager(
                root=root / "jobs",
                logs_root=root / "logs",
                runner=runner,
            )

            with self.assertRaises(InspectorError) as context:
                manager.create_job(
                    model="linears/linear",
                    preset="baseline",
                    datasets=["./Mnist"],
                    overrides={},
                    log_folder="path_like_dataset",
                )

            self.assertEqual(manager.jobs, {})
            self.assertEqual(manager.active_jobs(), [])
            self.assertEqual(runner.commands, [])
            self.assertFalse((root / "jobs").exists())
            self.assertFalse((root / "logs" / "path_like_dataset").exists())

        message = str(context.exception)
        self.assertIn("./Mnist", message)
        self.assertIn("filesystem path", message)
        self.assertIn("server-known dataset name", message)

    def test_training_run_plan_preserves_error_traceback_from_progress_events(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = TrainingJobManager(
                root=Path(tmp) / "jobs",
                logs_root=Path(tmp) / "logs",
                runner=FakeRunner(),
            )
            payload = manager.create_job(
                model="linears/linear",
                preset="baseline",
                datasets=["Mnist"],
                overrides={},
                log_folder="traceback_test",
            )
            job = manager.jobs[payload["id"]]
            manager._write_event(
                job,
                {
                    "type": "error",
                    "status": "failed",
                    "dataset": "Mnist",
                    "preset": "baseline",
                    "runId": payload["runPlan"]["runs"][0]["id"],
                    "error": "scalar conversion failed",
                    "traceback": (
                        "Traceback (most recent call last):\n"
                        "RuntimeError: scalar conversion failed"
                    ),
                },
            )

            failed_payload = manager.get_job(payload["id"])

        failed_run = failed_payload["runPlan"]["runs"][0]
        self.assertEqual(failed_run["status"], "Failed")
        self.assertEqual(failed_run["error"], "scalar conversion failed")
        self.assertIn("RuntimeError", failed_run["errorTraceback"])

    def test_training_run_progress_projection_tracks_running_epoch_metrics_and_log_dir(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager, created_payload, job = self._create_progress_projection_job(
                Path(tmp),
            )
            run = created_payload["runPlan"]["runs"][0]
            log_dir = "logs/progress_projection/linear/baseline/Mnist/version_0"
            manager._write_event(
                job,
                {
                    "type": "dataset_started",
                    "status": "running",
                    "dataset": "Mnist",
                    "preset": "baseline",
                    "runId": run["id"],
                    "runIndex": 1,
                    "logDir": log_dir,
                },
            )

            started_payload = manager.get_job(str(created_payload["id"]))

            manager._write_event(
                job,
                {
                    "type": "epoch_started",
                    "status": "running",
                    "dataset": "Mnist",
                    "preset": "baseline",
                    "runId": run["id"],
                    "runIndex": 1,
                    "epoch": 1,
                    "step": 3,
                    "logDir": log_dir,
                },
            )
            manager._write_event(
                job,
                {
                    "type": "step",
                    "status": "running",
                    "dataset": "Mnist",
                    "preset": "baseline",
                    "runId": run["id"],
                    "runIndex": 1,
                    "epoch": 2,
                    "step": 7,
                    "metrics": {"train/loss": 0.4},
                    "logDir": log_dir,
                },
            )
            manager._write_event(
                job,
                {
                    "type": "validation",
                    "status": "running",
                    "dataset": "Mnist",
                    "preset": "baseline",
                    "runId": run["id"],
                    "runIndex": 1,
                    "epoch": 0,
                    "step": 8,
                    "metrics": {"validation/accuracy": 0.75},
                    "logDir": log_dir,
                },
            )

            progressed_payload = manager.get_job(str(created_payload["id"]))

        started_run = started_payload["runPlan"]["runs"][0]
        self.assertEqual(started_run["status"], "Running")
        self.assertEqual(started_run["currentEpoch"], 0)
        self.assertEqual(started_run["logDir"], log_dir)
        self.assertEqual(started_payload["runPlan"]["summary"]["runningRuns"], 1)
        self.assertEqual(started_payload["runPlan"]["summary"]["pendingRuns"], 2)

        progressed_run = progressed_payload["runPlan"]["runs"][0]
        self.assertEqual(progressed_run["status"], "Running")
        self.assertEqual(progressed_run["currentEpoch"], 3)
        self.assertEqual(progressed_run["metrics"], {"validation/accuracy": 0.75})
        self.assertEqual(progressed_run["logDir"], log_dir)
        self.assertEqual(progressed_payload["currentPreset"], "baseline")
        self.assertEqual(progressed_payload["currentDataset"], "Mnist")
        self.assertEqual(progressed_payload["step"], 8)
        self.assertEqual(progressed_payload["metrics"], {"validation/accuracy": 0.75})
        self.assertEqual(progressed_payload["logDir"], log_dir)
        self.assertEqual(
            progressed_payload["runPlan"]["summary"],
            {
                "totalRuns": 3,
                "completedRuns": 0,
                "runningRuns": 1,
                "pendingRuns": 2,
                "failedRuns": 0,
                "cancelledRuns": 0,
                "skippedRuns": 0,
                "totalEpochs": 12,
                "completedEpochs": 3,
                "remainingEpochs": 9,
            },
        )

    def test_training_run_progress_projection_completes_run_and_updates_summary(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager, created_payload, job = self._create_progress_projection_job(
                Path(tmp),
            )
            run = created_payload["runPlan"]["runs"][0]
            log_dir = "logs/progress_projection/linear/baseline/Mnist/version_0"
            manager._write_event(
                job,
                {
                    "type": "dataset_completed",
                    "status": "running",
                    "dataset": "Mnist",
                    "preset": "baseline",
                    "runId": run["id"],
                    "runIndex": 1,
                    "metrics": {"validation/accuracy": 0.82},
                    "logDir": log_dir,
                },
            )

            payload = manager.get_job(str(created_payload["id"]))

        completed_run = payload["runPlan"]["runs"][0]
        self.assertEqual(completed_run["status"], "Completed")
        self.assertEqual(completed_run["currentEpoch"], 4)
        self.assertEqual(completed_run["metrics"], {"validation/accuracy": 0.82})
        self.assertEqual(completed_run["logDir"], log_dir)
        self.assertEqual(
            [run["status"] for run in payload["runPlan"]["runs"]],
            ["Completed", "Pending", "Pending"],
        )
        self.assertEqual(
            payload["runPlan"]["summary"],
            {
                "totalRuns": 3,
                "completedRuns": 1,
                "runningRuns": 0,
                "pendingRuns": 2,
                "failedRuns": 0,
                "cancelledRuns": 0,
                "skippedRuns": 0,
                "totalEpochs": 12,
                "completedEpochs": 4,
                "remainingEpochs": 8,
            },
        )
        self.assertEqual(
            payload["resultLinks"],
            [
                {
                    "preset": "baseline",
                    "dataset": "Mnist",
                    "logDir": log_dir,
                }
            ],
        )

    def test_failed_event_preserves_traceback_and_summary(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager, created_payload, job = self._create_progress_projection_job(
                Path(tmp),
                process=FakeProcess(exit_code=1),
            )
            run = created_payload["runPlan"]["runs"][0]
            log_dir = "logs/progress_projection/linear/baseline/Mnist/version_0"
            manager._write_event(
                job,
                {
                    "type": "dataset_started",
                    "status": "running",
                    "dataset": "Mnist",
                    "preset": "baseline",
                    "runId": run["id"],
                    "runIndex": 1,
                    "logDir": log_dir,
                },
            )
            manager._write_event(
                job,
                {
                    "type": "step",
                    "status": "running",
                    "dataset": "Mnist",
                    "preset": "baseline",
                    "runId": run["id"],
                    "runIndex": 1,
                    "epoch": 1,
                    "step": 12,
                    "metrics": {"train/loss": 0.6},
                    "logDir": log_dir,
                },
            )
            manager._write_event(
                job,
                {
                    "type": "error",
                    "status": "failed",
                    "dataset": "Mnist",
                    "preset": "baseline",
                    "runId": run["id"],
                    "runIndex": 1,
                    "epoch": 2,
                    "step": 13,
                    "error": "optimizer exploded",
                    "traceback": (
                        "Traceback (most recent call last):\n"
                        "RuntimeError: optimizer exploded"
                    ),
                    "logDir": log_dir,
                },
            )

            payload = manager.get_job(str(created_payload["id"]))

        failed_run = payload["runPlan"]["runs"][0]
        self.assertEqual(payload["status"], "failed")
        self.assertEqual(failed_run["status"], "Failed")
        self.assertEqual(failed_run["currentEpoch"], 3)
        self.assertEqual(failed_run["error"], "optimizer exploded")
        self.assertIn("RuntimeError", failed_run["errorTraceback"])
        self.assertEqual(failed_run["metrics"], {"train/loss": 0.6})
        self.assertEqual(
            [run["status"] for run in payload["runPlan"]["runs"]],
            ["Failed", "Skipped", "Skipped"],
        )
        self.assertEqual(payload["step"], 13)
        self.assertEqual(payload["metrics"], {"train/loss": 0.6})
        self.assertEqual(
            payload["runPlan"]["summary"],
            {
                "totalRuns": 3,
                "completedRuns": 0,
                "runningRuns": 0,
                "pendingRuns": 0,
                "failedRuns": 1,
                "cancelledRuns": 0,
                "skippedRuns": 2,
                "totalEpochs": 12,
                "completedEpochs": 3,
                "remainingEpochs": 0,
            },
        )

    def test_failed_process_marks_running_and_pending_rows(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            process = FakeProcess()
            manager, created_payload, job = self._create_progress_projection_job(
                Path(tmp),
                process=process,
            )
            runs = created_payload["runPlan"]["runs"]
            manager._write_event(
                job,
                {
                    "type": "dataset_completed",
                    "status": "running",
                    "dataset": "Mnist",
                    "preset": "baseline",
                    "runId": runs[0]["id"],
                    "runIndex": 1,
                    "metrics": {"validation/accuracy": 0.82},
                    "logDir": (
                        "logs/progress_projection/linear/baseline/Mnist/version_0"
                    ),
                },
            )
            manager._write_event(
                job,
                {
                    "type": "dataset_started",
                    "status": "running",
                    "dataset": "Cifar10",
                    "preset": "baseline",
                    "runId": runs[1]["id"],
                    "runIndex": 2,
                    "logDir": (
                        "logs/progress_projection/linear/baseline/Cifar10/version_0"
                    ),
                },
            )
            manager._write_event(
                job,
                {
                    "type": "step",
                    "status": "running",
                    "dataset": "Cifar10",
                    "preset": "baseline",
                    "runId": runs[1]["id"],
                    "runIndex": 2,
                    "epoch": 1,
                    "step": 9,
                    "metrics": {"train/loss": 0.7},
                },
            )
            process.exit_code = 2

            payload = manager.get_job(str(created_payload["id"]))

        projected_runs = payload["runPlan"]["runs"]
        self.assertEqual(payload["status"], "failed")
        self.assertEqual(payload["exitCode"], 2)
        self.assertEqual(
            [run["status"] for run in projected_runs],
            ["Completed", "Failed", "Skipped"],
        )
        self.assertEqual(projected_runs[1]["currentEpoch"], 2)
        self.assertIsNone(projected_runs[1]["error"])
        self.assertIsNone(projected_runs[2]["error"])
        self.assertEqual(
            payload["runPlan"]["summary"],
            {
                "totalRuns": 3,
                "completedRuns": 1,
                "runningRuns": 0,
                "pendingRuns": 0,
                "failedRuns": 1,
                "cancelledRuns": 0,
                "skippedRuns": 1,
                "totalEpochs": 12,
                "completedEpochs": 6,
                "remainingEpochs": 0,
            },
        )

    def test_cancelled_job_marks_running_and_pending_rows(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            process = FakeProcess()
            manager, created_payload, job = self._create_progress_projection_job(
                Path(tmp),
                process=process,
            )
            run = created_payload["runPlan"]["runs"][0]
            log_dir = "logs/progress_projection/linear/baseline/Mnist/version_0"
            manager._write_event(
                job,
                {
                    "type": "dataset_started",
                    "status": "running",
                    "dataset": "Mnist",
                    "preset": "baseline",
                    "runId": run["id"],
                    "runIndex": 1,
                    "logDir": log_dir,
                },
            )
            manager._write_event(
                job,
                {
                    "type": "step",
                    "status": "running",
                    "dataset": "Mnist",
                    "preset": "baseline",
                    "runId": run["id"],
                    "runIndex": 1,
                    "epoch": 1,
                    "step": 6,
                    "metrics": {"train/loss": 0.9},
                    "logDir": log_dir,
                },
            )

            manager.cancel_job(str(created_payload["id"]))
            payload = manager.get_job(str(created_payload["id"]))

        projected_runs = payload["runPlan"]["runs"]
        self.assertEqual(payload["status"], "cancelled")
        self.assertTrue(process.terminated)
        self.assertEqual(payload["events"][-1]["type"], "cancelled")
        self.assertEqual(
            [run["status"] for run in projected_runs],
            ["Cancelled", "Skipped", "Skipped"],
        )
        self.assertEqual(projected_runs[0]["currentEpoch"], 2)
        self.assertEqual(projected_runs[0]["metrics"], {"train/loss": 0.9})
        self.assertEqual(projected_runs[0]["logDir"], log_dir)
        self.assertEqual(
            payload["runPlan"]["summary"],
            {
                "totalRuns": 3,
                "completedRuns": 0,
                "runningRuns": 0,
                "pendingRuns": 0,
                "failedRuns": 0,
                "cancelledRuns": 1,
                "skippedRuns": 2,
                "totalEpochs": 12,
                "completedEpochs": 2,
                "remainingEpochs": 0,
            },
        )

    def test_training_job_rejects_invalid_search_requests(self) -> None:
        manager = TrainingJobManager(runner=FakeRunner())

        invalid_searches = [
            {"mode": "grid", "values": {"missing_axis": [1]}},
            {"mode": "grid", "values": {"stack_hidden_dim": []}},
            {
                "mode": "random",
                "values": {"stack_hidden_dim": [64]},
                "randomSamples": 0,
            },
            {"mode": "grid", "values": {"stack_hidden_dim": [999]}},
            {"mode": "grid", "values": {}},
        ]

        for search in invalid_searches:
            with self.subTest(search=search):
                with self.assertRaises(InspectorError):
                    manager.create_job(
                        model="linears/linear",
                        preset="baseline",
                        datasets=["Mnist"],
                        overrides={},
                        search=search,
                        log_folder="invalid_search",
                    )

    def test_training_job_rejects_locked_search_axis(self) -> None:
        manager = TrainingJobManager(runner=FakeRunner())

        with self.assertRaises(InspectorError) as context:
            manager.create_job(
                model="linears/linear",
                preset="baseline",
                presets=["baseline", "post-norm"],
                datasets=["Mnist"],
                overrides={},
                search={
                    "mode": "grid",
                    "values": {"stack_layer_norm_position": ["BEFORE", "AFTER"]},
                },
                log_folder="locked_search",
            )

        self.assertIn("locked", str(context.exception))

    def test_training_job_rejects_invalid_log_folders(self) -> None:
        manager = TrainingJobManager(runner=FakeRunner())

        for log_folder in (
            "",
            "my experiment",
            "my-experiment",
            "my.folder",
            "my/folder",
            "_my_folder",
            "my_folder_",
            "my__folder",
        ):
            with self.subTest(log_folder=log_folder):
                with self.assertRaises(InspectorError):
                    manager.create_job(
                        model="linears/linear",
                        preset="baseline",
                        datasets=["Mnist"],
                        overrides={},
                        log_folder=log_folder,
                    )

    def test_training_job_rejects_unknown_monitor(self) -> None:
        manager = TrainingJobManager(runner=FakeRunner())

        with self.assertRaises(InspectorError):
            manager.create_job(
                model="linears/linear",
                preset="baseline",
                datasets=["Mnist"],
                overrides={},
                log_folder="test_model",
                monitors=["sampler"],
            )

    def test_training_job_rejects_locked_overrides(self) -> None:
        manager = TrainingJobManager(runner=FakeRunner())

        with self.assertRaises(InspectorError):
            manager.create_job(
                model="linears/linear",
                preset="baseline",
                presets=["baseline", "gating"],
                datasets=["Mnist"],
                overrides={"gate_flag": "false"},
                log_folder="test_model",
            )


if __name__ == "__main__":
    unittest.main()

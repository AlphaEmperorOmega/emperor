from __future__ import annotations

import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from emperor_workbench.failures import FailureKind
from emperor_workbench.training_jobs import TrainingJobFailure
from emperor_workbench.training_jobs._progress_projection import (
    TrainingLiveProjectionCache,
)
from emperor_workbench.training_jobs._progress_store import TrainingProgressStore
from tests.support.training_jobs import (
    FakeProcess,
    FakeRunner,
)
from tests.unit.training_jobs._support import (
    TrainingJobServiceHarness,
)


class TrainingJobReconciliationTests(unittest.TestCase):
    def test_operator_reconciles_only_unknown_job_without_live_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            process = FakeProcess()
            manager = TrainingJobServiceHarness(
                root=root / "jobs",
                logs_root=root / "logs",
                runner=FakeRunner(process),
            )
            created = manager.create_job_payload(
                model="linears/linear",
                preset="baseline",
                datasets=["Mnist"],
                overrides={},
                log_folder="unknown_reconciliation",
                monitors=[],
            )
            job_id = str(created["id"])
            job = manager.jobs[job_id]
            manager.runtime._processes.pop(job_id)
            job.status = "unknown"
            job.pid = 2_000_000_000
            job.worker_pid = 2_000_000_000
            job.process_group_id = None
            manager.runtime.job_store.save(job)

            reconciled = manager.reconcile_job_payload(
                job_id,
                action="mark-failed",
                reason="  operator verified the worker is gone  ",
            )
            events = manager.get_job_events_payload(job_id)["events"]

            self.assertEqual(reconciled["status"], "failed")
            self.assertIsNone(reconciled["exitCode"])
            self.assertEqual(manager.active_job_payloads(), [])
            self.assertEqual(events[-1]["type"], "operator_reconciled_failed")
            self.assertEqual(
                events[-1]["reason"],
                "operator verified the worker is gone",
            )
            with self.assertRaises(TrainingJobFailure) as terminal:
                manager.reconcile_job_payload(
                    job_id,
                    action="mark-failed",
                    reason="retry",
                )

        self.assertEqual(terminal.exception.kind, FailureKind.CONFLICT)

    def test_operator_cannot_reconcile_unknown_job_with_live_process(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = TrainingJobServiceHarness(
                root=root / "jobs",
                logs_root=root / "logs",
                runner=FakeRunner(FakeProcess()),
            )
            created = manager.create_job_payload(
                model="linears/linear",
                preset="baseline",
                datasets=["Mnist"],
                overrides={},
                log_folder="live_reconciliation",
                monitors=[],
            )
            job_id = str(created["id"])
            job = manager.jobs[job_id]
            job.status = "unknown"
            manager.runtime.job_store.save(job)

            with self.assertRaises(TrainingJobFailure) as live:
                manager.reconcile_job_payload(
                    job_id,
                    action="mark-failed",
                    reason="incorrect operator assertion",
                )

        self.assertEqual(live.exception.kind, FailureKind.CONFLICT)

    def test_concurrent_terminal_release_preserves_bounded_read_checkpoints(
        self,
    ) -> None:
        class CountingProgressStore(TrainingProgressStore):
            def __init__(self) -> None:
                super().__init__()
                self.evict_count = 0
                self.count_lock = Lock()

            def evict(self, job) -> None:
                with self.count_lock:
                    self.evict_count += 1
                super().evict(job)

        class CountingProjectionCache(TrainingLiveProjectionCache):
            def __init__(self) -> None:
                super().__init__()
                self.evict_count = 0
                self.count_lock = Lock()

            def evict(self, job_id: str) -> None:
                with self.count_lock:
                    self.evict_count += 1
                super().evict(job_id)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            process = FakeProcess()
            progress_store = CountingProgressStore()
            manager = TrainingJobServiceHarness(
                root=root / "jobs",
                logs_root=root / "logs",
                runner=FakeRunner(process),
            )
            manager.runtime.progress_store = progress_store
            projection_cache = CountingProjectionCache()
            manager.runtime._live_projection_cache = projection_cache
            created = manager.create_job_payload(
                model="linears/linear",
                preset="baseline",
                datasets=["Mnist"],
                overrides={},
                log_folder="concurrent_release",
                monitors=[],
            )
            job = manager.jobs[str(created["id"])]
            process.exit_code = 0
            job.status = "completed"
            job.exit_code = 0
            manager.runtime.job_store.save(job)

            with ThreadPoolExecutor(max_workers=16) as pool:
                list(
                    pool.map(
                        lambda _: manager.runtime._release_terminal_resources(job),
                        range(32),
                    )
                )

        self.assertEqual(progress_store.evict_count, 0)
        self.assertEqual(projection_cache.evict_count, 0)
        self.assertIn(job.id, manager.runtime._released_job_ids)

    def test_concurrent_get_active_and_cancel_share_one_lifecycle_transition(
        self,
    ) -> None:
        class CountingProcess(FakeProcess):
            def __init__(self) -> None:
                super().__init__()
                self.terminate_count = 0
                self.count_lock = Lock()

            def terminate(self) -> None:
                with self.count_lock:
                    self.terminate_count += 1
                super().terminate()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            process = CountingProcess()
            manager = TrainingJobServiceHarness(
                root=root / "jobs",
                logs_root=root / "logs",
                runner=FakeRunner(process),
            )
            created = manager.create_job_payload(
                model="linears/linear",
                preset="baseline",
                datasets=["Mnist"],
                overrides={},
                log_folder="concurrent_lifecycle",
                monitors=[],
            )
            job_id = str(created["id"])

            def operate(index: int):
                operation = index % 3
                if operation == 0:
                    return manager.get_job_payload(job_id)
                if operation == 1:
                    return manager.active_job_payloads()
                return manager.cancel_job_payload(job_id)

            with ThreadPoolExecutor(max_workers=16) as pool:
                futures = [pool.submit(operate, index) for index in range(96)]
                results = [future.result(timeout=10) for future in futures]

            final = manager.get_job_payload(job_id)

        self.assertEqual(len(results), 96)
        self.assertEqual(final["status"], "cancelled")
        self.assertEqual(final["eventCounts"]["cancelled"], 1)
        self.assertEqual(process.terminate_count, 1)

    def test_training_job_manager_active_jobs_excludes_terminal_statuses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            running_manager = TrainingJobServiceHarness(
                root=root / "running",
                logs_root=root / "logs-running",
                runner=FakeRunner(FakeProcess()),
            )
            running_job = running_manager.create_job_payload(
                model="linears/linear",
                preset="baseline",
                datasets=["Mnist"],
                overrides={},
                log_folder="running_model",
                monitors=[],
            )
            self.assertEqual(
                running_manager.active_job_payloads(),
                [
                    {
                        "id": running_job["id"],
                        "status": "running",
                        "logFolder": "running_model",
                    }
                ],
            )

            completed_manager = TrainingJobServiceHarness(
                root=root / "completed",
                logs_root=root / "logs-completed",
                runner=FakeRunner(FakeProcess(exit_code=0)),
            )
            completed_manager.create_job_payload(
                model="linears/linear",
                preset="baseline",
                datasets=["Mnist"],
                overrides={},
                log_folder="completed_model",
                monitors=[],
            )
            self.assertEqual(completed_manager.active_job_payloads(), [])

            failed_manager = TrainingJobServiceHarness(
                root=root / "failed",
                logs_root=root / "logs-failed",
                runner=FakeRunner(FakeProcess(exit_code=1)),
            )
            failed_manager.create_job_payload(
                model="linears/linear",
                preset="baseline",
                datasets=["Mnist"],
                overrides={},
                log_folder="failed_model",
                monitors=[],
            )
            self.assertEqual(failed_manager.active_job_payloads(), [])

            cancelled_manager = TrainingJobServiceHarness(
                root=root / "cancelled",
                logs_root=root / "logs-cancelled",
                runner=FakeRunner(FakeProcess()),
            )
            cancelled_job = cancelled_manager.create_job_payload(
                model="linears/linear",
                preset="baseline",
                datasets=["Mnist"],
                overrides={},
                log_folder="cancelled_model",
                monitors=[],
            )
            cancelled_manager.cancel_job_payload(cancelled_job["id"])
            self.assertEqual(cancelled_manager.active_job_payloads(), [])


if __name__ == "__main__":
    unittest.main()

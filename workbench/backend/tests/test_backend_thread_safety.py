from __future__ import annotations

import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

from workbench.backend.run_history.query import LogRunQueryService
from workbench.backend.run_history.scanner import LogRunScanner
from workbench.backend.tensorboard.readers import (
    TensorBoardMonitorReader,
    TensorBoardParameterStatusReader,
)
from workbench.backend.tests.helpers import (
    TrainingJobServiceHarness,
)
from workbench.backend.training_jobs.progress import (
    TrainingProgressSnapshot,
    TrainingProgressStore,
)
from workbench.backend.training_jobs.run_plan_adapter import (
    training_run_plan_from_payload,
)
from workbench.backend.training_jobs.store import (
    FileSystemTrainingJobStore,
    InMemoryTrainingJobStore,
    TrainingJobRecord,
)


def training_job_record(job_id: str, root: Path) -> TrainingJobRecord:
    return TrainingJobRecord(
        id=job_id,
        model="linears/linear",
        preset="baseline",
        presets=["baseline"],
        datasets=["mnist"],
        overrides={},
        search=None,
        planned_run_count=1,
        run_plan=training_run_plan_from_payload({
            "modelType": "linears",
            "model": "linear",
            "preset": "baseline",
            "presets": ["baseline"],
            "experimentTask": "classification",
            "datasets": ["mnist"],
            "overrides": {},
            "search": None,
            "logFolder": "logs",
            "isRandomSearch": False,
            "runs": [
                {
                    "id": "run-1",
                    "dataset": "mnist",
                    "preset": "baseline",
                    "status": "Pending",
                    "currentEpoch": 0,
                    "totalEpochs": 2,
                }
            ],
            "summary": {"totalRuns": 1, "pendingRuns": 1},
        }),
        monitors=[],
        log_folder="logs",
        command=["python", "-m", "worker"],
        root=root / job_id,
        pid=1000,
    )


class BackendThreadSafetyTests(unittest.TestCase):
    def test_in_memory_training_job_store_handles_concurrent_mutation(self) -> None:
        store = InMemoryTrainingJobStore()

        def save_and_read(index: int) -> None:
            job = training_job_record(f"job-{index}", Path("/tmp/jobs"))
            store.save(job)
            self.assertIs(store.get(job.id), job)
            self.assertGreaterEqual(len(store.list()), 1)

        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(save_and_read, range(64)))

        self.assertEqual(len(store.list()), 64)

    def test_file_system_training_job_store_handles_concurrent_mutation(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "jobs"
            store = FileSystemTrainingJobStore(root)

            def save_and_read(index: int) -> None:
                job = training_job_record(f"job-{index}", root)
                store.save(job)
                self.assertIsNotNone(store.get(job.id))
                self.assertGreaterEqual(len(store.list()), 1)

            with ThreadPoolExecutor(max_workers=8) as executor:
                list(executor.map(save_and_read, range(64)))

            self.assertEqual(len(store.list()), 64)

    def test_training_live_projection_cache_handles_concurrent_reads(self) -> None:
        manager = TrainingJobServiceHarness(job_store=InMemoryTrainingJobStore())
        job = training_job_record("job-1", Path("/tmp/jobs"))
        events = [
            {
                "type": "dataset_started",
                "status": "running",
                "runId": "run-1",
                "dataset": "mnist",
                "preset": "baseline",
                "runIndex": 1,
            },
            {
                "type": "validation",
                "status": "running",
                "runId": "run-1",
                "dataset": "mnist",
                "preset": "baseline",
                "runIndex": 1,
                "epoch": 0,
                "metrics": {"validation/accuracy": 0.5},
            },
            {
                "type": "dataset_completed",
                "status": "completed",
                "runId": "run-1",
                "dataset": "mnist",
                "preset": "baseline",
                "runIndex": 1,
            },
        ]
        snapshots = [
            TrainingProgressSnapshot(
                events=events[:count],
                new_events=events[:count],
                total_count=count,
                reset=count == 0,
            )
            for count in range(len(events) + 1)
        ]

        def project(index: int) -> int:
            snapshot = snapshots[index % len(snapshots)]
            return manager.runtime._live_projection(job, snapshot).event_count

        with ThreadPoolExecutor(max_workers=8) as executor:
            counts = list(executor.map(project, range(80)))

        self.assertTrue(all(0 <= count <= len(events) for count in counts))

    def test_training_progress_store_handles_concurrent_cached_reads(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "jobs"
            job = training_job_record("job-1", root)
            store = TrainingProgressStore()
            for index in range(20):
                store.append_event(
                    job,
                    {
                        "type": "validation",
                        "status": "running",
                        "step": index,
                    },
                )

            def read_count(_: int) -> int:
                return store.read_snapshot(job).total_count

            with ThreadPoolExecutor(max_workers=8) as executor:
                counts = list(executor.map(read_count, range(80)))

            self.assertEqual(counts, [20] * 80)

    def test_log_query_and_monitor_caches_handle_concurrent_clears(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "run-1"
            root.mkdir()
            (root / "events.out.tfevents.invalid").write_bytes(b"invalid")
            query_service = LogRunQueryService(scanner=LogRunScanner())
            monitor_reader = TensorBoardMonitorReader()
            parameter_reader = TensorBoardParameterStatusReader()

            def read_or_clear(index: int) -> None:
                if index % 2 == 0:
                    query_service.read_tags(root)
                    monitor_reader.read(
                        job_id="job-1",
                        node_path="main",
                        dataset="mnist",
                        log_dir=root.as_posix(),
                    )
                    parameter_reader.read(
                        source_id="job-1",
                        preset="baseline",
                        dataset="mnist",
                        log_dir=root.as_posix(),
                    )
                elif index % 4 == 1:
                    query_service.clear_run_caches([root])
                    monitor_reader.clear_roots({root.as_posix()})
                    parameter_reader.clear_roots({root.as_posix()})
                else:
                    query_service.clear_cache()
                    monitor_reader.clear_cache()
                    parameter_reader.clear_cache()

            with ThreadPoolExecutor(max_workers=8) as executor:
                list(executor.map(read_or_clear, range(80)))

            query_service.clear_cache()
            monitor_reader.clear_cache()
            parameter_reader.clear_cache()


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

from viewer.backend.job_store import (
    FileSystemTrainingJobStore,
    InMemoryTrainingJobStore,
    TrainingJobRecord,
)
from viewer.backend.log_runs import LogRunQueryService, LogRunScanner
from viewer.backend.monitor_data import (
    TensorBoardMonitorReader,
    TensorBoardParameterStatusReader,
)
from viewer.backend.training_jobs import TrainingJobManager
from viewer.backend.training_progress_store import (
    TrainingProgressSnapshot,
    TrainingProgressStore,
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
        run_plan={
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
            "summary": {"pending": 1},
        },
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
        manager = TrainingJobManager(job_store=InMemoryTrainingJobStore())
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
            return manager._live_projection(job, snapshot).event_count

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
        query_service = LogRunQueryService(scanner=LogRunScanner())
        monitor_reader = TensorBoardMonitorReader()
        parameter_reader = TensorBoardParameterStatusReader()
        root = Path("/tmp/logs/run-1")
        tags_key = (root.as_posix(), ())
        scalar_key = (root.as_posix(), (), "train/loss", 500, "tail")
        monitor_key = (root.as_posix(), (), "job-1", "main", "mnist")
        parameter_key = (root.as_posix(), (), "job-1", "baseline", "mnist")

        def mutate(index: int) -> None:
            query_service._cache_set(
                query_service._tags_cache,
                tags_key,
                {
                    "scalars": ["train/loss"],
                    "histograms": [],
                    "images": [],
                    "texts": [],
                },
            )
            query_service._cache_set(
                query_service._scalar_cache,
                scalar_key,
                {
                    "points": [{"step": index, "wallTime": 0.0, "value": 1.0}],
                    "sourcePointCount": 1,
                    "truncated": False,
                },
            )
            monitor_reader._cache_set(
                monitor_key,
                {
                    "jobId": "job-1",
                    "nodePath": "main",
                    "dataset": "mnist",
                    "logDir": root.as_posix(),
                    "scalarSeries": [],
                    "histograms": [],
                    "images": [],
                },
            )
            parameter_reader._cache_set(
                parameter_key,
                {
                    "sourceId": "job-1",
                    "preset": "baseline",
                    "dataset": "mnist",
                    "logDir": root.as_posix(),
                    "nodes": [],
                },
            )
            if index % 2 == 0:
                query_service.clear_run_caches([root])
                monitor_reader.clear_roots({root.as_posix()})
                parameter_reader.clear_roots({root.as_posix()})
            else:
                query_service.clear_cache()
                monitor_reader.clear_cache()
                parameter_reader.clear_cache()

        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(mutate, range(80)))

        query_service.clear_cache()
        monitor_reader.clear_cache()
        parameter_reader.clear_cache()


if __name__ == "__main__":
    unittest.main()

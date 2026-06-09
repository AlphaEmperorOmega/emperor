from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from viewer.backend.job_store import (
    FileSystemTrainingJobStore,
    InMemoryTrainingJobStore,
    TrainingJobRecord,
)


def make_record(job_id: str = "job-1") -> TrainingJobRecord:
    return TrainingJobRecord(
        id=job_id,
        model="linears/linear",
        preset="baseline",
        presets=["baseline"],
        datasets=["Mnist"],
        overrides={},
        search=None,
        planned_run_count=1,
        run_plan={"runs": [], "summary": {"totalRuns": 0}},
        monitors=[],
        log_folder="test_model",
        command=["python", "-m", "viewer.backend.training_worker"],
        root=Path("/tmp/emperor-viewer-training") / job_id,
        pid=1234,
    )


class InMemoryTrainingJobStoreTests(unittest.TestCase):
    def test_save_get_and_list_records(self) -> None:
        store = InMemoryTrainingJobStore()
        record = make_record()

        store.save(record)

        self.assertIs(store.get("job-1"), record)
        self.assertEqual(store.list(), [record])

    def test_get_missing_record_returns_none(self) -> None:
        store = InMemoryTrainingJobStore()

        self.assertIsNone(store.get("missing"))

    def test_in_memory_store_has_no_cross_instance_persistence(self) -> None:
        first_store = InMemoryTrainingJobStore()
        second_store = InMemoryTrainingJobStore()
        first_store.save(make_record())

        self.assertIsNone(second_store.get("job-1"))
        self.assertEqual(second_store.list(), [])


class FileSystemTrainingJobStoreTests(unittest.TestCase):
    def test_save_get_and_list_records_across_store_instances(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            first_store = FileSystemTrainingJobStore(root)
            record = make_record("job-1")
            record.root = root / "job-1"

            first_store.save(record)

            metadata_path = record.root / "metadata.json"
            self.assertTrue(metadata_path.is_file())
            self.assertIs(first_store.get("job-1"), record)
            self.assertEqual(first_store.list(), [record])

            second_store = FileSystemTrainingJobStore(root)
            recovered = second_store.get("job-1")

        self.assertIsNotNone(recovered)
        assert recovered is not None
        self.assertEqual(recovered.id, "job-1")
        self.assertEqual(recovered.model, "linears/linear")
        self.assertEqual(recovered.preset, "baseline")
        self.assertEqual(recovered.presets, ["baseline"])
        self.assertEqual(recovered.datasets, ["Mnist"])
        self.assertEqual(recovered.overrides, {})
        self.assertIsNone(recovered.search)
        self.assertEqual(recovered.planned_run_count, 1)
        self.assertEqual(recovered.run_plan, {"runs": [], "summary": {"totalRuns": 0}})
        self.assertEqual(recovered.monitors, [])
        self.assertEqual(recovered.log_folder, "test_model")
        self.assertEqual(
            recovered.command,
            ["python", "-m", "viewer.backend.training_worker"],
        )
        self.assertEqual(recovered.root, root / "job-1")
        self.assertEqual(recovered.payload_path, root / "job-1" / "payload.json")
        self.assertEqual(recovered.progress_path, root / "job-1" / "progress.jsonl")
        self.assertEqual(recovered.log_path, root / "job-1" / "training.log")
        self.assertEqual(recovered.pid, 1234)
        self.assertEqual(recovered.status, "running")
        self.assertIsNone(recovered.exit_code)

    def test_save_updates_existing_metadata_for_new_store_readback(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = FileSystemTrainingJobStore(root)
            record = make_record("job-1")
            record.root = root / "job-1"
            store.save(record)

            record.status = "completed"
            record.exit_code = 0
            record.updated_at = "2026-06-06T12:00:00+00:00"
            store.save(record)

            recovered = FileSystemTrainingJobStore(root).get("job-1")

        self.assertIsNotNone(recovered)
        assert recovered is not None
        self.assertEqual(recovered.status, "completed")
        self.assertEqual(recovered.exit_code, 0)
        self.assertEqual(recovered.updated_at, "2026-06-06T12:00:00+00:00")

    def test_list_reads_all_metadata_records_in_stable_order(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = FileSystemTrainingJobStore(root)
            for job_id in ("job-b", "job-a"):
                record = make_record(job_id)
                record.root = root / job_id
                store.save(record)

            recovered_ids = [
                record.id for record in FileSystemTrainingJobStore(root).list()
            ]

        self.assertEqual(recovered_ids, ["job-a", "job-b"])

    def test_corrupt_metadata_records_are_ignored(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            job_root = root / "job-1"
            job_root.mkdir()
            (job_root / "metadata.json").write_text("{not json", encoding="utf-8")

            store = FileSystemTrainingJobStore(root)

            self.assertIsNone(store.get("job-1"))
            self.assertEqual(store.list(), [])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from emperor_workbench.training_jobs._filesystem_store import (
    METADATA_FIELDS,
    FileSystemTrainingJobStore,
)
from tests.unit.training_jobs._support import make_record


class FileSystemTrainingJobStoreTests(unittest.TestCase):
    def test_metadata_codec_preserves_exact_persisted_key_set(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            record = make_record("job-1")
            record.root = root / record.id
            record.experiment_task = "classification"
            record.cancellation_mode = "strict-cgroup"
            record.worker_pid = 4321
            record.process_group_id = 4321
            record.cgroup_path = "/sys/fs/cgroup/jobs/job-job-1"

            FileSystemTrainingJobStore(root).save(record)
            payload = json.loads(
                (record.root / "metadata.json").read_text(encoding="utf-8")
            )

        self.assertEqual(
            set(payload),
            METADATA_FIELDS,
        )
        self.assertEqual(payload["experiment_task"], "classification")
        self.assertEqual(payload["cancellation_mode"], "strict-cgroup")
        self.assertEqual(payload["worker_pid"], 4321)
        self.assertEqual(payload["process_group_id"], 4321)
        self.assertEqual(
            payload["cgroup_path"],
            "/sys/fs/cgroup/jobs/job-job-1",
        )

    def test_metadata_codec_rejects_retired_task_and_containment_fields(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            record = make_record("job-1")
            record.root = root / record.id
            record.experiment_task = "classification"
            store = FileSystemTrainingJobStore(root)
            store.save(record)
            metadata_path = record.root / "metadata.json"
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            payload["experimentTask"] = payload.pop("experiment_task")
            for key in (
                "cancellation_mode",
                "worker_pid",
                "process_group_id",
                "cgroup_path",
            ):
                payload.pop(key)
            metadata_path.write_text(json.dumps(payload), encoding="utf-8")

            recovered = FileSystemTrainingJobStore(root).get(record.id)

        self.assertIsNone(recovered)

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
        self.assertEqual(recovered.run_plan.runs, [])
        self.assertEqual(recovered.run_plan.summary.total_runs, 0)
        self.assertEqual(recovered.monitors, [])
        self.assertEqual(recovered.log_folder, "test_model")
        self.assertEqual(
            recovered.observed_command,
            ["python", "-m", "emperor_workbench.training_jobs.worker"],
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

    def test_get_rejects_unsafe_job_ids_without_touching_parent_paths(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "jobs"
            outside = Path(tmp) / "outside"
            outside.mkdir()
            (outside / "metadata.json").write_text("{}", encoding="utf-8")

            store = FileSystemTrainingJobStore(root)

            for job_id in ("../outside", "nested/job", r"nested\\job", ""):
                with self.subTest(job_id=job_id):
                    self.assertIsNone(store.get(job_id))

    def test_list_does_not_cache_metadata_with_unsafe_job_id(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "jobs"
            bad_root = root / "bad"
            bad_root.mkdir(parents=True)
            (bad_root / "metadata.json").write_text(
                json.dumps(
                    {
                        "id": "../outside",
                        "modelType": "linears",
                        "model": "linear",
                        "preset": "baseline",
                        "presets": ["baseline"],
                        "datasets": ["Mnist"],
                        "overrides": {},
                        "search": None,
                        "planned_run_count": 1,
                        "run_plan": {"runs": [], "summary": {"totalRuns": 0}},
                        "monitors": [],
                        "log_folder": "test_model",
                        "command": [
                            "python",
                            "-m",
                            "emperor_workbench.training_jobs.worker",
                        ],
                        "root": str(bad_root),
                        "created_at": "2026-06-06T00:00:00+00:00",
                        "updated_at": "2026-06-06T00:00:00+00:00",
                        "status": "running",
                        "pid": 1234,
                        "exit_code": None,
                    }
                ),
                encoding="utf-8",
            )

            store = FileSystemTrainingJobStore(root)

            self.assertEqual(store.list(), [])
            self.assertIsNone(store.get("../outside"))

    def test_list_and_get_reject_outside_root_job_directory_symlink(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            jobs_root = root / "jobs"
            jobs_root.mkdir()
            outside_store = FileSystemTrainingJobStore(root / "outside")
            outside_record = make_record("job-1")
            outside_record.root = outside_store.root / outside_record.id
            outside_store.save(outside_record)
            (jobs_root / "job-1").symlink_to(
                outside_record.root,
                target_is_directory=True,
            )

            store = FileSystemTrainingJobStore(jobs_root)

            self.assertEqual(store.list(), [])
            self.assertIsNone(store.get("job-1"))

    def test_cached_record_is_evicted_after_job_directory_becomes_symlink(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            jobs_root = root / "jobs"
            outside_root = root / "outside" / "job-1"
            record = make_record("job-1")
            record.root = jobs_root / record.id
            store = FileSystemTrainingJobStore(jobs_root)
            store.save(record)
            self.assertIs(store.get(record.id), record)

            outside_root.parent.mkdir()
            record.root.rename(outside_root)
            record.root.symlink_to(outside_root, target_is_directory=True)

            self.assertEqual(store.list(), [])
            self.assertIsNone(store.get(record.id))

            record.root.unlink()
            outside_root.rename(record.root)
            recovered = store.get(record.id)

        self.assertIsNotNone(recovered)
        self.assertIsNot(recovered, record)

    def test_list_and_get_reject_metadata_symlink(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            jobs_root = root / "jobs"
            job_root = jobs_root / "job-1"
            job_root.mkdir(parents=True)
            outside_store = FileSystemTrainingJobStore(root / "outside")
            outside_record = make_record("job-1")
            outside_record.root = outside_store.root / outside_record.id
            outside_store.save(outside_record)
            (job_root / "metadata.json").symlink_to(
                outside_record.root / "metadata.json"
            )

            store = FileSystemTrainingJobStore(jobs_root)

            self.assertEqual(store.list(), [])
            self.assertIsNone(store.get("job-1"))

    def test_cached_record_is_evicted_after_metadata_becomes_symlink(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            jobs_root = root / "jobs"
            outside_metadata = root / "outside" / "metadata.json"
            record = make_record("job-1")
            record.root = jobs_root / record.id
            store = FileSystemTrainingJobStore(jobs_root)
            store.save(record)
            self.assertIs(store.get(record.id), record)

            outside_metadata.parent.mkdir()
            metadata_path = record.root / "metadata.json"
            metadata_path.rename(outside_metadata)
            metadata_path.symlink_to(outside_metadata)

            self.assertIsNone(store.get(record.id))
            self.assertEqual(store.list(), [])

            metadata_path.unlink()
            outside_metadata.rename(metadata_path)
            recovered = store.get(record.id)

        self.assertIsNotNone(recovered)
        self.assertIsNot(recovered, record)

    def test_list_does_not_cache_directory_and_payload_id_mismatch(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            jobs_root = root / "jobs"
            source_store = FileSystemTrainingJobStore(root / "source")
            source_record = make_record("job-b")
            source_record.root = source_store.root / source_record.id
            source_store.save(source_record)
            mismatched_root = jobs_root / "job-a"
            mismatched_root.mkdir(parents=True)
            (mismatched_root / "metadata.json").write_text(
                (source_record.root / "metadata.json").read_text(encoding="utf-8"),
                encoding="utf-8",
            )

            store = FileSystemTrainingJobStore(jobs_root)

            self.assertEqual(store.list(), [])
            self.assertIsNone(store.get("job-a"))
            self.assertIsNone(store.get("job-b"))

    def test_recovered_metadata_rejects_a_noncanonical_recorded_root(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = FileSystemTrainingJobStore(root)
            record = make_record("job-1")
            record.root = root / record.id
            store.save(record)
            metadata_path = record.root / "metadata.json"
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            payload["root"] = "/legacy/training/jobs/job-1"
            metadata_path.write_text(json.dumps(payload), encoding="utf-8")

            recovered = FileSystemTrainingJobStore(root).get(record.id)

        self.assertIsNone(recovered)

    def test_save_rejects_record_root_that_does_not_match_job_id(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            record = make_record("job-1")
            record.root = root / "different-job"
            store = FileSystemTrainingJobStore(root)

            with self.assertRaises(ValueError):
                store.save(record)

    def test_save_rejects_unsafe_job_id(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            record = make_record("../outside")
            record.root = root / "bad"
            store = FileSystemTrainingJobStore(root)

            with self.assertRaises(ValueError):
                store.save(record)


if __name__ == "__main__":
    unittest.main()

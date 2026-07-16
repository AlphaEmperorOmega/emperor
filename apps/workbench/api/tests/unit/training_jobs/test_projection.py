from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import emperor_workbench.training_jobs._snapshot as training_job_projector
from emperor_workbench.training_jobs._progress_projection import (
    TrainingLiveProjectionCache,
)
from emperor_workbench.training_jobs._progress_store import (
    TrainingProgressSnapshot,
    TrainingProgressStore,
)
from emperor_workbench.training_jobs._snapshot import TrainingJobProjector
from tests.support.training_jobs import training_job_payload
from tests.unit.training_jobs._support import make_job


class TrainingLiveProjectionTests(unittest.TestCase):
    def test_live_projection_streams_full_history_without_materializing_it(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            job = make_job(Path(tmp) / "job-1")
            job.root.mkdir(parents=True)
            events = [{"type": "step", "step": index} for index in range(5_000)]
            job.progress_path.write_text(
                "\n".join(json.dumps(event) for event in events) + "\n",
                encoding="utf-8",
            )
            store = TrainingProgressStore()
            projections = TrainingLiveProjectionCache()

            with patch.object(
                store,
                "_read_events_range",
                side_effect=AssertionError("must stream into the reducer"),
            ):
                snapshot = projections.consume_progress(job, store)
                projection = projections.project(job, snapshot)

        self.assertEqual(projection.event_count, 5_000)
        self.assertEqual(len(projection.events_tail), 100)
        self.assertEqual(projection.events_tail[-1]["step"], 4_999)


class TrainingJobProjectorTests(unittest.TestCase):
    def test_projector_combines_record_events_log_tail_and_result_links(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            job = make_job(Path(tmp) / "job-1")
            job.root.mkdir(parents=True)
            job.log_path.write_text(
                "\n".join(f"log {index}" for index in range(82)),
                encoding="utf-8",
            )
            events = [
                {
                    "type": "dataset_completed",
                    "status": "completed",
                    "dataset": "Mnist",
                    "preset": "baseline",
                    "runId": "run-1",
                    "epoch": 2,
                    "metrics": {"train/loss": 0.1},
                    "logDir": "logs/collaborator",
                }
            ]

            projection = TrainingLiveProjectionCache().project(
                job,
                TrainingProgressSnapshot(
                    events=events,
                    new_events=events,
                    total_count=len(events),
                    reset=True,
                ),
            )
            payload = training_job_payload(
                TrainingJobProjector().project_snapshot(
                    job,
                    projection,
                )
            )

        self.assertEqual(payload["id"], "job-1")
        self.assertEqual(payload["currentPreset"], "baseline")
        self.assertEqual(payload["currentDataset"], "Mnist")
        self.assertEqual(payload["metrics"], {"train/loss": 0.1})
        self.assertEqual(payload["logDir"], "logs/collaborator")
        self.assertEqual(
            payload["resultLinks"],
            [
                {
                    "preset": "baseline",
                    "dataset": "Mnist",
                    "logDir": "logs/collaborator",
                }
            ],
        )
        self.assertEqual(payload["logTail"], [f"log {index}" for index in range(2, 82)])

    def test_projector_log_tail_reads_last_lines_across_small_chunks(self) -> None:
        original_chunk_size = training_job_projector.TRAINING_JOB_LOG_TAIL_CHUNK_BYTES
        training_job_projector.TRAINING_JOB_LOG_TAIL_CHUNK_BYTES = 32
        try:
            with tempfile.TemporaryDirectory() as tmp:
                job = make_job(Path(tmp) / "job-1")
                job.root.mkdir(parents=True)
                job.log_path.write_text(
                    "\n".join(f"log {index} {'x' * 40}" for index in range(20)),
                    encoding="utf-8",
                )

                tail = TrainingJobProjector().log_tail(job, line_count=3)
        finally:
            training_job_projector.TRAINING_JOB_LOG_TAIL_CHUNK_BYTES = (
                original_chunk_size
            )

        self.assertEqual(
            tail,
            [f"log {index} {'x' * 40}" for index in range(17, 20)],
        )

    def test_projector_log_tail_bounds_cr_invalid_utf8_and_no_newline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            job = make_job(Path(tmp) / "job-1")
            job.root.mkdir(parents=True)
            job.log_path.write_bytes(
                b"\r".join(f"line {index}".encode() for index in range(100))
                + b"\r"
                + (b"\xff" * (400 * 1024))
            )

            tail = TrainingJobProjector().log_tail_snapshot(job)

        encoded = "\n".join(tail.lines).encode("utf-8")
        self.assertLessEqual(
            len(encoded),
            training_job_projector.TRAINING_JOB_LOG_TAIL_MAX_BYTES,
        )
        self.assertLessEqual(len(tail.lines), 80)
        self.assertTrue(tail.truncated)
        self.assertIn("�", tail.lines[-1])


if __name__ == "__main__":
    unittest.main()

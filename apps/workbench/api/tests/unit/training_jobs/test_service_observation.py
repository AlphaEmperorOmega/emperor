from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from emperor_workbench.training_jobs import TrainingJobFailure
from tests.support.training_jobs import (
    FakeRunner,
)
from tests.unit.training_jobs._support import (
    TrainingJobServiceHarness,
    create_progress_test_job,
)


class TrainingJobObservationTests(unittest.TestCase):
    def test_training_job_missing_progress_jsonl_returns_empty_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager, created_payload, progress_path = create_progress_test_job(
                Path(tmp)
            )
            progress_path.unlink()

            payload = manager.get_job_payload(str(created_payload["id"]))

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
                        json.dumps(["valid JSON, wrong event shape"]),
                        json.dumps("valid JSON scalar"),
                        json.dumps(valid_events[2]),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            payload = manager.get_job_payload(str(created_payload["id"]))

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

    def test_training_job_skips_invalid_utf8_progress_record(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager, created_payload, progress_path = create_progress_test_job(
                Path(tmp)
            )
            progress_path.write_bytes(b'{"type": "job_started"}\n\xff\n')

            payload = manager.get_job_payload(str(created_payload["id"]))

        self.assertEqual(payload["eventCount"], 1)
        self.assertEqual(payload["events"][0]["type"], "job_started")

    def test_training_job_progress_jsonl_read_failure_raises_os_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager, created_payload, progress_path = create_progress_test_job(
                Path(tmp)
            )
            progress_path.unlink()
            progress_path.mkdir()

            with self.assertRaises(IsADirectoryError):
                manager.get_job_payload(str(created_payload["id"]))

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

            payload = manager.get_job_payload(str(created_payload["id"]))
            history = manager.get_job_events_payload(
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

            payload = manager.get_job_payload(str(created_payload["id"]))
            encoded = json.dumps(payload)
            job = manager.jobs[str(created_payload["id"])]
            stats = manager.runtime.progress_store.cache_stats(job)
            with patch.object(
                manager.runtime.progress_store,
                "_decode_event",
                wraps=manager.runtime.progress_store._decode_event,
            ) as decode_event:
                repeated = manager.get_job_payload(str(created_payload["id"]))

        self.assertEqual(payload["eventCount"], 150_000)
        self.assertEqual(payload["eventCounts"], {"step": 150_000})
        self.assertTrue(payload["eventsTruncated"])
        self.assertEqual(len(payload["events"]), 100)
        self.assertLess(len(encoded), 250_000)
        self.assertLessEqual(stats.retained_event_count, 100)
        self.assertEqual(repeated["eventCount"], 150_000)
        decode_event.assert_not_called()

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

            payload = manager.get_job_payload(str(created_payload["id"]))

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
        self.assertTrue(payload["logTailTruncated"])
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

            payload = manager.get_job_payload(str(created_payload["id"]))

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

    def test_training_job_get_unknown_id_raises_inspector_error(self) -> None:
        manager = TrainingJobServiceHarness(runner=FakeRunner())

        with self.assertRaises(TrainingJobFailure) as context:
            manager.get_job_payload("missing")

        self.assertEqual(str(context.exception), "Unknown training job 'missing'.")


if __name__ == "__main__":
    unittest.main()

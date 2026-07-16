from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from emperor_workbench.training_jobs import TrainingJobFailure
from emperor_workbench.training_jobs._progress_store import TrainingProgressStore
from tests.unit.training_jobs._support import make_job


class TrainingProgressStoreTests(unittest.TestCase):
    def test_overlong_new_progress_record_is_never_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            job = make_job(Path(tmp) / "job-1")
            store = TrainingProgressStore(max_record_bytes=64)

            with self.assertRaisesRegex(
                TrainingJobFailure,
                "64 byte record limit",
            ):
                store.append_event(job, {"type": "event", "value": "x" * 128})

            self.assertFalse(job.progress_path.exists())

    def test_overlong_progress_record_is_skipped_without_losing_later_event(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            job = make_job(Path(tmp) / "job-1")
            job.root.mkdir(parents=True)
            job.progress_path.write_bytes(
                b'{"type":"oversized","value":"'
                + b"x" * 128
                + b'"}\n'
                + b'{"type":"completed","status":"completed"}\n'
            )
            store = TrainingProgressStore(max_record_bytes=64)

            snapshot = store.read_summary(job)

        self.assertEqual(snapshot.total_count, 1)
        self.assertEqual(snapshot.events[0]["type"], "completed")

    def test_progress_store_appends_and_reads_jsonl_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            job = make_job(Path(tmp) / "job-1")
            store = TrainingProgressStore()
            job.root.mkdir(parents=True)
            job.progress_path.write_text("bad json\n\n", encoding="utf-8")

            store.append_event(job, {"type": "dataset_started"})
            events = store.read_snapshot(job).events

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["jobId"], "job-1")
        self.assertEqual(events[0]["type"], "dataset_started")
        self.assertIn("timestamp", events[0])

    def test_progress_store_retries_incomplete_final_jsonl_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            job = make_job(Path(tmp) / "job-1")
            store = TrainingProgressStore()
            job.root.mkdir(parents=True)
            partial = '{"type": "dataset_started"'
            job.progress_path.write_text(partial, encoding="utf-8")

            first_read = store.read_snapshot(job)
            with job.progress_path.open("a", encoding="utf-8") as handle:
                handle.write('}\n{"type": "dataset_completed"}\n')
            second_read = store.read_snapshot(job)

        self.assertEqual(first_read.events, [])
        self.assertEqual(
            [event["type"] for event in second_read.new_events],
            ["dataset_started", "dataset_completed"],
        )

    def test_progress_store_ignores_valid_non_object_json_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            job = make_job(Path(tmp) / "job-1")
            store = TrainingProgressStore()
            job.root.mkdir(parents=True)
            job.progress_path.write_text(
                "\n".join(
                    [
                        json.dumps(["not", "an", "event"]),
                        json.dumps("not an event"),
                        json.dumps(7),
                        json.dumps(None),
                        json.dumps({"type": "dataset_started"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            snapshot = store.read_snapshot(job)

        self.assertEqual(snapshot.events, [{"type": "dataset_started"}])

    def test_progress_cursor_is_owned_by_each_reader(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            job = make_job(Path(tmp) / "job-1")
            store = TrainingProgressStore()
            job.root.mkdir(parents=True)
            job.progress_path.write_text(
                json.dumps({"type": "job_started"}) + "\n",
                encoding="utf-8",
            )
            first = store.read_snapshot(job)
            with job.progress_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"type": "dataset_started"}) + "\n")

            consumed_elsewhere = store.read_summary(job)
            caller_delta = store.read_snapshot(job, cursor=first.cursor)

        self.assertEqual(consumed_elsewhere.total_count, 2)
        self.assertEqual(
            caller_delta.new_events,
            [{"type": "dataset_started"}],
        )

    def test_progress_store_bounds_retention_and_streams_history_pages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            job = make_job(Path(tmp) / "job-1")
            store = TrainingProgressStore()
            job.root.mkdir(parents=True)
            events = [{"type": "step", "step": index} for index in range(5_000)]
            job.progress_path.write_text(
                "\n".join(json.dumps(event) for event in events) + "\n",
                encoding="utf-8",
            )

            store.read_summary(job)
            stats = store.cache_stats(job)
            with patch.object(
                store,
                "_decode_event",
                wraps=store._decode_event,
            ) as decode_event:
                page = store.read_page(job, offset=2_450, limit=25)

        self.assertLessEqual(stats.retained_event_count, 100)
        self.assertEqual(stats.total_count, 5_000)
        self.assertEqual(page.total_count, 5_000)
        self.assertEqual(page.events, events[2_450:2_475])
        self.assertLessEqual(decode_event.call_count, 25 + 256)

    def test_progress_summary_retains_terminal_and_monitor_aggregates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            job = make_job(Path(tmp) / "job-1")
            store = TrainingProgressStore()
            job.root.mkdir(parents=True)
            completed = {
                "type": "dataset_completed",
                "status": "completed",
                "preset": "baseline",
                "dataset": "Mnist",
                "logDir": "logs/job-1/baseline/Mnist",
            }
            events = [completed] + [
                {"type": "step", "step": index} for index in range(250)
            ]
            job.progress_path.write_text(
                "\n".join(json.dumps(event) for event in events) + "\n",
                encoding="utf-8",
            )

            summary = store.read_summary(job)

        self.assertNotIn(completed, summary.events)
        self.assertEqual(summary.latest_terminal_event, completed)
        self.assertEqual(summary.monitor_events, [completed])

    def test_progress_store_bounds_cached_job_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TrainingProgressStore(max_cached_jobs=2)
            jobs = []
            for index in range(3):
                job = make_job(Path(tmp) / f"job-{index}")
                job.id = f"job-{index}"
                job.root.mkdir(parents=True)
                job.progress_path.write_text(
                    json.dumps({"type": "job_started"}) + "\n",
                    encoding="utf-8",
                )
                store.read_summary(job)
                jobs.append(job)

        self.assertEqual(store.cached_job_count, 2)
        self.assertEqual(store.cache_stats(jobs[0]).total_count, 0)


if __name__ == "__main__":
    unittest.main()

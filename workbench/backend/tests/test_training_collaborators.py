from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import workbench.backend.training_jobs.snapshot as training_job_projector
from workbench.backend.api.v1.training_mapping import training_job_to_payload
from workbench.backend.training_jobs.errors import TrainingJobFailure
from workbench.backend.training_jobs.launcher import (
    TRAINING_LOGS_ROOT_ENV,
    TrainingWorkerLauncher,
)
from workbench.backend.training_jobs.monitoring import TrainingMonitorLocator
from workbench.backend.training_jobs.progress import (
    TrainingProgressSnapshot,
    TrainingProgressStore,
)
from workbench.backend.training_jobs.projection import TrainingLiveProjectionCache
from workbench.backend.training_jobs.snapshot import TrainingJobProjector
from workbench.backend.training_jobs.store import TrainingJobRecord


class FakeProcess:
    pid = 123

    def poll(self) -> int | None:
        return None

    def terminate(self) -> None:
        return None

    def wait(self, timeout: float | None = None) -> int:
        return -15

    def kill(self) -> None:
        return None


class RecordingRunner:
    def __init__(self) -> None:
        self.process = FakeProcess()
        self.calls: list[dict[str, object]] = []

    def start(self, command, *, cwd, env, log_path):
        self.calls.append(
            {
                "command": command,
                "cwd": cwd,
                "env": env,
                "log_path": log_path,
            }
        )
        Path(log_path).write_text("worker log\n", encoding="utf-8")
        return self.process


def make_job(root: Path) -> TrainingJobRecord:
    return TrainingJobRecord(
        id="job-1",
        model="linears/linear",
        preset="baseline",
        presets=["baseline"],
        datasets=["Mnist"],
        overrides={},
        search=None,
        planned_run_count=1,
        run_plan={
            "model": "linears/linear",
            "preset": "baseline",
            "presets": ["baseline"],
            "datasets": ["Mnist"],
            "overrides": {},
            "search": None,
            "logFolder": "collaborator_test",
            "isRandomSearch": False,
            "runs": [
                {
                    "id": "run-1",
                    "index": 1,
                    "status": "Pending",
                    "preset": "baseline",
                    "dataset": "Mnist",
                    "changes": [],
                    "overrides": {},
                    "command": "train",
                    "totalEpochs": 2,
                    "currentEpoch": 0,
                    "metrics": {},
                    "logDir": None,
                    "error": None,
                    "errorTraceback": None,
                }
            ],
            "summary": {
                "totalRuns": 1,
                "pendingRuns": 1,
                "totalEpochs": 2,
                "remainingEpochs": 2,
            },
        },
        monitors=["linear"],
        log_folder="collaborator_test",
        command=["train"],
        root=root,
        pid=123,
    )


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

    def test_live_projection_streams_full_history_without_materializing_it(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            job = make_job(Path(tmp) / "job-1")
            job.root.mkdir(parents=True)
            events = [
                {"type": "step", "step": index} for index in range(5_000)
            ]
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


class TrainingWorkerLauncherTests(unittest.TestCase):
    def test_launcher_writes_payload_and_starts_worker_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner = RecordingRunner()
            launcher = TrainingWorkerLauncher(cwd=root, runner=runner)
            relative_logs_root = Path("relative-worker-logs")
            launch = launcher.launch(
                job_root=root / "job-1",
                payload={"id": "job-1"},
                logs_root=relative_logs_root,
            )

            payload_path = root / "job-1" / "payload.json"
            progress_path = root / "job-1" / "progress.jsonl"
            log_path = root / "job-1" / "training.log"

            self.assertEqual(json.loads(payload_path.read_text()), {"id": "job-1"})
            self.assertEqual(
                launch.command,
                [
                    sys.executable,
                    "-m",
                    "workbench.backend.training_worker",
                    "--payload",
                    str(payload_path),
                    "--progress",
                    str(progress_path),
                ],
            )
            self.assertIs(launch.process, runner.process)
            self.assertEqual(runner.calls[0]["log_path"], log_path)
            self.assertEqual(
                runner.calls[0]["env"][TRAINING_LOGS_ROOT_ENV],
                str(relative_logs_root.resolve()),
            )


class TrainingMonitorLocatorTests(unittest.TestCase):
    def test_locator_matches_normalized_presets_and_latest_log_dir(self) -> None:
        locator = TrainingMonitorLocator()
        with tempfile.TemporaryDirectory() as tmp:
            job = make_job(Path(tmp) / "job-1")
            job.presets = ["expert_gate"]

            events = [
                {
                    "dataset": "Mnist",
                    "preset": "expert_gate",
                    "logDir": "old",
                },
                {
                    "dataset": "Mnist",
                    "preset": "expert-gate",
                    "logDir": "new",
                },
            ]

            self.assertTrue(locator.preset_in_job(job, "expert-gate"))
            self.assertEqual(
                locator.log_dir_for_monitor_data(
                    events=events,
                    dataset="Mnist",
                    preset="expert_gate",
                ),
                "new",
            )


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
            payload = training_job_to_payload(
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

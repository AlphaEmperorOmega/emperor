from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import workbench.backend.training_jobs.snapshot as training_job_projector
from workbench.backend.api.v1.training_mapping import training_job_to_payload
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
                summarize=lambda runs: {"totalRuns": len(runs)},
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


if __name__ == "__main__":
    unittest.main()

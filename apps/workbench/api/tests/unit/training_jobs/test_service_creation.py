from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from emperor_workbench.training_jobs import TrainingJobFailure
from emperor_workbench.training_jobs._containment._launcher import (
    TrainingWorkerLauncher,
)
from emperor_workbench.training_jobs._memory_store import InMemoryTrainingJobStore
from tests.support.training_jobs import (
    FakeProcess,
    FakeRunner,
)
from tests.unit.training_jobs._support import (
    FailingTrainingJobStore,
    FakeCgroup,
    TrainingJobServiceHarness,
)


class TrainingJobCreationTests(unittest.TestCase):
    def test_create_job_reaps_worker_if_registration_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            process = FakeProcess()
            manager = TrainingJobServiceHarness(
                root=root / "jobs",
                logs_root=root / "logs",
                runner=FakeRunner(process),
                job_store=FailingTrainingJobStore(),
            )

            with self.assertRaises(RuntimeError):
                manager.create_job_payload(
                    model="linears/linear",
                    preset="baseline",
                    datasets=["Mnist"],
                    overrides={},
                    log_folder="registration_failure",
                    monitors=[],
                )

        self.assertTrue(process.terminated)
        self.assertFalse(process.killed)
        self.assertEqual(manager.runtime._processes, {})

    def test_create_job_payload_write_failure_has_no_runtime_ownership(
        self,
    ) -> None:
        class CountingCgroupManager:
            def __init__(self) -> None:
                self.create_count = 0

            def create_job_cgroup(self, job_id: str):
                self.create_count += 1
                return FakeCgroup()

            def from_job_id(self, job_id: str):
                return None

        class FailingPayloadLauncher(TrainingWorkerLauncher):
            def write_payload(
                self,
                job_root: Path,
                payload: dict,
            ) -> Path:
                raise OSError("payload write failed")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner = FakeRunner()
            cgroups = CountingCgroupManager()
            manager = TrainingJobServiceHarness(
                root=root / "jobs",
                logs_root=root / "logs",
                worker_launcher=FailingPayloadLauncher(
                    cwd=Path.cwd(),
                    runner=runner,
                    cancellation_mode="strict-cgroup",
                    cgroup_manager=cgroups,
                ),
            )

            with self.assertRaisesRegex(OSError, "payload write failed"):
                manager.create_job_payload(
                    model="linears/linear",
                    preset="baseline",
                    datasets=["Mnist"],
                    overrides={},
                    log_folder="payload_write_failure",
                    monitors=[],
                )

            self.assertEqual(runner.commands, [])
            self.assertEqual(cgroups.create_count, 0)
            self.assertEqual(manager.jobs, {})
            self.assertEqual(manager.runtime._processes, {})
            self.assertEqual(list((root / "jobs").glob("*/payload.json")), [])
            self.assertEqual(list((root / "jobs").glob("*/metadata.json")), [])

    def test_create_job_runner_start_failure_cleans_precreated_cgroup(
        self,
    ) -> None:
        class EmptyCountingCgroup(FakeCgroup):
            def __init__(self) -> None:
                super().__init__()
                self.processes = False
                self.cleanup_count = 0

            def cleanup_empty(self) -> None:
                self.cleanup_count += 1
                super().cleanup_empty()

        class CountingCgroupManager:
            def __init__(self, cgroup: EmptyCountingCgroup) -> None:
                self.cgroup = cgroup
                self.create_count = 0

            def create_job_cgroup(self, job_id: str):
                self.create_count += 1
                return self.cgroup

            def from_job_id(self, job_id: str):
                return None

        class FailingRunner:
            def __init__(self) -> None:
                self.start_count = 0

            def start(self, command, *, cwd, env, log_path):
                self.start_count += 1
                raise RuntimeError("runner start failed")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner = FailingRunner()
            cgroup = EmptyCountingCgroup()
            cgroups = CountingCgroupManager(cgroup)
            manager = TrainingJobServiceHarness(
                root=root / "jobs",
                logs_root=root / "logs",
                worker_launcher=TrainingWorkerLauncher(
                    cwd=Path.cwd(),
                    runner=runner,
                    cancellation_mode="strict-cgroup",
                    cgroup_manager=cgroups,
                ),
            )

            with self.assertRaisesRegex(RuntimeError, "runner start failed"):
                manager.create_job_payload(
                    model="linears/linear",
                    preset="baseline",
                    datasets=["Mnist"],
                    overrides={},
                    log_folder="runner_start_failure",
                    monitors=[],
                )

            self.assertEqual(runner.start_count, 1)
            self.assertEqual(cgroups.create_count, 1)
            self.assertEqual(cgroup.cleanup_count, 1)
            self.assertTrue(cgroup.cleaned)
            self.assertEqual(manager.jobs, {})
            self.assertEqual(manager.runtime._processes, {})
            self.assertEqual(list((root / "jobs").glob("*/metadata.json")), [])

    def test_training_job_creation_uses_fake_process_runner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runner = FakeRunner()
            logs_root = Path(tmp) / "logs"
            manager = TrainingJobServiceHarness(
                root=Path(tmp), logs_root=logs_root, runner=runner
            )

            payload = manager.create_job_payload(
                model="linears/linear",
                preset="baseline",
                datasets=["Mnist"],
                overrides={"hidden_dim": "128"},
                log_folder="test_model",
                monitors=["linear"],
            )
            payload_path = Path(tmp) / payload["id"] / "payload.json"
            worker_payload = json.loads(payload_path.read_text())

            self.assertEqual(payload["status"], "running")
            self.assertEqual(payload["preset"], "baseline")
            self.assertEqual(payload["presets"], ["baseline"])
            self.assertEqual(payload["datasets"], ["Mnist"])
            self.assertEqual(payload["monitors"], ["linear"])
            self.assertEqual(payload["logFolder"], "test_model")
            self.assertEqual(worker_payload["monitors"], ["linear"])
            self.assertEqual(
                set(worker_payload),
                {
                    "id",
                    "plannedRunCount",
                    "runPlan",
                    "monitors",
                },
            )
            self.assertEqual(
                set(worker_payload["runPlan"]),
                {
                    "modelType",
                    "model",
                    "preset",
                    "presets",
                    "experimentTask",
                    "datasets",
                    "overrides",
                    "search",
                    "logFolder",
                    "isRandomSearch",
                    "runs",
                    "summary",
                    "snapshotRevisions",
                },
            )
            self.assertEqual(worker_payload["runPlan"], payload["runPlan"])
            self.assertEqual(payload["pid"], 1234)
            self.assertTrue((logs_root / "test_model").is_dir())
            self.assertTrue(runner.commands)
            self.assertIn("emperor_workbench.training_jobs.worker", runner.commands[0])

    def test_training_job_manager_saves_created_job_to_injected_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            process = FakeProcess()
            store = InMemoryTrainingJobStore()
            manager = TrainingJobServiceHarness(
                root=root / "jobs",
                logs_root=root / "logs",
                runner=FakeRunner(process),
                job_store=store,
            )

            payload = manager.create_job_payload(
                model="linears/linear",
                preset="baseline",
                datasets=["Mnist"],
                overrides={},
                log_folder="stored_job",
                monitors=[],
            )
            record = store.get(str(payload["id"]))
            self.assertIsNotNone(record)
            assert record is not None
            self.assertEqual(record.model, "linears/linear")
            self.assertEqual(record.preset, "baseline")
            self.assertEqual(record.datasets, ["Mnist"])
            self.assertEqual(record.log_folder, "stored_job")
            self.assertEqual(record.pid, 1234)
            self.assertFalse(hasattr(record, "process"))
            self.assertIs(manager.jobs[str(payload["id"])], record)

            cancelled = manager.cancel_job_payload(str(payload["id"]))

            self.assertTrue(process.terminated)
            self.assertEqual(cancelled["status"], "cancelled")
            self.assertEqual(record.status, "cancelled")
            self.assertEqual(store.get(str(payload["id"])).status, "cancelled")

    def test_training_job_accepts_multiple_presets_and_multiplies_run_count(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = TrainingJobServiceHarness(
                root=Path(tmp) / "jobs",
                logs_root=Path(tmp) / "logs",
                runner=FakeRunner(),
            )

            payload = manager.create_job_payload(
                model="linears/linear",
                preset="baseline",
                presets=["baseline", "gating", "baseline"],
                datasets=["Mnist", "Cifar10"],
                overrides={},
                log_folder="multi_preset",
            )
            payload_path = Path(tmp) / "jobs" / payload["id"] / "payload.json"
            worker_payload = json.loads(payload_path.read_text())

        self.assertEqual(payload["preset"], "baseline")
        self.assertEqual(payload["presets"], ["baseline", "gating"])
        self.assertEqual(payload["plannedRunCount"], 4)
        self.assertEqual(
            worker_payload["runPlan"]["presets"],
            ["baseline", "gating"],
        )

    def test_training_job_rejects_unknown_selected_preset(self) -> None:
        manager = TrainingJobServiceHarness(runner=FakeRunner())

        with self.assertRaises(TrainingJobFailure):
            manager.create_job_payload(
                model="linears/linear",
                preset="baseline",
                presets=["baseline", "missing-preset"],
                datasets=["Mnist"],
                overrides={},
                log_folder="test_model",
            )

    def test_training_job_rejects_path_like_dataset_input_before_side_effects(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner = FakeRunner()
            manager = TrainingJobServiceHarness(
                root=root / "jobs",
                logs_root=root / "logs",
                runner=runner,
            )

            with self.assertRaises(TrainingJobFailure) as context:
                manager.create_job_payload(
                    model="linears/linear",
                    preset="baseline",
                    datasets=["./Mnist"],
                    overrides={},
                    log_folder="path_like_dataset",
                )

            self.assertEqual(manager.jobs, {})
            self.assertEqual(manager.active_job_payloads(), [])
            self.assertEqual(runner.commands, [])
            self.assertEqual(list((root / "jobs").iterdir()), [])
            self.assertFalse((root / "logs" / "path_like_dataset").exists())

        message = str(context.exception)
        self.assertIn("./Mnist", message)
        self.assertIn("filesystem path", message)
        self.assertIn("server-known dataset name", message)

    def test_training_job_rejects_symlink_top_level_log_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs_root = root / "logs"
            logs_root.mkdir()
            outside_logs = root / "outside_logs"
            outside_logs.mkdir()
            logs_root.joinpath("linked").symlink_to(
                outside_logs,
                target_is_directory=True,
            )
            runner = FakeRunner()
            manager = TrainingJobServiceHarness(
                root=root / "jobs",
                logs_root=logs_root,
                runner=runner,
            )

            with self.assertRaises(TrainingJobFailure) as context:
                manager.create_job_payload(
                    model="linears/linear",
                    preset="baseline",
                    datasets=["Mnist"],
                    overrides={},
                    log_folder="linked",
                    monitors=[],
                )

        self.assertEqual(
            str(context.exception),
            "Refusing to write symlink log experiment: linked",
        )
        self.assertEqual(runner.commands, [])
        self.assertEqual(runner.log_paths, [])
        self.assertEqual(manager.jobs, {})

    def test_training_job_rejects_invalid_search_requests(self) -> None:
        manager = TrainingJobServiceHarness(runner=FakeRunner())

        invalid_searches = [
            {"mode": "grid", "values": {"missing_axis": [1]}},
            {"mode": "grid", "values": {"hidden_dim": []}},
            {
                "mode": "random",
                "values": {"hidden_dim": [64]},
                "randomSamples": 0,
            },
            {"mode": "grid", "values": {"hidden_dim": [999]}},
            {"mode": "grid", "values": {}},
        ]

        for search in invalid_searches:
            with self.subTest(search=search):
                with self.assertRaises(TrainingJobFailure):
                    manager.create_job_payload(
                        model="linears/linear",
                        preset="baseline",
                        datasets=["Mnist"],
                        overrides={},
                        search=search,
                        log_folder="invalid_search",
                    )

    def test_training_job_rejects_locked_search_axis(self) -> None:
        manager = TrainingJobServiceHarness(runner=FakeRunner())

        with self.assertRaises(TrainingJobFailure) as context:
            manager.create_job_payload(
                model="linears/linear",
                preset="baseline",
                presets=["baseline", "post-norm"],
                datasets=["Mnist"],
                overrides={},
                search={
                    "mode": "grid",
                    "values": {"stack_layer_norm_position": ["BEFORE", "AFTER"]},
                },
                log_folder="locked_search",
            )

        self.assertIn("locked", str(context.exception))

    def test_training_job_rejects_invalid_log_folders(self) -> None:
        manager = TrainingJobServiceHarness(runner=FakeRunner())

        for log_folder in (
            "",
            "my experiment",
            "my-experiment",
            "my.folder",
            "my/folder",
            "_my_folder",
            "my_folder_",
            "my__folder",
        ):
            with self.subTest(log_folder=log_folder):
                with self.assertRaises(TrainingJobFailure):
                    manager.create_job_payload(
                        model="linears/linear",
                        preset="baseline",
                        datasets=["Mnist"],
                        overrides={},
                        log_folder=log_folder,
                    )

    def test_training_job_rejects_unknown_monitor(self) -> None:
        manager = TrainingJobServiceHarness(runner=FakeRunner())

        with self.assertRaises(TrainingJobFailure):
            manager.create_job_payload(
                model="linears/linear",
                preset="baseline",
                datasets=["Mnist"],
                overrides={},
                log_folder="test_model",
                monitors=["sampler"],
            )

    def test_training_job_rejects_locked_overrides(self) -> None:
        manager = TrainingJobServiceHarness(runner=FakeRunner())

        with self.assertRaises(TrainingJobFailure):
            manager.create_job_payload(
                model="linears/linear",
                preset="baseline",
                presets=["baseline", "gating"],
                datasets=["Mnist"],
                overrides={"gate_flag": "false"},
                log_folder="test_model",
            )


if __name__ == "__main__":
    unittest.main()

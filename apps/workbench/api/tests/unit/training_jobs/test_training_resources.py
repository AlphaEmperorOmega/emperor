from __future__ import annotations

import os
import stat
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from emperor_workbench.failures import FailureKind
from emperor_workbench.training_jobs import TrainingJobFailure, TrainingResourceLimits
from emperor_workbench.training_jobs._containment._cgroup_v2 import (
    CgroupV2Manager,
)
from emperor_workbench.training_jobs._containment._launcher import (
    TrainingWorkerLauncher,
)
from tests.support.training_jobs import (
    FakeRunner,
)
from tests.unit.training_jobs._support import TrainingJobServiceHarness


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _create_job(
    harness: TrainingJobServiceHarness,
    *,
    log_folder: str,
) -> dict[str, object]:
    return harness.create_job_payload(
        model="linears/linear",
        preset="baseline",
        datasets=["Mnist"],
        overrides={},
        log_folder=log_folder,
        monitors=[],
    )


class TrainingWorkerEnvironmentTests(unittest.TestCase):
    def test_worker_environment_allowlists_runtime_values_and_excludes_secrets(
        self,
    ) -> None:
        environment = {
            "PATH": "/runtime/bin",
            "VIRTUAL_ENV": "/runtime/venv",
            "LANG": "en_US.UTF-8",
            "CUDA_VISIBLE_DEVICES": "0",
            "NVIDIA_VISIBLE_DEVICES": "all",
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "WORKBENCH_API_TOKEN": "backend-secret",
            "AWS_SECRET_ACCESS_KEY": "cloud-secret",
            "TRAINING_SENTINEL_SECRET": "must-not-leak",
        }
        with patch.dict(os.environ, environment, clear=True):
            worker_environment = TrainingWorkerLauncher(
                cwd=Path.cwd(),
                runner=FakeRunner(),
            ).worker_env()

        self.assertEqual(worker_environment["PATH"], "/runtime/bin")
        self.assertEqual(worker_environment["VIRTUAL_ENV"], "/runtime/venv")
        self.assertEqual(worker_environment["CUDA_VISIBLE_DEVICES"], "0")
        self.assertEqual(worker_environment["OMP_NUM_THREADS"], "4")
        self.assertEqual(worker_environment["MKL_NUM_THREADS"], "4")
        self.assertNotIn("WORKBENCH_API_TOKEN", worker_environment)
        self.assertNotIn("AWS_SECRET_ACCESS_KEY", worker_environment)
        self.assertNotIn("TRAINING_SENTINEL_SECRET", worker_environment)


class TrainingCgroupResourceTests(unittest.TestCase):
    def test_job_cgroup_receives_balanced_memory_cpu_and_process_limits(self) -> None:
        with TemporaryDirectory() as tmp:
            mount = Path(tmp) / "cgroup"
            base = mount / "delegated"
            base.mkdir(parents=True)
            mount.joinpath("cgroup.controllers").write_text(
                "cpu memory pids",
                encoding="utf-8",
            )
            limits = TrainingResourceLimits(
                memory_bytes=16 * 1024**3,
                cpu_count=8,
                process_count=512,
            )
            with (
                patch(
                    "emperor_workbench.training_jobs._containment._cgroup_v2.CGROUP_V2_MOUNT",
                    mount,
                ),
                patch(
                    "emperor_workbench.training_jobs._containment._cgroup_v2._is_linux",
                    return_value=True,
                ),
            ):
                job = CgroupV2Manager(
                    base_path=base,
                    namespace="training",
                    resource_limits=limits,
                ).create_job_cgroup("job-1")

            self.assertEqual(
                job.path.joinpath("memory.max").read_text(encoding="utf-8"),
                str(16 * 1024**3),
            )
            self.assertEqual(
                job.path.joinpath("cpu.max").read_text(encoding="utf-8"),
                "800000 100000",
            )
            self.assertEqual(
                job.path.joinpath("pids.max").read_text(encoding="utf-8"),
                "512",
            )


class TrainingJobAdmissionAndModeTests(unittest.TestCase):
    def test_third_active_training_job_is_rejected_before_launch(self) -> None:
        with TemporaryDirectory() as tmp:
            runner = FakeRunner()
            harness = TrainingJobServiceHarness(
                root=Path(tmp) / "jobs",
                logs_root=Path(tmp) / "logs",
                runner=runner,
                max_active_training_jobs=2,
            )

            _create_job(harness, log_folder="quota_one")
            _create_job(harness, log_folder="quota_two")
            with self.assertRaises(TrainingJobFailure) as raised:
                _create_job(harness, log_folder="quota_three")

            self.assertEqual(raised.exception.kind, FailureKind.UNAVAILABLE)
            self.assertIn("2 active Training Jobs", raised.exception.detail)
            self.assertEqual(len(runner.commands), 2)
            self.assertFalse((Path(tmp) / "logs" / "quota_three").exists())

    def test_training_state_and_job_files_are_private_under_hostile_umask(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "jobs"
            root.mkdir(mode=0o777)
            root.chmod(0o777)
            previous_umask = os.umask(0)
            try:
                harness = TrainingJobServiceHarness(
                    root=root,
                    logs_root=Path(tmp) / "logs",
                    runner=FakeRunner(),
                )
                payload = _create_job(harness, log_folder="private_modes")
            finally:
                os.umask(previous_umask)

            job = harness.jobs[str(payload["id"])]
            self.assertEqual(_mode(root), 0o700)
            self.assertEqual(_mode(job.root), 0o700)
            for private_file in (
                job.payload_path,
                job.progress_path,
                job.log_path,
                job.root / "metadata.json",
            ):
                with self.subTest(path=private_file.name):
                    self.assertTrue(private_file.is_file())
                    self.assertEqual(_mode(private_file), 0o600)


if __name__ == "__main__":
    unittest.main()

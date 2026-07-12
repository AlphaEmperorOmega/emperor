from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from workbench.backend.api import WorkbenchApiSettings, create_app
from workbench.backend.training_jobs.cgroups import (
    CgroupV2Manager,
    StrictCancellationUnavailable,
)
from workbench.backend.training_jobs.launcher import TrainingWorkerLauncher


class TrainingCancellationCapabilityTests(unittest.TestCase):
    def test_process_group_app_construction_does_not_construct_cgroups(
        self,
    ) -> None:
        with patch.object(
            CgroupV2Manager,
            "__init__",
            side_effect=StrictCancellationUnavailable("missing /proc"),
        ) as construct_cgroups:
            app = create_app(
                WorkbenchApiSettings(
                    training_cancellation_mode="process-group",
                )
            )
            capability = (
                app.state.workbench_services.training_jobs
                .cancellation_capability()
            )

        construct_cgroups.assert_not_called()
        self.assertEqual(capability, "process-group")

    def test_strict_cgroup_app_construction_defers_unavailable_probe(
        self,
    ) -> None:
        with patch.object(
            CgroupV2Manager,
            "__init__",
            side_effect=StrictCancellationUnavailable("missing /proc"),
        ) as construct_cgroups:
            app = create_app(
                WorkbenchApiSettings(
                    training_cancellation_mode="strict-cgroup",
                )
            )

            construct_cgroups.assert_not_called()
            capability = (
                app.state.workbench_services.training_jobs
                .cancellation_capability()
            )

        construct_cgroups.assert_called_once_with()
        self.assertEqual(capability, "unsupported")

    def test_process_group_capability_is_total_on_non_posix_hosts(self) -> None:
        cwd = Path.cwd()
        with patch("workbench.backend.training_jobs.launcher.os.name", "nt"):
            launcher = TrainingWorkerLauncher(
                cwd=cwd,
                cancellation_mode="process-group",
            )

            capability = launcher.cancellation_capability()

        self.assertEqual(capability, "unsupported")

    def test_strict_capability_is_total_when_proc_is_missing(self) -> None:
        with TemporaryDirectory() as tmp:
            mount = Path(tmp)
            mount.joinpath("cgroup.controllers").touch()
            with (
                patch(
                    "workbench.backend.training_jobs.cgroups.CGROUP_V2_MOUNT",
                    mount,
                ),
                patch(
                    "workbench.backend.training_jobs.cgroups._is_linux",
                    return_value=True,
                ),
                patch.object(
                    Path,
                    "read_text",
                    side_effect=FileNotFoundError("missing /proc/self/cgroup"),
                ),
            ):
                manager = CgroupV2Manager()

                capability = manager.is_available()

        self.assertFalse(capability)

    def test_strict_capability_does_not_probe_proc_on_cgroup_v1(self) -> None:
        with TemporaryDirectory() as tmp:
            with (
                patch(
                    "workbench.backend.training_jobs.cgroups.CGROUP_V2_MOUNT",
                    Path(tmp),
                ),
                patch(
                    "workbench.backend.training_jobs.cgroups._is_linux",
                    return_value=True,
                ),
                patch(
                    "workbench.backend.training_jobs.cgroups."
                    "_current_cgroup_relative_path",
                    side_effect=AssertionError("must not inspect /proc"),
                ) as inspect_proc,
            ):
                manager = CgroupV2Manager()

                capability = manager.is_available()

        self.assertFalse(capability)
        inspect_proc.assert_not_called()

    def test_strict_capability_is_total_on_non_linux_hosts(self) -> None:
        with patch(
            "workbench.backend.training_jobs.cgroups._is_linux",
            return_value=False,
        ):
            manager = CgroupV2Manager()

            capability = manager.is_available()

        self.assertFalse(capability)


class TrainingCgroupRecoveryTests(unittest.TestCase):
    def test_recovery_derives_canonical_cgroup_from_validated_job_id(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            canonical = root / "training" / "job-job-1"
            canonical.mkdir(parents=True)
            manager = CgroupV2Manager(base_path=root, namespace="training")

            recovered = manager.from_job_id("job-1")

        self.assertIsNotNone(recovered)
        assert recovered is not None
        self.assertEqual(recovered.path, canonical)

    def test_recovery_rejects_unsafe_job_ids_and_canonical_symlink(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            namespace = root / "training"
            outside = root / "outside"
            namespace.mkdir()
            outside.mkdir()
            (namespace / "job-linked").symlink_to(
                outside,
                target_is_directory=True,
            )
            manager = CgroupV2Manager(base_path=root, namespace="training")

            for job_id in ("../outside", "nested/job", r"nested\job", ""):
                with self.subTest(job_id=job_id):
                    self.assertIsNone(manager.from_job_id(job_id))
            self.assertIsNone(manager.from_job_id("linked"))

    def test_recovery_rejects_symlinked_namespace(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            outside = root / "outside"
            outside.joinpath("job-job-1").mkdir(parents=True)
            (root / "training").symlink_to(outside, target_is_directory=True)
            manager = CgroupV2Manager(base_path=root, namespace="training")

            self.assertIsNone(manager.from_job_id("job-1"))


if __name__ == "__main__":
    unittest.main()

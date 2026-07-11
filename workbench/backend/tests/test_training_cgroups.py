from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from workbench.backend.training_jobs.cgroups import CgroupV2Manager


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

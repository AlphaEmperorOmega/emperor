from __future__ import annotations

import tomllib
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DISTRIBUTION_ROOTS = (
    PROJECT_ROOT,
    PROJECT_ROOT / "apps" / "workbench" / "api",
)
EXPECTED_LICENSE = "CC-BY-NC-4.0"
EXPECTED_LICENSE_FILES = ["LICENSE", "NOTICE"]
MINIMUM_BUILD_REQUIREMENT = "setuptools>=77.0.0"


def _project_config(distribution_root: Path) -> dict:
    with (distribution_root / "pyproject.toml").open("rb") as project_file:
        return tomllib.load(project_file)


class DistributionLicenseMetadataTests(unittest.TestCase):
    def test_distributions_publish_the_spdx_license_expression(self) -> None:
        for distribution_root in DISTRIBUTION_ROOTS:
            with self.subTest(distribution_root=distribution_root):
                project = _project_config(distribution_root)["project"]

                self.assertEqual(project["license"], EXPECTED_LICENSE)
                self.assertEqual(
                    project["license-files"],
                    EXPECTED_LICENSE_FILES,
                )

    def test_distributions_require_pep_639_build_support(self) -> None:
        for distribution_root in DISTRIBUTION_ROOTS:
            with self.subTest(distribution_root=distribution_root):
                build_system = _project_config(distribution_root)["build-system"]

                self.assertIn(
                    MINIMUM_BUILD_REQUIREMENT,
                    build_system["requires"],
                )

    def test_declared_legal_files_exist_in_each_distribution_root(self) -> None:
        root_license = (PROJECT_ROOT / "LICENSE").read_bytes()
        for distribution_root in DISTRIBUTION_ROOTS:
            with self.subTest(distribution_root=distribution_root):
                for file_name in EXPECTED_LICENSE_FILES:
                    self.assertTrue((distribution_root / file_name).is_file())
                self.assertEqual(
                    (distribution_root / "LICENSE").read_bytes(),
                    root_license,
                )


if __name__ == "__main__":
    unittest.main()

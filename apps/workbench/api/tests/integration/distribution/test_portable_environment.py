from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.support import REPOSITORY_ROOT

PROJECT_ROOT = REPOSITORY_ROOT


def _load_emperor_dev():
    name = "_emperor_dev_portability_tests"
    spec = importlib.util.spec_from_file_location(
        name,
        PROJECT_ROOT / "tools" / "emperor_dev.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the portable launcher for testing.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


emperor_dev = _load_emperor_dev()


class PortableEnvironmentTests(unittest.TestCase):
    def test_supported_platforms_select_their_profile_lock(self) -> None:
        cases = (
            ("linux", "x86_64", "cpu"),
            ("linux", "x86_64", "cuda"),
            ("windows", "x86_64", "cpu"),
            ("macos", "arm64", "cpu"),
        )
        for os_name, architecture, profile in cases:
            with self.subTest(platform=f"{os_name}-{architecture}-{profile}"):
                path = emperor_dev.constraint_path(
                    profile,
                    emperor_dev.PlatformSpec(os_name, architecture),
                )
                self.assertTrue(path.is_file())
                self.assertEqual(
                    path.name,
                    f"python-3.13-{os_name}-{architecture}-{profile}.txt",
                )

    def test_intel_macos_is_rejected_instead_of_claiming_an_unavailable_wheel(
        self,
    ) -> None:
        with (
            patch.object(emperor_dev.platform, "system", return_value="Darwin"),
            patch.object(emperor_dev.platform, "machine", return_value="x86_64"),
            self.assertRaisesRegex(SystemExit, "macOS arm64"),
        ):
            emperor_dev.detect_platform()

    def test_cuda_rejection_names_the_supported_command(self) -> None:
        with self.assertRaisesRegex(SystemExit, "mise run setup --profile cpu"):
            emperor_dev.constraint_path(
                "cuda",
                emperor_dev.PlatformSpec("windows", "x86_64"),
            )

    def test_virtualenv_layout_is_platform_aware(self) -> None:
        root = Path("environment")
        self.assertEqual(
            emperor_dev.venv_python(root, os_name="windows"),
            root / "Scripts" / "python.exe",
        )
        self.assertEqual(
            emperor_dev.venv_python(root, os_name="macos"),
            root / "bin" / "python",
        )

    def test_setup_signature_tracks_recursively_included_constraints(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            constraints = root / "constraints"
            constraints.mkdir()
            project = root / "pyproject.toml"
            project.write_text("first", encoding="utf-8")
            base = constraints / "base.txt"
            base.write_text("torch==1\n", encoding="utf-8")
            lock = constraints / "platform.txt"
            lock.write_text("-c base.txt\n", encoding="utf-8")
            spec = emperor_dev.PlatformSpec("linux", "x86_64")
            with (
                patch.object(emperor_dev, "REPOSITORY_ROOT", root),
                patch.object(emperor_dev, "SETUP_INPUTS", (project,)),
            ):
                before = emperor_dev._setup_signature("cpu", spec, lock)
                base.write_text("torch==2\n", encoding="utf-8")
                after = emperor_dev._setup_signature("cpu", spec, lock)

        self.assertNotEqual(before, after)

    def test_nested_constraint_versions_are_available_to_cpu_bootstrap(self) -> None:
        lock = emperor_dev.constraint_path(
            "cpu",
            emperor_dev.PlatformSpec("linux", "x86_64"),
        )
        self.assertEqual(emperor_dev._constraint_version(lock, "torch"), "2.12.0+cpu")

    def test_cpu_profile_rejects_a_cuda_enabled_torch_runtime(self) -> None:
        spec = emperor_dev.PlatformSpec("linux", "x86_64")
        self.assertFalse(
            emperor_dev._torch_build_matches_profile(
                "cpu",
                spec,
                torch_version="2.12.0+cu130",
                cuda_version="13.0",
            )
        )
        self.assertTrue(
            emperor_dev._torch_build_matches_profile(
                "cpu",
                spec,
                torch_version="2.12.0+cpu",
                cuda_version=None,
            )
        )

    def test_macos_cpu_profile_accepts_the_native_untagged_build(self) -> None:
        self.assertTrue(
            emperor_dev._torch_build_matches_profile(
                "cpu",
                emperor_dev.PlatformSpec("macos", "arm64"),
                torch_version="2.12.0",
                cuda_version=None,
            )
        )


if __name__ == "__main__":
    unittest.main()

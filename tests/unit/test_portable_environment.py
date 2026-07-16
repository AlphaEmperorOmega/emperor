from __future__ import annotations

import importlib.util
import sys
import tempfile
import tomllib
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER_PATH = PROJECT_ROOT / "tools" / "emperor_dev.py"
LAUNCHER_SPEC = importlib.util.spec_from_file_location(
    "emperor_dev_under_test",
    LAUNCHER_PATH,
)
if LAUNCHER_SPEC is None or LAUNCHER_SPEC.loader is None:
    raise RuntimeError(f"Unable to load portable launcher: {LAUNCHER_PATH}")
emperor_dev = importlib.util.module_from_spec(LAUNCHER_SPEC)
sys.modules[LAUNCHER_SPEC.name] = emperor_dev
LAUNCHER_SPEC.loader.exec_module(emperor_dev)


class PortableEnvironmentProfileTests(unittest.TestCase):
    linux = emperor_dev.PlatformSpec("linux", "x86_64")
    windows = emperor_dev.PlatformSpec("windows", "x86_64")
    macos = emperor_dev.PlatformSpec("macos", "arm64")

    def test_profile_policy_selects_native_lockfiles(self) -> None:
        cases = (
            ("cpu", self.linux, "python-3.13-linux-x86_64-cpu.txt"),
            ("cpu", self.windows, "python-3.13-windows-x86_64-cpu.txt"),
            ("cpu", self.macos, "python-3.13-macos-arm64-cpu.txt"),
            ("cuda", self.linux, "python-3.13-linux-x86_64-cuda.txt"),
            (
                "cuda-legacy",
                self.linux,
                "python-3.13-linux-x86_64-cuda-legacy.txt",
            ),
        )

        for profile, platform_spec, expected_name in cases:
            with self.subTest(profile=profile, platform=platform_spec.key):
                self.assertEqual(
                    emperor_dev.constraint_path(profile, platform_spec).name,
                    expected_name,
                )

    def test_cuda_profiles_reject_windows_and_macos(self) -> None:
        for profile in ("cuda", "cuda-legacy"):
            for platform_spec in (self.windows, self.macos):
                with self.subTest(profile=profile, platform=platform_spec.key):
                    with self.assertRaisesRegex(
                        SystemExit,
                        "only supported on Linux x86_64",
                    ):
                        emperor_dev.constraint_path(profile, platform_spec)

    def test_setup_and_dev_parsers_accept_cuda_legacy(self) -> None:
        for command in ("setup", "dev"):
            with self.subTest(command=command):
                parsed = emperor_dev.parse_args(
                    [command, "--profile", "cuda-legacy"]
                )
                self.assertEqual(parsed.command, command)
                self.assertEqual(parsed.profile, "cuda-legacy")

    def test_mise_setup_and_dev_selectors_accept_cuda_legacy(self) -> None:
        with (PROJECT_ROOT / "mise.toml").open("rb") as mise_file:
            mise = tomllib.load(mise_file)

        for task_name in ("setup", "dev"):
            with self.subTest(task=task_name):
                self.assertIn(
                    'choices "cpu" "cuda" "cuda-legacy"',
                    mise["tasks"][task_name]["usage"],
                )

    def test_env_sh_legacy_profile_shortcut_selects_cuda_legacy(self) -> None:
        env_wrapper = (PROJECT_ROOT / "env.sh").read_text(encoding="utf-8")

        self.assertIn(
            '--legacy-profile) _emperor_profile="cuda-legacy" ;;',
            env_wrapper,
        )
        self.assertIn(
            'mise run dev --profile "$_emperor_profile"',
            env_wrapper,
        )
        self.assertIn(
            "source env.sh [--legacy-profile]",
            env_wrapper,
        )

    def test_cuda_legacy_uses_exact_cu126_wheel_command(self) -> None:
        python = Path("/verified/venv/bin/python")

        command = emperor_dev._pytorch_install_command(
            python,
            "cuda-legacy",
            self.linux,
        )

        self.assertEqual(
            command,
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--only-binary=:all:",
                "--force-reinstall",
                "--no-deps",
                "--index-url",
                "https://download.pytorch.org/whl/cu126",
                "torch==2.12.0+cu126",
                "torchvision==0.27.0+cu126",
            ],
        )

    def test_existing_profiles_keep_explicit_torch_sources(self) -> None:
        python = Path("/verified/venv/bin/python")
        cases = (
            (
                "cpu",
                self.linux,
                "https://download.pytorch.org/whl/cpu",
                ("torch==2.12.0+cpu", "torchvision==0.27.0+cpu"),
            ),
            (
                "cuda",
                self.linux,
                "https://download.pytorch.org/whl/cu130",
                ("torch==2.12.0+cu130", "torchvision==0.27.0+cu130"),
            ),
        )

        for profile, platform_spec, index, requirements in cases:
            with self.subTest(profile=profile):
                command = emperor_dev._pytorch_install_command(
                    python,
                    profile,
                    platform_spec,
                )
                self.assertIsNotNone(command)
                assert command is not None
                self.assertEqual(command[command.index("--index-url") + 1], index)
                self.assertEqual(tuple(command[-2:]), requirements)

        self.assertIsNone(
            emperor_dev._pytorch_install_command(python, "cpu", self.macos)
        )

    def test_cuda_legacy_accepts_cuda_12_6_with_sm_61(self) -> None:
        self.assertTrue(
            emperor_dev._torch_build_matches_profile(
                "cuda-legacy",
                self.linux,
                torch_version="2.12.0+cu126",
                cuda_version="12.6",
                architectures=("sm_61", "sm_70", "sm_75"),
            )
        )

    def test_cuda_legacy_accepts_forward_compatible_sm_60_cubin(self) -> None:
        self.assertTrue(
            emperor_dev._torch_build_matches_profile(
                "cuda-legacy",
                self.linux,
                torch_version="2.12.0+cu126",
                cuda_version="12.6",
                architectures=("sm_50", "sm_60", "sm_70"),
            )
        )
        self.assertFalse(
            emperor_dev._compiled_architectures_support(
                ("sm_61",),
                ("sm_62",),
            )
        )

    def test_cuda_legacy_rejects_incompatible_builds(self) -> None:
        cases = (
            ("2.12.0+cpu", None, (), "CPU build"),
            ("2.12.0+cu130", "13.0", ("sm_61",), "CUDA 13 build"),
            ("2.12.1+cu126", "12.6", ("sm_61",), "wrong Torch version"),
            ("2.12.0+cu126", "12.6", ("sm_70",), "missing sm_61"),
        )

        for torch_version, cuda_version, architectures, label in cases:
            with self.subTest(case=label):
                self.assertFalse(
                    emperor_dev._torch_build_matches_profile(
                        "cuda-legacy",
                        self.linux,
                        torch_version=torch_version,
                        cuda_version=cuda_version,
                        architectures=architectures,
                    )
                )

    def test_existing_cpu_and_cuda_build_validation_remains_exact(self) -> None:
        cases = (
            ("cpu", self.linux, "2.12.0+cpu", None, (), True),
            ("cpu", self.linux, "2.12.0", None, (), False),
            ("cpu", self.macos, "2.12.0", None, (), True),
            ("cuda", self.linux, "2.12.0+cu130", "13.0", (), True),
            ("cuda", self.linux, "2.12.0+cu126", "12.6", (), False),
        )

        for profile, platform_spec, version, cuda, architectures, accepted in cases:
            with self.subTest(profile=profile, version=version, cuda=cuda):
                self.assertEqual(
                    emperor_dev._torch_build_matches_profile(
                        profile,
                        platform_spec,
                        torch_version=version,
                        cuda_version=cuda,
                        architectures=architectures,
                    ),
                    accepted,
                )

    def test_setup_signature_covers_lock_index_and_validation_policy(self) -> None:
        policy = emperor_dev.PROFILE_POLICIES["cuda-legacy"]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            constraints = root / "constraints"
            constraints.mkdir()
            lock = constraints / "legacy.txt"
            lock.write_text("torch==2.12.0+cu126\n", encoding="utf-8")
            setup_input = root / "pyproject.toml"
            setup_input.write_text("[project]\n", encoding="utf-8")

            with (
                patch.object(emperor_dev, "REPOSITORY_ROOT", root),
                patch.object(emperor_dev, "SETUP_INPUTS", (setup_input,)),
            ):
                baseline = emperor_dev._setup_signature(
                    policy,
                    self.linux,
                    lock,
                )

                lock.write_text("torch==2.12.1+cu126\n", encoding="utf-8")
                changed_lock = emperor_dev._setup_signature(
                    policy,
                    self.linux,
                    lock,
                )
                lock.write_text("torch==2.12.0+cu126\n", encoding="utf-8")

                sources = tuple(
                    (
                        platform_key,
                        replace(source, index=f"{source.index}/policy-change"),
                    )
                    for platform_key, source in policy.torch_sources
                )
                changed_index = emperor_dev._setup_signature(
                    replace(policy, torch_sources=sources),
                    self.linux,
                    lock,
                )
                changed_validation = emperor_dev._setup_signature(
                    replace(
                        policy,
                        required_architectures=("sm_61", "sm_70"),
                    ),
                    self.linux,
                    lock,
                )

        self.assertNotEqual(baseline, changed_lock)
        self.assertNotEqual(baseline, changed_index)
        self.assertNotEqual(baseline, changed_validation)

    def test_setup_recreates_owned_venv_when_verified_inputs_change(self) -> None:
        marker = {
            "platform": self.linux.key,
            "profile": "cpu",
            "signature": "current",
        }

        self.assertIsNone(
            emperor_dev._venv_recreation_reason(
                marker=marker,
                profile="cpu",
                platform_key=self.linux.key,
                signature="current",
                version=emperor_dev.SUPPORTED_PYTHON,
            )
        )
        self.assertEqual(
            emperor_dev._venv_recreation_reason(
                marker=marker,
                profile="cpu",
                platform_key=self.linux.key,
                signature="changed",
                version=emperor_dev.SUPPORTED_PYTHON,
            ),
            "the verified setup inputs changed",
        )

    def test_cuda_legacy_lock_is_standalone_sorted_and_cu126_only(self) -> None:
        base_path = PROJECT_ROOT / "constraints" / "python-3.13-linux-x86_64.txt"
        legacy_path = (
            PROJECT_ROOT
            / "constraints"
            / "python-3.13-linux-x86_64-cuda-legacy.txt"
        )
        base_pins = {
            line
            for line in base_path.read_text(encoding="utf-8").splitlines()
            if line
            and not line.startswith("#")
            and not line.casefold().startswith(
                ("cuda-", "nvidia-", "torch==", "torchvision==", "triton==")
            )
        }
        legacy_lines = legacy_path.read_text(encoding="utf-8").splitlines()
        legacy_pins = [
            line for line in legacy_lines if line and not line.startswith("#")
        ]

        self.assertTrue(base_pins.issubset(legacy_pins))
        self.assertFalse(
            any(
                line.startswith(("-c ", "--constraint"))
                for line in legacy_pins
            )
        )
        self.assertEqual(legacy_pins, sorted(legacy_pins))
        self.assertIn("torch==2.12.0+cu126", legacy_pins)
        self.assertIn("torchvision==0.27.0+cu126", legacy_pins)
        self.assertIn("cuda-toolkit==12.6.3", legacy_pins)
        self.assertIn("nvidia-cuda-runtime-cu12==12.6.77", legacy_pins)
        self.assertFalse(any("cu13" in line.casefold() for line in legacy_pins))
        self.assertFalse(any(line == "cuda-toolkit==13.0.2" for line in legacy_pins))


if __name__ == "__main__":
    unittest.main()

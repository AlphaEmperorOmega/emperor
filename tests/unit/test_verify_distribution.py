from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_verify_distribution():
    name = "_emperor_verify_distribution_tests"
    spec = importlib.util.spec_from_file_location(
        name,
        PROJECT_ROOT / "tools" / "verify_distribution.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the distribution verifier.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


verify_distribution = _load_verify_distribution()


def _metadata(
    distribution_name: str,
    runtime_requirements: frozenset[str],
    dev_requirements: frozenset[str],
) -> str:
    lines = [
        "Metadata-Version: 2.4",
        f"Name: {distribution_name}",
        "Version: 0.1.0",
        "Provides-Extra: dev",
    ]
    lines.extend(
        f"Requires-Dist: {requirement}" for requirement in sorted(runtime_requirements)
    )
    lines.extend(
        f'Requires-Dist: {requirement}; extra == "dev"'
        for requirement in sorted(dev_requirements)
    )
    return "\n".join(lines) + "\n"


def _without_dependency(
    requirements: frozenset[str],
    dependency_name: str,
) -> frozenset[str]:
    return frozenset(
        requirement
        for requirement in requirements
        if verify_distribution._canonical_dependency_name(requirement)
        != dependency_name
    )


class DistributionMetadataVerificationTests(unittest.TestCase):
    def _root_wheel(
        self,
        root: Path,
        *,
        runtime_requirements: frozenset[str],
        dev_requirements: frozenset[str],
        emperor_source: str = "",
    ) -> Path:
        wheel = root / "emperor-0.1.0-py3-none-any.whl"
        with zipfile.ZipFile(wheel, mode="w") as archive:
            archive.writestr("emperor/__init__.py", emperor_source)
            archive.writestr("model_runtime/__init__.py", "")
            archive.writestr("models/__init__.py", "")
            archive.writestr(
                "emperor-0.1.0.dist-info/METADATA",
                _metadata(
                    "emperor",
                    runtime_requirements,
                    dev_requirements,
                ),
            )
        return wheel

    def _workbench_wheel(
        self,
        root: Path,
        *,
        runtime_requirements: frozenset[str],
        dev_requirements: frozenset[str],
        console_target: str = "emperor_workbench.cli:main",
        include_legacy_launch: bool = False,
        missing_member: str | None = None,
    ) -> Path:
        wheel = root / "emperor_workbench-0.1.0-py3-none-any.whl"
        with zipfile.ZipFile(wheel, mode="w") as archive:
            for member in verify_distribution.WORKBENCH_REQUIRED_WHEEL_MEMBERS:
                if member != missing_member:
                    archive.writestr(member, "")
            if include_legacy_launch:
                archive.writestr("emperor_workbench/launch.py", "")
            archive.writestr(
                "emperor_workbench-0.1.0.dist-info/METADATA",
                _metadata(
                    "emperor-workbench",
                    runtime_requirements,
                    dev_requirements,
                ),
            )
            archive.writestr(
                "emperor_workbench-0.1.0.dist-info/entry_points.txt",
                (f"[console_scripts]\nemperor-workbench = {console_target}\n"),
            )
        return wheel

    def test_root_wheel_records_the_exact_direct_dependency_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            wheel = self._root_wheel(
                Path(temporary),
                runtime_requirements=verify_distribution.ROOT_RUNTIME_REQUIREMENTS,
                dev_requirements=verify_distribution.ROOT_DEV_REQUIREMENTS,
            )

            manifest = verify_distribution._verify_wheel(wheel)

        self.assertEqual(
            manifest["runtime_dependencies"],
            sorted(verify_distribution.ROOT_RUNTIME_DEPENDENCIES),
        )
        self.assertEqual(
            manifest["dev_dependencies"],
            sorted(verify_distribution.ROOT_DEV_DEPENDENCIES),
        )

    def test_no_deps_install_cannot_hide_a_missing_root_declaration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            wheel = self._root_wheel(
                Path(temporary),
                runtime_requirements=_without_dependency(
                    verify_distribution.ROOT_RUNTIME_REQUIREMENTS,
                    "tokenizers",
                ),
                dev_requirements=verify_distribution.ROOT_DEV_REQUIREMENTS,
            )

            with self.assertRaisesRegex(
                verify_distribution.VerificationError,
                r"missing=\['tokenizers'\]",
            ):
                verify_distribution._verify_wheel(wheel)

    def test_workbench_wheel_records_direct_runtime_and_dev_dependencies(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            wheel = self._workbench_wheel(
                Path(temporary),
                runtime_requirements=(
                    verify_distribution.WORKBENCH_RUNTIME_REQUIREMENTS
                ),
                dev_requirements=verify_distribution.WORKBENCH_DEV_REQUIREMENTS,
            )

            manifest = verify_distribution._verify_workbench_wheel(wheel)

        self.assertEqual(
            manifest["runtime_dependencies"],
            sorted(verify_distribution.WORKBENCH_RUNTIME_DEPENDENCIES),
        )
        self.assertEqual(
            manifest["dev_dependencies"],
            sorted(verify_distribution.WORKBENCH_DEV_DEPENDENCIES),
        )
        self.assertEqual(
            manifest["console_scripts"],
            ["emperor-workbench=emperor_workbench.cli:main"],
        )
        self.assertEqual(
            manifest["required_members"],
            sorted(verify_distribution.WORKBENCH_REQUIRED_WHEEL_MEMBERS),
        )

    def test_workbench_wheel_requires_every_installed_worker_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for missing_member in sorted(
                verify_distribution.WORKBENCH_REQUIRED_WORKER_MEMBERS
            ):
                with self.subTest(missing_member=missing_member):
                    wheel = self._workbench_wheel(
                        root,
                        runtime_requirements=(
                            verify_distribution.WORKBENCH_RUNTIME_REQUIREMENTS
                        ),
                        dev_requirements=(
                            verify_distribution.WORKBENCH_DEV_REQUIREMENTS
                        ),
                        missing_member=missing_member,
                    )

                    with self.assertRaises(
                        verify_distribution.VerificationError
                    ) as raised:
                        verify_distribution._verify_workbench_wheel(wheel)

                    self.assertIn("Workbench wheel is missing", str(raised.exception))
                    self.assertIn(missing_member, str(raised.exception))

    def test_workbench_wheel_rejects_a_noncanonical_console_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            wheel = self._workbench_wheel(
                Path(temporary),
                runtime_requirements=(
                    verify_distribution.WORKBENCH_RUNTIME_REQUIREMENTS
                ),
                dev_requirements=verify_distribution.WORKBENCH_DEV_REQUIREMENTS,
                console_target="emperor_workbench.launch:main",
            )

            with self.assertRaisesRegex(
                verify_distribution.VerificationError,
                "canonical entry-point contract",
            ):
                verify_distribution._verify_workbench_wheel(wheel)

    def test_workbench_wheel_rejects_the_legacy_launch_module(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            wheel = self._workbench_wheel(
                Path(temporary),
                runtime_requirements=(
                    verify_distribution.WORKBENCH_RUNTIME_REQUIREMENTS
                ),
                dev_requirements=verify_distribution.WORKBENCH_DEV_REQUIREMENTS,
                include_legacy_launch=True,
            )

            with self.assertRaisesRegex(
                verify_distribution.VerificationError,
                "Legacy Workbench launcher",
            ):
                verify_distribution._verify_workbench_wheel(wheel)

    def test_shared_path_cannot_hide_a_missing_workbench_declaration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            wheel = self._workbench_wheel(
                Path(temporary),
                runtime_requirements=_without_dependency(
                    verify_distribution.WORKBENCH_RUNTIME_REQUIREMENTS,
                    "pydantic",
                ),
                dev_requirements=verify_distribution.WORKBENCH_DEV_REQUIREMENTS,
            )

            with self.assertRaisesRegex(
                verify_distribution.VerificationError,
                r"missing=\['pydantic'\]",
            ):
                verify_distribution._verify_workbench_wheel(wheel)

    def test_requirement_extras_are_part_of_the_wheel_contract(self) -> None:
        runtime_requirements = (
            verify_distribution.WORKBENCH_RUNTIME_REQUIREMENTS
            - {"uvicorn[standard]>=0.51,<0.52"}
        ) | {"uvicorn>=0.51,<0.52"}
        with tempfile.TemporaryDirectory() as temporary:
            wheel = self._workbench_wheel(
                Path(temporary),
                runtime_requirements=frozenset(runtime_requirements),
                dev_requirements=verify_distribution.WORKBENCH_DEV_REQUIREMENTS,
            )

            with self.assertRaisesRegex(
                verify_distribution.VerificationError,
                "version/extras/marker contract",
            ):
                verify_distribution._verify_workbench_wheel(wheel)

    def test_host_shared_path_cannot_hide_a_new_external_import(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            wheel = self._root_wheel(
                Path(temporary),
                runtime_requirements=verify_distribution.ROOT_RUNTIME_REQUIREMENTS,
                dev_requirements=verify_distribution.ROOT_DEV_REQUIREMENTS,
                emperor_source="import httpx\n",
            )

            with self.assertRaisesRegex(
                verify_distribution.VerificationError,
                r"external modules.*\['httpx'\]",
            ):
                verify_distribution._verify_wheel(wheel)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import importlib.util
import json
import sys
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

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


def _write_sdist(path: Path, root_name: str, members: set[str]) -> None:
    with tarfile.open(path, mode="w:gz") as archive:
        for member in sorted(members):
            archive.addfile(tarfile.TarInfo(f"{root_name}/{member}"))


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


class SourceDistributionVerificationTests(unittest.TestCase):
    def test_root_sdist_requires_runtime_scripts(self) -> None:
        members = {
            "constraints/python-3.13-linux-x86_64-cuda-legacy.txt",
            "download_logs.sh",
            "env.ps1",
            "env.sh",
            "experiment.sh",
            "run_test.sh",
            "src/emperor/__init__.py",
            "src/model_runtime/__init__.py",
            "src/models/__init__.py",
            "tests/test_package.py",
        }
        with tempfile.TemporaryDirectory() as temporary:
            sdist = Path(temporary) / "emperor-0.1.0.tar.gz"
            _write_sdist(sdist, "emperor-0.1.0", members)

            manifest = verify_distribution._verify_sdist(sdist)

        self.assertNotIn("docs", manifest)
        self.assertEqual(manifest["emperor"], 1)
        self.assertEqual(manifest["models"], 1)
        self.assertEqual(manifest["model_runtime"], 1)

    def test_root_sdist_rejects_a_missing_runtime_script(self) -> None:
        members = {
            "constraints/python-3.13-linux-x86_64-cuda-legacy.txt",
            "download_logs.sh",
            "env.ps1",
            "experiment.sh",
            "run_test.sh",
            "src/emperor/__init__.py",
            "src/model_runtime/__init__.py",
            "src/models/__init__.py",
            "tests/test_package.py",
        }
        with tempfile.TemporaryDirectory() as temporary:
            sdist = Path(temporary) / "emperor-0.1.0.tar.gz"
            _write_sdist(sdist, "emperor-0.1.0", members)

            with self.assertRaisesRegex(
                verify_distribution.VerificationError,
                "env.sh",
            ):
                verify_distribution._verify_sdist(sdist)

    def test_workbench_sdist_requires_runtime_and_test_support(self) -> None:
        members = set(verify_distribution.WORKBENCH_REQUIRED_SDIST_MEMBERS)
        with tempfile.TemporaryDirectory() as temporary:
            sdist = Path(temporary) / "emperor_workbench-0.1.0.tar.gz"
            _write_sdist(sdist, "emperor_workbench-0.1.0", members)

            manifest = verify_distribution._verify_workbench_sdist(sdist)

        self.assertGreater(manifest["emperor_workbench"], 0)
        self.assertGreater(manifest["tests"], 0)

    def test_workbench_sdist_rejects_frontend_files(self) -> None:
        members = {
            *verify_distribution.WORKBENCH_REQUIRED_SDIST_MEMBERS,
            "web/index.html",
        }
        with tempfile.TemporaryDirectory() as temporary:
            sdist = Path(temporary) / "emperor_workbench-0.1.0.tar.gz"
            _write_sdist(sdist, "emperor_workbench-0.1.0", members)

            with self.assertRaisesRegex(
                verify_distribution.VerificationError,
                "Frontend leaked",
            ):
                verify_distribution._verify_workbench_sdist(sdist)

    def test_workbench_copy_keeps_api_and_excludes_frontend(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = root / "repository"
            api_file = repository / "apps" / "workbench" / "api" / "sentinel.py"
            web_file = repository / "apps" / "workbench" / "web" / "sentinel.js"
            api_file.parent.mkdir(parents=True)
            web_file.parent.mkdir(parents=True)
            api_file.write_text("api", encoding="utf-8")
            web_file.write_text("web", encoding="utf-8")
            destination = root / "copy"

            verify_distribution._copy_workbench(repository, destination)

            self.assertEqual(
                (destination / "api" / "sentinel.py").read_text(encoding="utf-8"),
                "api",
            )
            self.assertFalse((destination / "web").exists())


class DependencyBridgeVerificationTests(unittest.TestCase):
    def test_rejects_first_party_modules_from_the_host(self) -> None:
        payload = {
            "module_origins": {
                "emperor": "/host/site-packages/emperor/__init__.py",
            },
            "sys_path": [],
        }
        with patch.object(
            verify_distribution,
            "_run",
            return_value=json.dumps(payload),
        ):
            with self.assertRaisesRegex(
                verify_distribution.VerificationError,
                "first-party host packages",
            ):
                verify_distribution._require_dependency_bridge_isolated(
                    Path("/venv/bin/python"),
                    outside=Path("/outside"),
                    repository=Path("/checkout"),
                )

    def test_rejects_checkout_paths(self) -> None:
        payload = {
            "module_origins": {},
            "sys_path": ["/checkout/src"],
        }
        with patch.object(
            verify_distribution,
            "_run",
            return_value=json.dumps(payload),
        ):
            with self.assertRaisesRegex(
                verify_distribution.VerificationError,
                "checkout paths",
            ):
                verify_distribution._require_dependency_bridge_isolated(
                    Path("/venv/bin/python"),
                    outside=Path("/outside"),
                    repository=Path("/checkout"),
                )

    def test_allows_the_host_dependency_bridge(self) -> None:
        payload = {
            "module_origins": {},
            "sys_path": ["/checkout/host-site-packages"],
        }
        with (
            patch.object(
                verify_distribution,
                "_run",
                return_value=json.dumps(payload),
            ),
            patch.object(
                verify_distribution,
                "get_path",
                return_value="/checkout/host-site-packages",
            ),
        ):
            verify_distribution._require_dependency_bridge_isolated(
                Path("/venv/bin/python"),
                outside=Path("/outside"),
                repository=Path("/checkout"),
            )


class DependencyCheckVerificationTests(unittest.TestCase):
    def test_runs_pip_check_in_the_isolated_environment(self) -> None:
        python = Path("/venv/bin/python")
        outside = Path("/outside")
        environment = {"PYTHONSAFEPATH": "1"}
        with (
            patch.object(verify_distribution, "_run") as run,
            patch.object(
                verify_distribution,
                "_isolated_environment",
                return_value=environment,
            ),
        ):
            verify_distribution._pip_check(python, outside)

        run.assert_called_once_with(
            [str(python), "-P", "-m", "pip", "check"],
            cwd=outside,
            env=environment,
        )


class WorkbenchLauncherVerificationTests(unittest.TestCase):
    def test_runs_both_launchers_and_probes_server_health(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            python = root / "venv" / "bin" / "python"
            python.parent.mkdir(parents=True)
            executable = python.parent / (
                "emperor-workbench.exe"
                if verify_distribution.os.name == "nt"
                else "emperor-workbench"
            )
            executable.touch()
            outside = root / "outside"
            outside.mkdir()
            state_root = root / "state"
            response = MagicMock()
            response.__enter__.return_value.status = 200
            process = MagicMock(returncode=None)
            process.poll.return_value = None
            with (
                patch.object(
                    verify_distribution,
                    "_isolated_environment",
                    return_value={},
                ),
                patch.object(
                    verify_distribution,
                    "_run",
                    return_value="Run the Emperor Workbench API.",
                ) as run,
                patch.object(
                    verify_distribution,
                    "_available_loopback_port",
                    return_value=43121,
                ),
                patch.object(
                    verify_distribution.subprocess,
                    "Popen",
                    return_value=process,
                ) as popen,
                patch.object(
                    verify_distribution.urllib.request,
                    "urlopen",
                    return_value=response,
                ) as urlopen,
                patch.object(
                    verify_distribution,
                    "_stop_smoke_process",
                    return_value="",
                ) as stop,
                patch.object(
                    verify_distribution.time,
                    "monotonic",
                    side_effect=(0.0, 0.0),
                ),
            ):
                payload = verify_distribution._installed_workbench_cli_smoke(
                    python,
                    outside,
                    state_root,
                )

        self.assertEqual(
            [invocation.args[0] for invocation in run.call_args_list],
            [
                [str(executable), "--help"],
                [str(python), "-P", "-m", "emperor_workbench", "--help"],
            ],
        )
        environment = run.call_args_list[0].kwargs["env"]
        self.assertEqual(
            environment["EMPEROR_PROJECT_ADAPTER_COMMAND"],
            f"{python} -P -m models.adapter_cli",
        )
        self.assertEqual(
            environment["WORKBENCH_API_TRAINING_JOBS_ROOT"],
            str(state_root / "training-jobs"),
        )
        self.assertEqual(
            popen.call_args.args[0],
            [
                str(executable),
                "--host",
                "127.0.0.1",
                "--port",
                "43121",
            ],
        )
        urlopen.assert_called_once_with(
            "http://127.0.0.1:43121/health",
            timeout=0.5,
        )
        stop.assert_called_once_with(process)
        self.assertEqual(
            payload,
            {
                "console_help": True,
                "module_help": True,
                "server_health_status": 200,
            },
        )

    def test_rejects_noncanonical_launcher_help(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            python = root / "venv" / "bin" / "python"
            python.parent.mkdir(parents=True)
            executable = python.parent / (
                "emperor-workbench.exe"
                if verify_distribution.os.name == "nt"
                else "emperor-workbench"
            )
            executable.touch()
            outside = root / "outside"
            outside.mkdir()
            with (
                patch.object(
                    verify_distribution,
                    "_isolated_environment",
                    return_value={},
                ),
                patch.object(
                    verify_distribution,
                    "_run",
                    return_value="unexpected help",
                ),
            ):
                with self.assertRaisesRegex(
                    verify_distribution.VerificationError,
                    "canonical CLI help",
                ):
                    verify_distribution._installed_workbench_cli_smoke(
                        python,
                        outside,
                        root / "state",
                    )

    def test_workbench_smoke_merges_launcher_results(self) -> None:
        workbench_payload = {
            "asgi_title": "Emperor Workbench API",
            "emperor_workbench_path": "/venv/emperor_workbench/__init__.py",
        }
        launcher_payload = {
            "console_help": True,
            "module_help": True,
            "server_health_status": 200,
        }
        with (
            patch.object(
                verify_distribution,
                "_isolated_environment",
                return_value={},
            ),
            patch.object(
                verify_distribution,
                "_run",
                return_value=json.dumps(workbench_payload),
            ) as run,
            patch.object(
                verify_distribution,
                "_installed_workbench_cli_smoke",
                return_value=launcher_payload,
            ) as launcher_smoke,
        ):
            payload = verify_distribution._workbench_smoke(
                Path("/venv/bin/python"),
                Path("/outside"),
                Path("/state"),
            )

        probe_code = run.call_args.args[0][-1]
        self.assertIn(
            "from emperor_workbench.api import app as global_app, create_app",
            probe_code,
        )
        self.assertIn("'asgi_title': global_app.title", probe_code)
        self.assertEqual(
            run.call_args.kwargs["env"]["WORKBENCH_API_STATE_ROOT"],
            "/state/state",
        )
        launcher_smoke.assert_called_once_with(
            Path("/venv/bin/python"),
            Path("/outside"),
            Path("/state/cli"),
        )
        self.assertEqual(payload, workbench_payload | launcher_payload)


if __name__ == "__main__":
    unittest.main()

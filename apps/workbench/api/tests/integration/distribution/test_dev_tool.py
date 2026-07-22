from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.integration.distribution._environment_support import (
    PROJECT_ROOT,
    emperor_dev,
)

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "emperor-matplotlib")
)


class PortableLauncherTests(unittest.TestCase):
    def test_workbench_cli_launches_the_canonical_asgi_target(self) -> None:
        from emperor_workbench.cli import main

        with patch("uvicorn.run") as run:
            main(
                [
                    "--host",
                    "0.0.0.0",
                    "--port",
                    "43210",
                    "--reload",
                ]
            )

        run.assert_called_once_with(
            "emperor_workbench.api:app",
            host="0.0.0.0",
            port=43210,
            reload=True,
            reset_contextvars=True,
        )

    def test_workbench_cli_defaults_are_configurable_from_environment(self) -> None:
        from emperor_workbench.cli import main

        with (
            patch.dict(
                os.environ,
                {
                    "WORKBENCH_BACKEND_HOST": "127.0.0.2",
                    "WORKBENCH_BACKEND_PORT": "43211",
                },
            ),
            patch("uvicorn.run") as run,
        ):
            main([])

        run.assert_called_once_with(
            "emperor_workbench.api:app",
            host="127.0.0.2",
            port=43211,
            reload=False,
            reset_contextvars=True,
        )

    def test_delegated_python_uses_installed_packages_without_pythonpath(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(
                emperor_dev,
                "_require_venv_python",
                return_value=Path("environment") / "python",
            ),
            patch.object(subprocess, "run") as run,
        ):
            run.return_value.returncode = 0
            emperor_dev._run_venv(["-P", "-c", "import emperor_workbench"])

        self.assertNotIn("PYTHONPATH", run.call_args.kwargs["env"])

    def test_dependency_probe_requires_project_packages(self) -> None:
        with patch.object(emperor_dev.subprocess, "run") as run:
            run.return_value.returncode = 0

            available = emperor_dev._dependencies_available(Path("python"))

        self.assertTrue(available)
        command = run.call_args.args[0]
        self.assertTrue(
            {"emperor", "emperor_workbench", "model_runtime", "models"}.issubset(
                command
            )
        )

    def test_service_specs_are_exposed_for_launcher_contracts(self) -> None:
        self.assertTrue(callable(emperor_dev.service_specs))
        self.assertFalse(hasattr(emperor_dev, "_service_specs"))

    def test_windows_service_job_names_are_scoped_and_validated(self) -> None:
        name = "_emperor_windows_jobs_test"
        spec = importlib.util.spec_from_file_location(
            name,
            PROJECT_ROOT / "tools" / "_emperor_windows_jobs.py",
        )
        self.assertIsNotNone(spec)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)

        self.assertEqual(
            module.service_job_object_name("backend", 9999),
            "Local\\EmperorWorkbenchService-backend-9999",
        )
        for service, port in (
            ("unsafe/name", 9999),
            ("backend", 0),
            ("backend", 65536),
        ):
            with self.subTest(service=service, port=port):
                with self.assertRaises(ValueError):
                    module.service_job_object_name(service, port)

    def test_workbench_status_trusts_the_checkout_when_already_in_venv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            environment = {
                key: value for key, value in os.environ.items() if key != "PYTHONPATH"
            }
            environment.update(
                {
                    "PYTHONSAFEPATH": "1",
                    "WORKBENCH_RUNTIME_ROOT": temporary,
                }
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    "-P",
                    str(PROJECT_ROOT / "tools" / "emperor_dev.py"),
                    "workbench",
                    "status",
                ],
                cwd=PROJECT_ROOT,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.count("stopped"), 2)

    def test_runtime_subcommands_forward_help_instead_of_consuming_it(self) -> None:
        for command in ("experiment", "test", "logs:archive", "python"):
            with self.subTest(command=command):
                parsed = emperor_dev.parse_args([command, "--help"])
                self.assertEqual(parsed.command, command)
                self.assertEqual(parsed.arguments, ["--help"])

    def test_powershell_wrapper_exposes_status_and_stop_switches(self) -> None:
        source = (PROJECT_ROOT / "env.ps1").read_text(encoding="utf-8")
        self.assertIn("[switch]$WorkbenchStatus", source)
        self.assertIn("[switch]$WorkbenchStop", source)
        self.assertIn("torchenv\\Scripts\\Activate.ps1", source)
        fast_path_start = source.index('if ($Task -ne "workbench:start")')
        dev_start = source.index("mise run dev")
        self.assertLess(fast_path_start, dev_start)
        fast_path = source[fast_path_start:dev_start]
        self.assertIn("mise run $Task", fast_path)
        self.assertIn("return", fast_path)

    def test_runtime_metadata_rejects_creation_time_and_command_mismatches(
        self,
    ) -> None:
        import psutil

        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary)
            recorded_command = (
                sys.executable,
                "-c",
                "import time; time.sleep(30)",
            )
            process = subprocess.Popen(  # noqa: S603
                recorded_command,
                cwd=PROJECT_ROOT,
            )
            try:
                observed = psutil.Process(process.pid)
                spec = emperor_dev.ServiceSpec(
                    name="probe",
                    port=43210,
                    command=recorded_command,
                    command_identity="time.sleep",
                    cwd=PROJECT_ROOT,
                    environment={},
                    ready_url="http://127.0.0.1:43210/health",
                )
                payload = {
                    "argv": list(spec.command),
                    "commandIdentity": spec.command_identity,
                    "createTime": observed.create_time(),
                    "pid": process.pid,
                    "port": spec.port,
                }
                with patch.object(emperor_dev, "RUNTIME_ROOT", runtime):
                    spec.metadata_path.write_text(json.dumps(payload), encoding="utf-8")
                    self.assertIsNotNone(
                        emperor_dev._validated_runtime_process(
                            spec,
                            require_current_command=True,
                        )
                    )
                    payload["createTime"] = float(payload["createTime"]) - 1
                    spec.metadata_path.write_text(json.dumps(payload), encoding="utf-8")
                    self.assertIsNone(
                        emperor_dev._validated_runtime_process(
                            spec,
                            require_current_command=True,
                        )
                    )
                    payload["createTime"] = observed.create_time()
                    payload["commandIdentity"] = "different-command"
                    spec.metadata_path.write_text(json.dumps(payload), encoding="utf-8")
                    self.assertIsNone(
                        emperor_dev._validated_runtime_process(
                            spec,
                            require_current_command=True,
                        )
                    )
            finally:
                process.terminate()
                process.wait(timeout=5)

    def test_stop_uses_startup_metadata_after_launcher_command_changes(self) -> None:
        import psutil

        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary)
            recorded_command = (
                sys.executable,
                "-c",
                "import time; time.sleep(30)",
            )
            process = subprocess.Popen(  # noqa: S603
                recorded_command,
                cwd=PROJECT_ROOT,
            )
            try:
                observed = psutil.Process(process.pid)
                spec = emperor_dev.ServiceSpec(
                    name="probe",
                    port=43210,
                    command=(*recorded_command, "--new-launcher-command"),
                    command_identity="time.sleep",
                    cwd=PROJECT_ROOT,
                    environment={},
                    ready_url="http://127.0.0.1:43210/health",
                )
                payload = {
                    "argv": list(recorded_command),
                    "commandIdentity": spec.command_identity,
                    "createTime": observed.create_time(),
                    "jobName": None,
                    "pid": process.pid,
                    "port": spec.port,
                }
                with patch.object(emperor_dev, "RUNTIME_ROOT", runtime):
                    spec.metadata_path.write_text(json.dumps(payload), encoding="utf-8")
                    emperor_dev._stop_service(spec, quiet=True)

                self.assertIsNotNone(process.poll())
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=5)

    def test_retired_backend_metadata_identity_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary)
            spec = emperor_dev.ServiceSpec(
                name="backend",
                port=43210,
                command=(
                    sys.executable,
                    "-m",
                    "emperor_workbench",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "43210",
                ),
                command_identity="emperor_workbench",
                cwd=PROJECT_ROOT,
                environment={},
                ready_url="http://127.0.0.1:43210/health",
            )
            retired_command = (
                sys.executable,
                "-m",
                "emperor_workbench.launch",
                "--host",
                "127.0.0.1",
                "--port",
                "43210",
            )
            payload = {
                "argv": list(retired_command),
                "commandIdentity": "emperor_workbench.launch",
                "createTime": 123.0,
                "jobName": None,
                "pid": 456,
                "port": spec.port,
            }
            with patch.object(emperor_dev, "RUNTIME_ROOT", runtime):
                spec.metadata_path.write_text(json.dumps(payload), encoding="utf-8")
                self.assertIsNone(
                    emperor_dev._validated_runtime_process(
                        spec,
                        require_current_command=False,
                    )
                )

                payload["unexpected"] = True
                spec.metadata_path.write_text(json.dumps(payload), encoding="utf-8")
                self.assertIsNone(
                    emperor_dev._validated_runtime_process(
                        spec,
                        require_current_command=False,
                    )
                )

    def test_readiness_requires_an_http_response(self) -> None:
        class RunningProcess:
            pid = 123

            @staticmethod
            def is_running() -> bool:
                return True

            @staticmethod
            def status() -> str:
                return "running"

        spec = emperor_dev.ServiceSpec(
            name="probe",
            port=43210,
            command=("python",),
            command_identity="probe",
            cwd=PROJECT_ROOT,
            environment={},
            ready_url="http://127.0.0.1:43210/health",
        )
        with (
            patch.object(emperor_dev, "_http_ready", return_value=False),
            self.assertRaisesRegex(SystemExit, "did not become HTTP-ready"),
        ):
            emperor_dev._wait_ready(spec, RunningProcess(), timeout=0.001)

    def test_retired_pid_metadata_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary)
            spec = emperor_dev.ServiceSpec(
                name="backend",
                port=43210,
                command=(sys.executable, "-m", "emperor_workbench"),
                command_identity="emperor_workbench",
                cwd=PROJECT_ROOT,
                environment=os.environ.copy(),
                ready_url="http://127.0.0.1:43210/health",
            )
            retired_path = runtime / "backend.pid"
            retired_path.write_text("123\n", encoding="utf-8")
            with patch.object(emperor_dev, "RUNTIME_ROOT", runtime):
                self.assertIsNone(
                    emperor_dev._validated_runtime_process(
                        spec,
                        require_current_command=False,
                    )
                )
            self.assertEqual(retired_path.read_text(encoding="utf-8"), "123\n")

    def test_canonical_workbench_cli_has_no_launch_module_alias(self) -> None:
        self.assertIsNone(
            importlib.util.find_spec("emperor_workbench.launch"),
        )


if __name__ == "__main__":
    unittest.main()

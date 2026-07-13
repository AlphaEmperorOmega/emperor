from __future__ import annotations

import importlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, call, patch

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "emperor-matplotlib")
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _load_emperor_dev():
    name = "_emperor_dev_environment_tests"
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

REQUIRED_BACKEND_TEST_MODULES = (
    ("torch", "torch"),
    ("fastapi", "fastapi"),
    ("tensorboard", "tensorboard"),
    ("pydantic", "pydantic"),
    ("pydantic-settings", "pydantic_settings"),
    ("httpx", "httpx"),
    ("lightning", "lightning.pytorch"),
    ("ruff", "ruff"),
    ("filelock", "filelock"),
    ("psutil", "psutil"),
)


class BackendTestEnvironmentTests(unittest.TestCase):
    def test_required_backend_test_dependencies_are_importable(self) -> None:
        missing: list[str] = []
        for package_name, module_name in REQUIRED_BACKEND_TEST_MODULES:
            with self.subTest(module=module_name):
                try:
                    importlib.import_module(module_name)
                except ModuleNotFoundError as error:
                    missing.append(
                        f"{package_name} ({module_name}; missing {error.name})"
                    )
        self.assertEqual(missing, [])

    def test_programmatic_uvicorn_servers_enable_contextvar_isolation(self) -> None:
        server_cases = (
            (
                "workbench.backend.tests.contract_e2e_server",
                [
                    "contract_e2e_server",
                    "--root",
                    "{root}",
                    "--port",
                    "54321",
                    "--token",
                    "test-token",
                    "--frontend-origin",
                    "http://127.0.0.1:9000",
                ],
                (),
            ),
            (
                "workbench.backend.tests.browser_performance_server",
                [
                    "browser_performance_server",
                    "--root",
                    "{root}",
                    "--port",
                    "54322",
                    "--frontend-origin",
                    "http://127.0.0.1:9000",
                ],
                ("_seed_log_runs", "_write_import_fixture"),
            ),
        )
        for module_name, arguments, setup_names in server_cases:
            with self.subTest(module=module_name), tempfile.TemporaryDirectory() as tmp:
                module = importlib.import_module(module_name)
                argv = [argument.format(root=tmp) for argument in arguments]
                setup_patchers = [patch.object(module, name) for name in setup_names]
                for patcher in setup_patchers:
                    patcher.start()
                try:
                    with (
                        patch.object(sys, "argv", argv),
                        patch.object(
                            module,
                            "create_app_with_training_service",
                            return_value=object(),
                        ),
                        patch.object(module.uvicorn, "run") as run,
                    ):
                        module.main()
                finally:
                    for patcher in reversed(setup_patchers):
                        patcher.stop()
                run.assert_called_once()
                self.assertIs(run.call_args.kwargs["reset_contextvars"], True)


class PortableLauncherTests(unittest.TestCase):
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
            emperor_dev._run_venv(["-P", "-c", "import workbench"])

        self.assertNotIn("PYTHONPATH", run.call_args.kwargs["env"])

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
        for command in ("experiment", "test", "logs-archive", "python"):
            with self.subTest(command=command):
                parsed = emperor_dev.parse_args([command, "--help"])
                self.assertEqual(parsed.command, command)
                self.assertEqual(parsed.arguments, ["--help"])

    def test_unix_entry_points_are_thin_compatibility_wrappers(self) -> None:
        cases = {
            "env.sh": "mise run",
            "experiment.sh": "-m models.project_cli",
            "run_test.sh": "-m models.project_cli test",
            "download_logs.sh": "-m models.project_cli logs:archive",
        }
        forbidden = ("/dev/tcp", "lsof", "fuser", "nohup", "setsid", "cksum")
        for name, delegation in cases.items():
            with self.subTest(script=name):
                source = (PROJECT_ROOT / name).read_text(encoding="utf-8")
                self.assertIn(delegation, source)
                self.assertFalse(any(token in source for token in forbidden))

    def test_unix_workbench_stop_bypasses_setup(self) -> None:
        script = """
mise() {
  printf 'mise-call:%s\\n' "$*"
  if [ "$1" = "run" ] && [ "$2" = "setup" ]; then
    return 23
  fi
}
export -f mise
source "$1" --workbench-stop
"""
        completed = subprocess.run(
            ["bash", "-c", script, "bash", str(PROJECT_ROOT / "env.sh")],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(
            completed.stdout.splitlines(),
            ["mise-call:run workbench:stop"],
        )

    def test_unix_workbench_start_is_atomic_across_setup(self) -> None:
        script = """
mise() {
  printf 'mise-call:%s\\n' "$*"
  if [ "$1" = "run" ] && [ "$2" = "setup" ]; then
    printf 'Setup ready: probe\\n'
    return 23
  fi
}
export -f mise
env_script="$1"
set --
source "$env_script"
"""
        completed = subprocess.run(
            ["bash", "-c", script, "bash", str(PROJECT_ROOT / "env.sh")],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(
            completed.stdout.splitlines(),
            ["mise-call:run dev --profile cpu"],
        )

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

    def test_pid_metadata_rejects_creation_time_and_command_mismatches(self) -> None:
        import psutil

        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary)
            process = subprocess.Popen(  # noqa: S603 - current Python executable
                [sys.executable, "-c", "import time; time.sleep(30)"],
            )
            try:
                observed = psutil.Process(process.pid)
                spec = emperor_dev.ServiceSpec(
                    name="probe",
                    port=43210,
                    command=(sys.executable,),
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
                    self.assertIsNotNone(emperor_dev._validated_process(spec))
                    payload["createTime"] = float(payload["createTime"]) - 1
                    spec.metadata_path.write_text(json.dumps(payload), encoding="utf-8")
                    self.assertIsNone(emperor_dev._validated_process(spec))
                    payload["createTime"] = observed.create_time()
                    payload["commandIdentity"] = "different-command"
                    spec.metadata_path.write_text(json.dumps(payload), encoding="utf-8")
                    self.assertIsNone(emperor_dev._validated_process(spec))
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
            process = subprocess.Popen(recorded_command)  # noqa: S603
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

    def test_start_retires_legacy_pid_before_checking_the_port(self) -> None:
        class StartedProcess:
            pid = 456

        class ObservedProcess:
            @staticmethod
            def create_time() -> float:
                return 123.0

        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary)
            spec = emperor_dev.ServiceSpec(
                name="backend",
                port=43210,
                command=(sys.executable, "-c", "pass"),
                command_identity="pass",
                cwd=PROJECT_ROOT,
                environment=os.environ.copy(),
                ready_url="http://127.0.0.1:43210/health",
            )
            events = Mock()

            def port_closed(_port: int) -> bool:
                events.port_check()
                return False

            with (
                patch.object(emperor_dev, "RUNTIME_ROOT", runtime),
                patch.object(emperor_dev, "_validated_process", return_value=None),
                patch.object(
                    emperor_dev,
                    "_retire_legacy_service",
                    side_effect=lambda _spec: events.retire(),
                ),
                patch.object(
                    emperor_dev,
                    "_port_open",
                    side_effect=port_closed,
                ),
                patch.object(
                    emperor_dev.subprocess,
                    "Popen",
                    return_value=StartedProcess(),
                ),
                patch("psutil.Process", return_value=ObservedProcess()),
                patch.object(emperor_dev, "_wait_ready"),
            ):
                emperor_dev._start_service(spec)

        self.assertEqual(
            events.mock_calls[:2],
            [call.retire(), call.port_check()],
        )

    def test_legacy_pid_requires_matching_identity_and_creation_time(self) -> None:
        import psutil

        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary)
            spec = emperor_dev.ServiceSpec(
                name="backend",
                port=43210,
                command=(
                    sys.executable,
                    "-m",
                    "workbench.backend.launch",
                    "--port",
                    "43210",
                ),
                command_identity="workbench.backend.launch",
                cwd=PROJECT_ROOT,
                environment={},
                ready_url="http://127.0.0.1:43210/health",
            )
            legacy_pid = runtime / "backend.pid"
            legacy_pid.write_text("123\n", encoding="utf-8")
            process = Mock(pid=123)
            process.create_time.return_value = legacy_pid.stat().st_mtime
            process.status.return_value = psutil.STATUS_RUNNING
            process.cwd.return_value = str(PROJECT_ROOT)
            process.cmdline.return_value = [
                sys.executable,
                "-m",
                "uvicorn",
                "workbench.backend.api:app",
                "--port",
                "43210",
            ]

            with (
                patch.object(emperor_dev, "RUNTIME_ROOT", runtime),
                patch("psutil.Process", return_value=process),
            ):
                self.assertIs(
                    emperor_dev._validated_legacy_process(spec),
                    process,
                )
                process.cmdline.return_value = [sys.executable, "unrelated.py"]
                self.assertIsNone(emperor_dev._validated_legacy_process(spec))
                process.cmdline.return_value = [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "workbench.backend.api:app",
                    "--port",
                    "43210",
                ]
                process.create_time.return_value = legacy_pid.stat().st_mtime + 2
                self.assertIsNone(emperor_dev._validated_legacy_process(spec))


if __name__ == "__main__":
    unittest.main()

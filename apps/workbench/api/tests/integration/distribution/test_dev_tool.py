from __future__ import annotations

import importlib.util
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

    def test_download_logs_selects_configured_or_repository_python(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            script = root / "download_logs.sh"
            script.write_text(
                (PROJECT_ROOT / "download_logs.sh").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            trace = root / "python-invocation.txt"
            interpreter_source = """#!/usr/bin/env bash
printf '%s\\n' "$0" "$@" > "$EMPEROR_TEST_TRACE"
"""
            repository_python = root / "torchenv" / "bin" / "python"
            repository_python.parent.mkdir(parents=True)
            repository_python.write_text(interpreter_source, encoding="utf-8")
            repository_python.chmod(0o755)
            configured_python = root / "configured-python"
            configured_python.write_text(interpreter_source, encoding="utf-8")
            configured_python.chmod(0o755)
            path_bin = root / "path-bin"
            path_bin.mkdir()
            path_python = path_bin / "python3"
            path_python.write_text("#!/usr/bin/env bash\nexit 91\n", encoding="utf-8")
            path_python.chmod(0o755)

            base_environment = os.environ.copy()
            base_environment.pop("PYTHON", None)
            base_environment.pop("EMPEROR_PYTHON", None)
            base_environment["EMPEROR_TEST_TRACE"] = str(trace)
            base_environment["PATH"] = os.pathsep.join(
                (str(path_bin), base_environment["PATH"])
            )

            cases = (
                ("repository", None, repository_python),
                ("configured", configured_python, configured_python),
            )
            for name, override, expected_python in cases:
                with self.subTest(selection=name):
                    environment = base_environment.copy()
                    if override is not None:
                        environment["EMPEROR_PYTHON"] = str(override)
                    completed = subprocess.run(
                        ["bash", str(script), "logs", "archive.zip"],
                        cwd=root,
                        env=environment,
                        capture_output=True,
                        text=True,
                        check=False,
                    )

                    self.assertEqual(completed.returncode, 0, completed.stderr)
                    self.assertEqual(
                        trace.read_text(encoding="utf-8").splitlines(),
                        [
                            str(expected_python),
                            "-m",
                            "models.project_cli",
                            "logs:archive",
                            "logs",
                            "archive.zip",
                        ],
                    )

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

    def test_canonical_workbench_cli_has_no_launch_module_alias(self) -> None:
        self.assertIsNone(
            importlib.util.find_spec("emperor_workbench.launch"),
        )


if __name__ == "__main__":
    unittest.main()

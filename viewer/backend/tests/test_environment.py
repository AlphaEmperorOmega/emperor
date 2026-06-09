from __future__ import annotations

import importlib
import os
import shlex
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ENV_SCRIPT = PROJECT_ROOT / "env.sh"


REQUIRED_BACKEND_TEST_MODULES = (
    ("torch", "torch"),
    ("fastapi", "fastapi"),
    ("tensorboard", "tensorboard"),
    ("pydantic", "pydantic"),
    ("pydantic-settings", "pydantic_settings"),
    ("httpx", "httpx"),
    ("lightning", "lightning.pytorch"),
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

        self.assertEqual(
            missing,
            [],
            "Missing backend test dependencies. Install the project dependencies "
            "with `python -m pip install -e .` or run tests from an environment "
            f"where these modules are available: {', '.join(missing)}",
        )


class EnvScriptTests(unittest.TestCase):
    def test_dependency_marker_skips_current_installs_and_reinstalls_stale(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as project_dir:
            project_path = Path(project_dir)
            venv_bin = project_path / "torchenv" / "bin"
            marker_path = project_path / "torchenv" / ".emperor-pyproject.cksum"
            calls_path = project_path / "python-calls.log"
            venv_bin.mkdir(parents=True)
            (project_path / "pyproject.toml").write_text(
                "[project]\nname = \"fake-emperor\"\n",
                encoding="utf-8",
            )
            python_path = venv_bin / "python"
            python_path.write_text(
                "#!/usr/bin/env bash\n"
                f"printf '%s\\n' \"$*\" >> {shlex.quote(str(calls_path))}\n"
                "exit 0\n",
                encoding="utf-8",
            )
            python_path.chmod(0o755)

            script = textwrap.dedent(
                f"""
                set -e
                source {shlex.quote(str(ENV_SCRIPT))} --viewer-status >/dev/null
                PROJECT_ROOT={shlex.quote(str(project_path))}
                VENV_PATH={shlex.quote(str(project_path / "torchenv"))}
                VIEWER_DEPENDENCY_MARKER={shlex.quote(str(marker_path))}

                pyproject_dependency_signature > "$VIEWER_DEPENDENCY_MARKER"
                ensure_project_dependencies
                printf 'stale\\n' > "$VIEWER_DEPENDENCY_MARKER"
                ensure_project_dependencies
                test "$(cat "$VIEWER_DEPENDENCY_MARKER")" != "stale"
                """
            )

            subprocess.run(
                ["bash", "-c", script],
                check=True,
                capture_output=True,
                text=True,
            )

            calls = calls_path.read_text(encoding="utf-8").splitlines()

        pip_calls = [call for call in calls if call.startswith("-m pip install")]
        dependency_checks = [call for call in calls if call.startswith("- ")]
        self.assertEqual(
            dependency_checks,
            [
                "- torch fastapi uvicorn tensorboard pydantic "
                "pydantic_settings httpx lightning.pytorch",
                "- torch fastapi uvicorn tensorboard pydantic "
                "pydantic_settings httpx lightning.pytorch",
            ],
        )
        self.assertEqual(
            pip_calls,
            [
                "-m pip install --upgrade pip",
                "-m pip install -e .",
            ],
        )

    def test_start_viewer_validates_dependencies_before_backend_launch(self) -> None:
        with tempfile.TemporaryDirectory() as runtime_dir:
            script = textwrap.dedent(
                f"""
                set -e
                source {shlex.quote(str(ENV_SCRIPT))} --viewer-status >/dev/null
                VIEWER_RUNTIME_PATH={shlex.quote(runtime_dir)}
                ensure_mise() {{ echo ensure_mise; }}
                mise() {{ echo mise "$@"; }}
                ensure_project_dependencies() {{ echo ensure_project_dependencies; }}
                install_frontend_dependencies() {{ echo install_frontend_dependencies; }}
                start_viewer_backend() {{ echo start_viewer_backend; }}
                start_viewer_frontend() {{ echo start_viewer_frontend; }}
                start_viewer
                """
            )

            result = subprocess.run(
                ["bash", "-c", script],
                check=True,
                capture_output=True,
                text=True,
            )

        calls = result.stdout.splitlines()
        self.assertLess(
            calls.index("ensure_project_dependencies"),
            calls.index("start_viewer_backend"),
            calls,
        )

    def test_backend_start_failure_returns_nonzero_and_removes_pid(self) -> None:
        with tempfile.TemporaryDirectory() as project_dir:
            project_path = Path(project_dir)
            venv_bin = project_path / "torchenv" / "bin"
            runtime_path = project_path / "viewer" / ".runtime"
            venv_bin.mkdir(parents=True)
            runtime_path.mkdir(parents=True)
            python_path = venv_bin / "python"
            python_path.write_text(
                "#!/usr/bin/env bash\n"
                "echo 'fake uvicorn failure' >&2\n"
                "exit 1\n",
                encoding="utf-8",
            )
            python_path.chmod(0o755)

            script = textwrap.dedent(
                f"""
                set -e
                source {shlex.quote(str(ENV_SCRIPT))} --viewer-status >/dev/null
                PROJECT_ROOT={shlex.quote(str(project_path))}
                VENV_PATH={shlex.quote(str(project_path / "torchenv"))}
                VIEWER_RUNTIME_PATH={shlex.quote(str(runtime_path))}
                VIEWER_BACKEND_PID="$VIEWER_RUNTIME_PATH/backend.pid"
                VIEWER_BACKEND_LOG="$VIEWER_RUNTIME_PATH/backend.log"
                VIEWER_BACKEND_PORT=65534
                wait_for_port() {{ return 1; }}

                if start_viewer_backend; then
                  echo "backend unexpectedly succeeded"
                  exit 1
                fi

                test ! -f "$VIEWER_BACKEND_PID"
                """
            )

            subprocess.run(
                ["bash", "-c", script],
                check=True,
                capture_output=True,
                text=True,
            )


if __name__ == "__main__":
    unittest.main()

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
    ("ruff", "ruff"),
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
    def test_stale_existing_venv_is_recreated_before_starting_viewer(self) -> None:
        with tempfile.TemporaryDirectory() as project_dir:
            project_path = Path(project_dir)
            venv_bin = project_path / "torchenv" / "bin"
            venv_bin.mkdir(parents=True)
            (venv_bin / "python").write_text(
                "#!/usr/bin/env bash\nexit 0\n",
                encoding="utf-8",
            )
            (venv_bin / "python").chmod(0o755)
            script_path = project_path / "env.sh"
            injection = textwrap.dedent(
                """
                ensure_mise() { echo ensure_mise; }
                mise() { echo mise "$@"; }
                mise_python_version() { echo 3.13; }
                venv_python_version() { echo 3.12; }
                create_venv() {
                  echo create_venv;
                  mkdir -p "$VENV_PATH/bin";
                  printf '#!/usr/bin/env bash\\nexit 0\\n' > "$VENV_PATH/bin/python";
                  chmod +x "$VENV_PATH/bin/python";
                }
                activate_venv() { echo activate_venv; }
                ensure_project_dependencies() { echo ensure_project_dependencies; }
                install_frontend_dependencies() { echo install_frontend_dependencies; }
                start_viewer() { echo start_viewer; }
                """
            )
            script_path.write_text(
                ENV_SCRIPT.read_text(encoding="utf-8").replace(
                    '\nif [ "$VIEWER_ACTION" = "stop" ]; then',
                    f"\n{injection}\nif [ \"$VIEWER_ACTION\" = \"stop\" ]; then",
                    1,
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                ["bash", "-c", f"source {shlex.quote(str(script_path))}"],
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertIn(
            "Recreating virtual environment for Python 3.13",
            result.stdout,
        )
        self.assertIn("create_venv", result.stdout)
        self.assertLess(
            result.stdout.splitlines().index("create_venv"),
            result.stdout.splitlines().index("start_viewer"),
        )

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
                '[project]\nname = "fake-emperor"\n',
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
                "pydantic_settings httpx lightning.pytorch ruff",
                "- torch fastapi uvicorn tensorboard pydantic "
                "pydantic_settings httpx lightning.pytorch ruff",
            ],
        )
        self.assertEqual(
            pip_calls,
            [
                "-m pip install --upgrade pip",
                "-m pip install -e .[dev]",
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
                install_frontend_dependencies() {{
                    echo install_frontend_dependencies;
                }}
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

    def test_stop_viewer_discovers_known_backend_listener_without_pid_file(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as runtime_dir:
            pid_file = Path(runtime_dir) / "backend.pid"
            script = textwrap.dedent(
                f"""
                set -e
                source {shlex.quote(str(ENV_SCRIPT))} --viewer-status >/dev/null
                BACKEND_PID_FILE={shlex.quote(str(pid_file))}
                BACKEND_COMMAND="$VENV_PATH/bin/python -m uvicorn "
                BACKEND_COMMAND="${{BACKEND_COMMAND}}viewer.backend.api:app --reload"
                VIEWER_BACKEND_PORT=9999
                port_listening() {{ return 0; }}
                listener_pids_for_port() {{ echo 1234; }}
                process_command() {{
                  echo "$BACKEND_COMMAND"
                }}
                process_group_id() {{ echo 1234; }}
                kill() {{ echo "kill $*"; }}

                stop_viewer_service "backend" "$BACKEND_PID_FILE" "$VIEWER_BACKEND_PORT"
                test ! -f "$BACKEND_PID_FILE"
                """
            )

            result = subprocess.run(
                ["bash", "-c", script],
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertIn("kill -TERM -- -1234", result.stdout)
        self.assertIn(
            "Stopped viewer backend (1234, discovered from port 9999)",
            result.stdout,
        )

    def test_stop_viewer_signals_backend_group_for_reload_child_listener(self) -> None:
        with tempfile.TemporaryDirectory() as runtime_dir:
            pid_file = Path(runtime_dir) / "backend.pid"
            script = textwrap.dedent(
                f"""
                set -e
                source {shlex.quote(str(ENV_SCRIPT))} --viewer-status >/dev/null
                BACKEND_PID_FILE={shlex.quote(str(pid_file))}
                BACKEND_COMMAND="$VENV_PATH/bin/python -m uvicorn "
                BACKEND_COMMAND="${{BACKEND_COMMAND}}viewer.backend.api:app --reload"
                SPAWN_COMMAND="$VENV_PATH/bin/python -c "
                SPAWN_COMMAND="${{SPAWN_COMMAND}}from multiprocessing.spawn "
                SPAWN_COMMAND="${{SPAWN_COMMAND}}import spawn_main "
                SPAWN_COMMAND="${{SPAWN_COMMAND}}--multiprocessing-fork"
                VIEWER_BACKEND_PORT=9999
                port_listening() {{ return 0; }}
                listener_pids_for_port() {{ echo 5678; }}
                process_command() {{
                  if [ "$1" = "5678" ]; then
                    echo "$SPAWN_COMMAND"
                  else
                    echo "$BACKEND_COMMAND"
                  fi
                }}
                process_group_id() {{ echo 1234; }}
                kill() {{ echo "kill $*"; }}

                stop_viewer_service "backend" "$BACKEND_PID_FILE" "$VIEWER_BACKEND_PORT"
                test ! -f "$BACKEND_PID_FILE"
                """
            )

            result = subprocess.run(
                ["bash", "-c", script],
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertIn("kill -TERM -- -1234", result.stdout)
        self.assertIn(
            "Stopped viewer backend (5678, discovered from port 9999)",
            result.stdout,
        )

    def test_start_viewer_backend_restarts_no_pid_backend_listener(self) -> None:
        with tempfile.TemporaryDirectory() as project_dir:
            project_path = Path(project_dir)
            venv_bin = project_path / "torchenv" / "bin"
            runtime_path = project_path / "viewer" / ".runtime"
            args_path = project_path / "backend-args.txt"
            venv_bin.mkdir(parents=True)
            runtime_path.mkdir(parents=True)
            python_path = venv_bin / "python"
            python_path.write_text(
                "#!/usr/bin/env bash\n"
                "{\n"
                "  for arg in \"$@\"; do printf '%s\\n' \"$arg\"; done\n"
                f"}} > {shlex.quote(str(args_path))}\n"
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
                VIEWER_RUNTIME_PATH={shlex.quote(str(runtime_path))}
                VIEWER_BACKEND_PID="$VIEWER_RUNTIME_PATH/backend.pid"
                VIEWER_BACKEND_LOG="$VIEWER_RUNTIME_PATH/backend.log"
                VIEWER_BACKEND_PORT=65533
                PORT_OPEN=1
                BACKEND_COMMAND="$VENV_PATH/bin/python -m uvicorn "
                BACKEND_COMMAND="${{BACKEND_COMMAND}}viewer.backend.api:app --reload"
                port_listening() {{ [ "$PORT_OPEN" = "1" ]; }}
                listener_pids_for_port() {{ echo 1234; }}
                process_command() {{
                  echo "$BACKEND_COMMAND"
                }}
                kill() {{ echo "kill $*"; PORT_OPEN=0; }}
                wait_for_port() {{ return 0; }}

                start_viewer_backend
                for _attempt in $(seq 1 20); do
                  if [ -f {shlex.quote(str(args_path))} ]; then
                    break
                  fi
                  sleep 0.05
                done
                test -f {shlex.quote(str(args_path))}
                """
            )

            result = subprocess.run(
                ["bash", "-c", script],
                check=True,
                capture_output=True,
                text=True,
            )
            args = args_path.read_text(encoding="utf-8").splitlines()

        self.assertIn(
            "Viewer backend is listening on port 65533 without a pid file; "
            "restarting it to apply current launcher settings.",
            result.stdout,
        )
        self.assertIn("kill -TERM 1234", result.stdout)
        self.assertIn("--reload-dir", args)

    def test_start_viewer_backend_force_stops_no_pid_backend_after_term_timeout(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as project_dir:
            project_path = Path(project_dir)
            venv_bin = project_path / "torchenv" / "bin"
            runtime_path = project_path / "viewer" / ".runtime"
            args_path = project_path / "backend-args.txt"
            venv_bin.mkdir(parents=True)
            runtime_path.mkdir(parents=True)
            python_path = venv_bin / "python"
            python_path.write_text(
                "#!/usr/bin/env bash\n"
                "{\n"
                "  for arg in \"$@\"; do printf '%s\\n' \"$arg\"; done\n"
                f"}} > {shlex.quote(str(args_path))}\n"
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
                VIEWER_RUNTIME_PATH={shlex.quote(str(runtime_path))}
                VIEWER_BACKEND_PID="$VIEWER_RUNTIME_PATH/backend.pid"
                VIEWER_BACKEND_LOG="$VIEWER_RUNTIME_PATH/backend.log"
                VIEWER_BACKEND_PORT=65533
                FORCE_STOPPED=0
                BACKEND_COMMAND="$VENV_PATH/bin/python -m uvicorn "
                BACKEND_COMMAND="${{BACKEND_COMMAND}}viewer.backend.api:app --reload"
                port_listening() {{ return 0; }}
                listener_pids_for_port() {{ echo 1234; }}
                process_command() {{
                  echo "$BACKEND_COMMAND"
                }}
                process_group_id() {{ echo 1234; }}
                kill() {{
                  echo "kill $*";
                  if [ "$1" = "-KILL" ]; then
                    FORCE_STOPPED=1;
                  fi
                }}
                wait_for_port_to_close() {{
                  [ "$FORCE_STOPPED" = "1" ];
                }}
                wait_for_port() {{ return 0; }}

                start_viewer_backend
                for _attempt in $(seq 1 20); do
                  if [ -f {shlex.quote(str(args_path))} ]; then
                    break
                  fi
                  sleep 0.05
                done
                test -f {shlex.quote(str(args_path))}
                """
            )

            result = subprocess.run(
                ["bash", "-c", script],
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertIn("kill -TERM -- -1234", result.stdout)
        self.assertIn("kill -KILL -- -1234", result.stdout)
        self.assertIn("Force-stopped viewer backend", result.stdout)
        self.assertIn("did not close after TERM", result.stderr)

    def test_backend_reload_watches_source_dirs_not_mutable_runtime_state(self) -> None:
        with tempfile.TemporaryDirectory() as project_dir:
            project_path = Path(project_dir)
            venv_bin = project_path / "torchenv" / "bin"
            runtime_path = project_path / "viewer" / ".runtime"
            args_path = project_path / "backend-args.txt"
            venv_bin.mkdir(parents=True)
            runtime_path.mkdir(parents=True)
            python_path = venv_bin / "python"
            python_path.write_text(
                "#!/usr/bin/env bash\n"
                "{\n"
                "  for arg in \"$@\"; do printf '%s\\n' \"$arg\"; done\n"
                f"}} > {shlex.quote(str(args_path))}\n"
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
                VIEWER_RUNTIME_PATH={shlex.quote(str(runtime_path))}
                VIEWER_BACKEND_PID="$VIEWER_RUNTIME_PATH/backend.pid"
                VIEWER_BACKEND_LOG="$VIEWER_RUNTIME_PATH/backend.log"
                VIEWER_BACKEND_PORT=65533
                port_listening() {{ return 1; }}
                wait_for_port() {{ return 0; }}

                start_viewer_backend
                for _attempt in $(seq 1 20); do
                  if [ -f {shlex.quote(str(args_path))} ]; then
                    break
                  fi
                  sleep 0.05
                done
                test -f {shlex.quote(str(args_path))}
                """
            )

            subprocess.run(
                ["bash", "-c", script],
                check=True,
                capture_output=True,
                text=True,
            )
            args = args_path.read_text(encoding="utf-8").splitlines()

        reload_dirs = [
            args[index + 1]
            for index, arg in enumerate(args)
            if arg == "--reload-dir"
        ]
        self.assertIn("--reload", args)
        self.assertEqual(
            reload_dirs,
            [
                str(project_path / "emperor"),
                str(project_path / "models"),
                str(project_path / "viewer" / "backend"),
            ],
        )
        self.assertNotIn(str(project_path), reload_dirs)
        self.assertNotIn(str(project_path / "logs"), reload_dirs)
        self.assertNotIn(str(project_path / "viewer"), reload_dirs)

    def test_backend_start_failure_returns_nonzero_and_removes_pid(self) -> None:
        with tempfile.TemporaryDirectory() as project_dir:
            project_path = Path(project_dir)
            venv_bin = project_path / "torchenv" / "bin"
            runtime_path = project_path / "viewer" / ".runtime"
            venv_bin.mkdir(parents=True)
            runtime_path.mkdir(parents=True)
            python_path = venv_bin / "python"
            python_path.write_text(
                "#!/usr/bin/env bash\necho 'fake uvicorn failure' >&2\nexit 1\n",
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

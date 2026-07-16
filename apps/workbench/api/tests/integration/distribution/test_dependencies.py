from __future__ import annotations

import importlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from tests.integration.distribution._environment_support import (
    REQUIRED_BACKEND_TEST_MODULES,
)
from tests.support import training_jobs as test_helpers

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "emperor-matplotlib"),
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
                "tests.e2e.contract_server",
                [
                    "contract_server",
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
                "tests.e2e.browser_performance_server",
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
                    project_adapter = Mock()
                    with (
                        patch.object(sys, "argv", argv),
                        patch.object(
                            module,
                            "ProjectAdapterClient",
                            return_value=project_adapter,
                        ),
                        patch.object(
                            module,
                            "TrainingJobServiceHarness",
                            return_value=Mock(),
                        ) as training_harness,
                        patch.object(
                            module,
                            "create_app_with_training_service",
                            return_value=object(),
                        ) as create_app,
                        patch.object(module.uvicorn, "run") as run,
                    ):
                        module.main()
                finally:
                    for patcher in reversed(setup_patchers):
                        patcher.stop()
                run.assert_called_once()
                self.assertIs(run.call_args.kwargs["reset_contextvars"], True)
                self.assertIs(
                    training_harness.call_args.kwargs["project_adapter"],
                    project_adapter,
                )
                self.assertIs(
                    create_app.call_args.kwargs["project_adapter"],
                    project_adapter,
                )
                project_adapter.close.assert_called_once_with()

    def test_training_harness_uses_an_explicit_project_adapter_without_fixture_lookup(
        self,
    ) -> None:
        project_adapter = Mock()
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(
                test_helpers,
                "project_adapter_client",
                side_effect=AssertionError("unexpected fixture lookup"),
            ),
        ):
            test_helpers.TrainingJobServiceHarness(
                root=Path(temporary) / "jobs",
                project_adapter=project_adapter,
                runner=test_helpers.FakeRunner(),
            )


if __name__ == "__main__":
    unittest.main()

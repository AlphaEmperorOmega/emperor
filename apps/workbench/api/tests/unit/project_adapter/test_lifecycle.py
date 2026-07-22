from __future__ import annotations

import asyncio
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from emperor_workbench.failures import FailureKind
from emperor_workbench.project_adapter import (
    ModelPackageReference,
    ProjectAdapterClient,
    ProjectAdapterFailure,
)
from emperor_workbench.settings import WorkbenchApiSettings

from ._support import _FakeProcess, _response


def _metadata_payload(runtime_value: object) -> dict[str, object]:
    return {
        "identity": {"model_type": "linears", "model": "linear"},
        "catalog_key": "linears/linear",
        "presets": [],
        "default_experiment_task": "image-classification",
        "dataset_groups": [],
        "monitors": [],
        "runtime_defaults": {"VALUE": runtime_value},
    }


class ProjectAdapterLifecycleTests(unittest.TestCase):
    def test_metadata_cache_is_isolated_per_client(self) -> None:
        first = ProjectAdapterClient(("first",), persistent=False)
        second = ProjectAdapterClient(("second",), persistent=False)
        first.call = Mock(  # type: ignore[method-assign]
            return_value=_metadata_payload(1)
        )
        second.call = Mock(  # type: ignore[method-assign]
            return_value=_metadata_payload(2)
        )
        first_reference = ModelPackageReference("linears", "linear", first)
        second_reference = ModelPackageReference("linears", "linear", second)

        self.assertEqual(
            first_reference.metadata_payload()["runtime_defaults"],
            {"VALUE": 1},
        )
        self.assertEqual(
            first_reference.metadata_payload()["runtime_defaults"],
            {"VALUE": 1},
        )
        self.assertEqual(
            second_reference.metadata_payload()["runtime_defaults"],
            {"VALUE": 2},
        )
        self.assertEqual(first.call.call_count, 1)
        self.assertEqual(second.call.call_count, 1)

    def test_metadata_cache_returns_mutation_isolated_payloads(self) -> None:
        client = ProjectAdapterClient(("adapter",), persistent=False)
        client.call = Mock(  # type: ignore[method-assign]
            return_value=_metadata_payload(1)
        )
        reference = ModelPackageReference("linears", "linear", client)

        first = reference.metadata_payload()
        first["runtime_defaults"]["VALUE"] = 2

        self.assertEqual(
            reference.metadata_payload()["runtime_defaults"],
            {"VALUE": 1},
        )
        self.assertEqual(client.call.call_count, 1)

    def test_persistent_client_restarts_a_dead_process(self) -> None:
        first = _FakeProcess(_response("first"))
        second = _FakeProcess(_response("second"))
        with patch(
            "emperor_workbench.project_adapter._client.subprocess.Popen",
            side_effect=[first, second],
        ) as popen:
            client = ProjectAdapterClient(("adapter",))
            self.assertEqual(client.call("first"), "first")
            first.return_code = 1
            self.assertEqual(client.call("second"), "second")
            client.close()

        self.assertEqual(popen.call_count, 2)
        self.assertTrue(first.stdin.closed)
        first.stdout.close.assert_called_once_with()

    def test_persistent_call_enforces_its_response_timeout(self) -> None:
        process = _FakeProcess()
        release_reader = threading.Event()

        def blocked_read(_limit: int) -> bytes:
            release_reader.wait(2.0)
            return b""

        process.stdout.readline.side_effect = blocked_read
        terminate = process.terminate

        def release_on_terminate() -> None:
            terminate()
            release_reader.set()

        process.terminate = release_on_terminate  # type: ignore[method-assign]
        with (
            patch(
                "emperor_workbench.project_adapter._client.subprocess.Popen",
                return_value=process,
            ),
            self.assertRaises(ProjectAdapterFailure) as raised,
        ):
            ProjectAdapterClient(
                ("adapter",),
                timeout_seconds=0.01,
            ).call("blocked")

        self.assertEqual(raised.exception.kind, FailureKind.TIMEOUT)
        self.assertTrue(process.terminated)

    def test_persistent_timeout_covers_a_blocked_request_write(self) -> None:
        process = _FakeProcess()
        release_writer = threading.Event()

        def blocked_write(_payload: bytes) -> int:
            release_writer.wait(2.0)
            return 0

        process.stdin = Mock()
        process.stdin.write.side_effect = blocked_write
        terminate = process.terminate

        def release_on_terminate() -> None:
            terminate()
            release_writer.set()

        process.terminate = release_on_terminate  # type: ignore[method-assign]
        with (
            patch(
                "emperor_workbench.project_adapter._client.subprocess.Popen",
                return_value=process,
            ),
            self.assertRaises(ProjectAdapterFailure) as raised,
        ):
            ProjectAdapterClient(
                ("adapter",),
                timeout_seconds=0.01,
            ).call("blocked")

        self.assertEqual(raised.exception.kind, FailureKind.TIMEOUT)
        self.assertTrue(process.terminated)

    def test_persistent_response_size_is_bounded_and_discards_process(
        self,
    ) -> None:
        process = _FakeProcess(b"x" * 17)
        with (
            patch(
                "emperor_workbench.project_adapter._client."
                "MAX_PROJECT_ADAPTER_RESPONSE_BYTES",
                16,
            ),
            patch(
                "emperor_workbench.project_adapter._client.subprocess.Popen",
                return_value=process,
            ),
            self.assertRaises(ProjectAdapterFailure) as raised,
        ):
            ProjectAdapterClient(("adapter",)).call("example")

        self.assertEqual(raised.exception.kind, FailureKind.TOO_LARGE)
        self.assertTrue(process.terminated)

    def test_protocol_failure_discards_stale_persistent_responses(self) -> None:
        malformed = _FakeProcess(
            b"debug output\n",
            _response("stale"),
        )
        replacement = _FakeProcess(_response("fresh"))
        with patch(
            "emperor_workbench.project_adapter._client.subprocess.Popen",
            side_effect=[malformed, replacement],
        ):
            client = ProjectAdapterClient(("adapter",))
            with self.assertRaisesRegex(ProjectAdapterFailure, "invalid response"):
                client.call("first")
            self.assertEqual(client.call("second"), "fresh")
            client.close()

        self.assertTrue(malformed.terminated)

    def test_close_is_terminal_and_cannot_race_a_replacement_process(self) -> None:
        client = ProjectAdapterClient(("adapter",))
        client.close()

        with (
            patch(
                "emperor_workbench.project_adapter._client.subprocess.Popen"
            ) as popen,
            self.assertRaisesRegex(ProjectAdapterFailure, "client is closed"),
        ):
            client.call("after-close")

        popen.assert_not_called()

    def test_close_is_idempotent_and_escalates_to_kill(self) -> None:
        process = _FakeProcess(
            _response(None),
            ignore_terminate=True,
        )
        client = ProjectAdapterClient(("adapter",))
        client._process = process  # type: ignore[assignment]

        client.close()
        client.close()

        self.assertTrue(process.terminated)
        self.assertTrue(process.killed)
        self.assertEqual(process.wait_calls, 2)

    def test_each_application_owns_and_closes_its_adapter(self) -> None:
        first = Mock(spec=ProjectAdapterClient)
        second = Mock(spec=ProjectAdapterClient)
        first.catalog.return_value = (
            ModelPackageReference("linears", "linear", first),
        )
        second.catalog.return_value = (
            ModelPackageReference(
                "experts",
                "linear",
                second,
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            import_root = root / "import-defaults"
            with patch.dict(
                os.environ,
                {
                    "WORKBENCH_API_LOGS_ROOT": str(import_root / "logs"),
                    "WORKBENCH_API_SNAPSHOTS_ROOT": str(import_root / "snapshots"),
                    "WORKBENCH_API_STATE_ROOT": str(import_root / "state"),
                    "WORKBENCH_API_TRAINING_CANCELLATION_MODE": "process-group",
                },
            ):
                from emperor_workbench.api import create_app

            def settings(name: str) -> WorkbenchApiSettings:
                app_root = root / name
                return WorkbenchApiSettings(
                    logs_root=str(app_root / "logs"),
                    snapshots_root=str(app_root / "snapshots"),
                    state_root=str(app_root / "state"),
                    trusted_hosts=["testserver"],
                    training_cancellation_mode="process-group",
                )

            first_app = create_app(settings("first"), project_adapter=first)
            second_app = create_app(settings("second"), project_adapter=second)

            async def run_lifespans() -> None:
                async with first_app.router.lifespan_context(first_app):
                    async with second_app.router.lifespan_context(second_app):
                        first_services = first_app.state.workbench_container
                        second_services = second_app.state.workbench_container
                        self.assertIs(first_services.project_adapter, first)
                        self.assertIs(second_services.project_adapter, second)
                        self.assertIsNot(
                            first_services.project_adapter,
                            second_services.project_adapter,
                        )
                        self.assertFalse(
                            hasattr(first_services.inspection, "_project_adapter")
                        )
                        resolver = (
                            first_services.run_history._scanner._model_identity_resolver
                        )
                        barrier = threading.Barrier(16)
                        resolved_models: list[str | None] = [None] * 16

                        def resolve_model(index: int) -> None:
                            barrier.wait()
                            resolved_models[index] = resolver("linears/linear")

                        resolver_threads = [
                            threading.Thread(
                                target=resolve_model,
                                args=(index,),
                            )
                            for index in range(len(resolved_models))
                        ]
                        for thread in resolver_threads:
                            thread.start()
                        for thread in resolver_threads:
                            thread.join()

                        self.assertEqual(
                            resolved_models,
                            ["linears/linear"] * 16,
                        )
                        for index in range(1_000):
                            self.assertIsNone(resolver(f"unknown/{index}"))
                        first.catalog.assert_called_once_with()
                        second.catalog.assert_not_called()
                        self.assertIs(
                            first_services.training_jobs._runtime.run_plans,
                            first_services.training_run_plans,
                        )

            asyncio.run(run_lifespans())

        first.close.assert_called_once_with()
        second.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()

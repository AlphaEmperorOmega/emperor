from __future__ import annotations

import json
import threading
import unittest
from unittest.mock import Mock, patch

from model_runtime.cli import PROTOCOL_VERSION

from emperor_workbench.failures import FailureKind
from emperor_workbench.project_adapter import (
    ModelPackageReference,
    ProjectAdapterClient,
    ProjectAdapterFailure,
)

from ._support import _FakeOneShotProcess, _response


class ProjectAdapterWireTests(unittest.TestCase):
    def test_nonpersistent_call_validates_success_and_process_exit(self) -> None:
        process = _FakeOneShotProcess(_response({"answer": 42}))
        with patch(
            "emperor_workbench.project_adapter._client.subprocess.Popen",
            return_value=process,
        ) as popen:
            result = ProjectAdapterClient(
                ("adapter",),
                persistent=False,
            ).call("example", {"value": 1})

        self.assertEqual(result, {"answer": 42})
        request = json.loads(process.stdin.captured)
        self.assertEqual(request["version"], PROTOCOL_VERSION)
        self.assertEqual(request["operation"], "example")
        self.assertEqual(request["payload"], {"value": 1})
        popen.assert_called_once()

    def test_nonpersistent_call_rejects_malformed_and_incompatible_responses(
        self,
    ) -> None:
        cases = (
            (b"not-json", "invalid response"),
            (
                json.dumps({"version": True, "ok": True, "result": None}).encode(),
                "incompatible response",
            ),
            (
                json.dumps({"version": 1.0, "ok": True, "result": None}).encode(),
                "incompatible response",
            ),
            (
                b'{"version":1,"ok":true,"result":NaN}',
                "invalid response",
            ),
            (
                json.dumps(
                    {"version": PROTOCOL_VERSION + 1, "ok": True, "result": None}
                ).encode(),
                "incompatible response",
            ),
        )
        for stdout, message in cases:
            with self.subTest(message=message):
                with (
                    patch(
                        "emperor_workbench.project_adapter._client.subprocess.Popen",
                        return_value=_FakeOneShotProcess(stdout),
                    ),
                    self.assertRaisesRegex(ProjectAdapterFailure, message),
                ):
                    ProjectAdapterClient(
                        ("adapter",),
                        persistent=False,
                    ).call("example")

    def test_remote_failure_preserves_failure_semantics(self) -> None:
        process = _FakeOneShotProcess(
            json.dumps(
                {
                    "version": PROTOCOL_VERSION,
                    "ok": False,
                    "error": {
                        "message": "remote failure",
                        "kind": FailureKind.CONFLICT.value,
                        "type": "RemoteConflict",
                        "cause": {"message": "remote cause"},
                    },
                }
            ).encode()
        )
        with (
            patch(
                "emperor_workbench.project_adapter._client.subprocess.Popen",
                return_value=process,
            ),
            self.assertRaises(ProjectAdapterFailure) as raised,
        ):
            ProjectAdapterClient(
                ("adapter",),
                persistent=False,
            ).call("example")

        self.assertEqual(raised.exception.kind, FailureKind.CONFLICT)
        self.assertEqual(raised.exception.remote_type, "RemoteConflict")
        self.assertEqual(raised.exception.remote_cause_detail, "remote cause")

    def test_malformed_operation_results_are_unavailable_protocol_failures(
        self,
    ) -> None:
        catalog_client = ProjectAdapterClient(("adapter",), persistent=False)
        catalog_client.call = Mock(  # type: ignore[method-assign]
            return_value=[{"modelType": "linears"}]
        )
        with self.assertRaises(ProjectAdapterFailure) as catalog_failure:
            catalog_client.catalog()
        self.assertEqual(catalog_failure.exception.kind, FailureKind.UNAVAILABLE)

        configuration_client = ProjectAdapterClient(
            ("adapter",),
            persistent=False,
        )
        configuration_client.call = Mock(return_value={})  # type: ignore[method-assign]
        with self.assertRaises(ProjectAdapterFailure) as configuration_failure:
            configuration_client.configuration("linears/linear", "baseline")
        self.assertEqual(
            configuration_failure.exception.kind,
            FailureKind.UNAVAILABLE,
        )

        reference_client = ProjectAdapterClient(("adapter",), persistent=False)
        reference_client.call = Mock(return_value={})  # type: ignore[method-assign]
        reference = ModelPackageReference(
            "linears",
            "linear",
            reference_client,
        )
        calls = (
            lambda: reference.runtime_defaults,
            lambda: reference.resolve_experiment_task(None),
            lambda: reference.resolve_preset("baseline"),
        )
        for call in calls:
            with self.subTest(call=call):
                with self.assertRaises(ProjectAdapterFailure) as raised:
                    call()
                self.assertEqual(raised.exception.kind, FailureKind.UNAVAILABLE)

    def test_timeout_maps_to_timeout_failure(self) -> None:
        release_reader = threading.Event()
        process = _FakeOneShotProcess(b"")

        def blocked_read(_limit: int) -> bytes:
            release_reader.wait(2.0)
            return b""

        process.returncode = None
        process.stdout = Mock()
        process.stdout.read.side_effect = blocked_read
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
                persistent=False,
            ).call("example")

        self.assertEqual(raised.exception.kind, FailureKind.TIMEOUT)

    def test_request_size_is_rejected_before_process_start_in_both_modes(
        self,
    ) -> None:
        for persistent in (False, True):
            with self.subTest(persistent=persistent):
                with (
                    patch(
                        "emperor_workbench.project_adapter._wire."
                        "MAX_PROJECT_ADAPTER_REQUEST_BYTES",
                        16,
                    ),
                    patch(
                        "emperor_workbench.project_adapter._client.subprocess.Popen"
                    ) as popen,
                    self.assertRaises(ProjectAdapterFailure) as raised,
                ):
                    ProjectAdapterClient(
                        ("adapter",),
                        persistent=persistent,
                    ).call("example", {"payload": "too-large"})

                self.assertEqual(raised.exception.kind, FailureKind.TOO_LARGE)
                popen.assert_not_called()

    def test_nonpersistent_response_size_is_bounded(self) -> None:
        process = _FakeOneShotProcess(b"x" * 17)
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
            ProjectAdapterClient(
                ("adapter",),
                persistent=False,
            ).call("example")

        self.assertEqual(raised.exception.kind, FailureKind.TOO_LARGE)

    def test_nonpersistent_pipe_failure_is_unavailable(self) -> None:
        process = _FakeOneShotProcess(b"")
        process.stdin = Mock()
        process.stdin.write.side_effect = BrokenPipeError
        with (
            patch(
                "emperor_workbench.project_adapter._client.subprocess.Popen",
                return_value=process,
            ),
            self.assertRaises(ProjectAdapterFailure) as raised,
        ):
            ProjectAdapterClient(
                ("adapter",),
                persistent=False,
            ).call("example")

        self.assertEqual(raised.exception.kind, FailureKind.UNAVAILABLE)

from __future__ import annotations

import json
import threading
import unittest
from unittest.mock import Mock, patch

from model_runtime.cli import PROTOCOL_VERSION, WireCodecError, run_plan_to_wire
from model_runtime.inspection import InspectionRequest
from model_runtime.packages import ModelIdentity
from model_runtime.runs import (
    PlanningBudget,
    RunParameter,
    RunPlan,
    RunRequest,
    RunResult,
    RunSpec,
)

from emperor_workbench.failures import FailureKind
from emperor_workbench.project_adapter import (
    ModelPackageReference,
    ProjectAdapterClient,
    ProjectAdapterFailure,
)

from ._support import _FakeOneShotProcess, _response


class ProjectAdapterWireTests(unittest.TestCase):
    def test_overdepth_run_plan_is_an_invalid_protocol_result(self) -> None:
        request = RunRequest(presets=("baseline",), datasets=("Mnist",))
        plan = RunPlan(
            identity=ModelIdentity("linears", "linear"),
            presets=request.presets,
            experiment_task="image-classification",
            datasets=request.datasets,
            overrides={},
            search=None,
            runs=(
                RunSpec(
                    id="run-0001",
                    experiment_task="image-classification",
                    preset="baseline",
                    dataset="Mnist",
                    parameters=(RunParameter("HIDDEN_DIM", 64, "override"),),
                ),
            ),
        )
        payload = run_plan_to_wire(plan)
        nested_value: object = 0
        for _ in range(65):
            nested_value = [nested_value]
        payload["runs"][0]["parameters"][0]["value"] = nested_value
        client = ProjectAdapterClient(("adapter",), persistent=False)
        client.call = Mock(  # type: ignore[method-assign]
            return_value={"plan": payload, "random_state": None}
        )

        with self.assertRaises(ProjectAdapterFailure) as raised:
            client.plan_runs(
                "linears/linear",
                request,
                budget=PlanningBudget(),
            )

        self.assertEqual(raised.exception.kind, FailureKind.UNAVAILABLE)
        self.assertIsInstance(raised.exception.__cause__, WireCodecError)

    def test_inspection_memory_limit_crosses_client_protocol_seam(self) -> None:
        client = ProjectAdapterClient(("adapter",), persistent=False)
        client.call = Mock(return_value={})  # type: ignore[method-assign]

        with patch(
            "emperor_workbench.project_adapter._client._decode_wire_result",
            return_value=Mock(),
        ):
            client.inspect(
                "linears/linear",
                InspectionRequest(
                    preset="baseline",
                    overrides={"hidden_dim": 12},
                    dataset="Mnist",
                    experiment_task="image-classification",
                    memory_limit_bytes=768 * 1024**2,
                ),
            )

        client.call.assert_called_once_with(
            "inspect",
            {
                "model_id": "linears/linear",
                "preset": "baseline",
                "overrides": {"hidden_dim": 12},
                "dataset": "Mnist",
                "experiment_task": "image-classification",
                "memory_limit_bytes": 768 * 1024**2,
            },
        )

    def test_model_package_references_require_exact_identity_segments(self) -> None:
        client = ProjectAdapterClient(("adapter",), persistent=False)
        for model_type, model in (
            ("linears", "linear/extra"),
            ("linears", "linear-name"),
            ("linears/path", "linear"),
            (" linears", "linear"),
        ):
            with (
                self.subTest(model_type=model_type, model=model),
                self.assertRaisesRegex(ProjectAdapterFailure, "identity is invalid"),
            ):
                ModelPackageReference(model_type, model, client)

        client.call = Mock()  # type: ignore[method-assign]
        for model_id in ("linear", "linears/linear/extra", "linears/linear-name"):
            with (
                self.subTest(model_id=model_id),
                self.assertRaisesRegex(ProjectAdapterFailure, "Unknown model"),
            ):
                client.package(model_id)
        client.call.assert_not_called()

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
        long_run_id = "run-" + "x" * 1_024
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
                        "phase": "best_results_projection",
                        "affected_run_id": long_run_id,
                        "execution_id": "execution-a",
                        "completed_results": [
                            {
                                "run_id": long_run_id,
                                "experiment_task": "image-classification",
                                "preset": "baseline",
                                "dataset": "Mnist",
                                "log_dir": "logs/run/version_0",
                                "payload": {
                                    "status": "completed",
                                    "artifactId": "attempt-a",
                                },
                            }
                        ],
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
        self.assertEqual(raised.exception.phase, "best_results_projection")
        self.assertEqual(raised.exception.affected_run_id, long_run_id)
        self.assertEqual(raised.exception.execution_id, "execution-a")
        self.assertEqual(
            raised.exception.completed_results,
            (
                RunResult(
                    run_id=long_run_id,
                    experiment_task="image-classification",
                    preset="baseline",
                    dataset="Mnist",
                    log_dir="logs/run/version_0",
                    payload={"status": "completed", "artifactId": "attempt-a"},
                ),
            ),
        )

    def test_partial_run_failure_requires_bounded_consistent_fields(self) -> None:
        completed = {
            "run_id": "run-0001",
            "experiment_task": "image-classification",
            "preset": "baseline",
            "dataset": "Mnist",
            "log_dir": "logs/run/version_0",
            "payload": {"status": "completed", "artifactId": "attempt-a"},
        }
        valid = {
            "message": "remote failure",
            "kind": FailureKind.UNAVAILABLE.value,
            "type": "RunPlanExecutionError",
            "phase": "best_results_projection",
            "affected_run_id": "run-0001",
            "execution_id": "execution-a",
            "completed_results": [completed],
        }
        cases = (
            {**valid, "phase": "unknown"},
            {key: value for key, value in valid.items() if key != "execution_id"},
            {**valid, "completed_results": []},
            {**valid, "phase": "training"},
            {**valid, "affected_run_id": "run-0002"},
            {**valid, "completed_results": [{}]},
            {**valid, "completed_results": [completed] * 2_001},
        )
        for error in cases:
            response = json.dumps(
                {
                    "version": PROTOCOL_VERSION,
                    "ok": False,
                    "error": error,
                }
            ).encode()
            with (
                self.subTest(error=error.get("phase")),
                patch(
                    "emperor_workbench.project_adapter._client.subprocess.Popen",
                    return_value=_FakeOneShotProcess(response),
                ),
                self.assertRaisesRegex(
                    ProjectAdapterFailure,
                    "invalid partial Run failure",
                ),
            ):
                ProjectAdapterClient(("adapter",), persistent=False).call("example")

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

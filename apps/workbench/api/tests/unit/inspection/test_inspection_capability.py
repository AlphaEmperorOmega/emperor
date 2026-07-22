from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from unittest.mock import Mock, patch

import psutil
import torch
from model_runtime.cli import inspection_result_to_wire
from model_runtime.inspection import InspectionRequest, InspectionResult

import emperor_workbench.inspection as inspection_interface
from emperor_workbench.failures import FailureKind
from emperor_workbench.inspection import (
    InProcessInspectionExecutor,
    InspectionFailure,
    InspectionService,
    InspectionWorkerLimits,
    SubprocessInspectionExecutor,
)
from emperor_workbench.inspection._subprocess import _DEFAULT_WORKER_COMMAND
from emperor_workbench.inspection._worker_protocol import (
    decode_worker_request,
    decode_worker_response,
    domain_failure_envelope,
    encode_worker_request,
    success_envelope,
)
from emperor_workbench.model_packages import (
    ModelPackageCatalog,
    SelectedModelPackage,
)
from emperor_workbench.project_adapter import (
    ModelPackageReference,
    ProjectAdapterClient,
)
from emperor_workbench.run_history import (
    HistoricalCheckpointCandidate,
    HistoricalInspectionContext,
)
from tests.support.model_packages import project_adapter_client


class _HistoricalSource:
    def __init__(self, context: HistoricalInspectionContext) -> None:
        self.context = context
        self.run_ids: list[str] = []

    def inspection_context(self, run_id: str) -> HistoricalInspectionContext:
        self.run_ids.append(run_id)
        return self.context


class InspectionCapabilityTests(unittest.TestCase):
    def selected(self) -> SelectedModelPackage:
        return ModelPackageCatalog(project_adapter_client()).select("linears/linear")

    def test_public_interface_is_curated_and_transport_neutral(self) -> None:
        self.assertEqual(
            inspection_interface.__all__,
            [
                "InProcessInspectionExecutor",
                "InspectionExecutor",
                "InspectionFailure",
                "InspectionService",
                "InspectionWorkerLimits",
                "SubprocessInspectionExecutor",
            ],
        )
        service = InspectionService(InProcessInspectionExecutor())
        self.assertFalse(hasattr(service, "inspect_payload"))
        self.assertFalse(hasattr(service, "_project_adapter"))

    def test_service_inspects_one_selected_model_package(self) -> None:
        result = InspectionService(InProcessInspectionExecutor()).inspect(
            self.selected(),
            preset="baseline",
            overrides={"hidden_dim": "12"},
            dataset="Mnist",
        )

        self.assertIsInstance(result, InspectionResult)
        self.assertEqual(result.identity.catalog_key, "linears/linear")
        self.assertGreater(result.parameter_count, 0)

    def test_service_maps_selected_package_failures(self) -> None:
        with self.assertRaises(InspectionFailure) as raised:
            InspectionService(InProcessInspectionExecutor()).inspect(
                self.selected(),
                preset="baseline",
                overrides={"NO_SUCH_FIELD": "1"},
                dataset="Mnist",
            )

        self.assertEqual(raised.exception.kind, FailureKind.INVALID)
        self.assertIn("Unknown override", raised.exception.detail)

    def test_historical_inspection_consumes_only_the_source_interface(self) -> None:
        selected = self.selected()
        source = _HistoricalSource(
            HistoricalInspectionContext(
                run_id="run-1",
                model=selected.catalog_key,
                preset="baseline",
                dataset="Mnist",
                params=MappingProxyType({"hidden_dim": 12}),
                checkpoint_candidates=(),
            )
        )
        service = InspectionService(
            InProcessInspectionExecutor(),
            historical_source=source,
        )

        ordinary = service.inspect(
            selected,
            preset="baseline",
            overrides={},
            dataset="Mnist",
        )
        historical = service.inspect(
            selected,
            preset="baseline",
            overrides={},
            dataset="Mnist",
            log_run_id="run-1",
        )

        self.assertEqual(source.run_ids, ["run-1"])
        self.assertNotEqual(
            ordinary.parameter_count,
            historical.parameter_count,
        )

    def test_historical_inspection_annotates_a_frozen_checkpoint(self) -> None:
        selected = self.selected()
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint_path = Path(temporary) / "epoch=1-step=10.ckpt"
            torch.save(
                {
                    "state_dict": {
                        "input_model.model.weight_params": torch.zeros(784, 12),
                        "input_model.model.bias_params": torch.zeros(12),
                        "main_model.layers.0.model.weight_params": torch.zeros(
                            12,
                            12,
                        ),
                        "main_model.layers.0.model.bias_params": torch.zeros(12),
                        "output_model.model.weight_params": torch.zeros(12, 10),
                        "output_model.model.bias_params": torch.zeros(10),
                    }
                },
                checkpoint_path,
            )
            stat = checkpoint_path.stat()
            source = _HistoricalSource(
                HistoricalInspectionContext(
                    run_id="run-checkpoint",
                    model=selected.catalog_key,
                    preset="baseline",
                    dataset="Mnist",
                    params=MappingProxyType({}),
                    checkpoint_candidates=(
                        HistoricalCheckpointCandidate(
                            path=checkpoint_path,
                            size_bytes=stat.st_size,
                            modified_at_ns=stat.st_mtime_ns,
                        ),
                    ),
                )
            )

            result = InspectionService(
                InProcessInspectionExecutor(),
                historical_source=source,
            ).inspect(
                selected,
                preset="baseline",
                overrides={},
                dataset="Mnist",
                log_run_id="run-checkpoint",
            )

        checkpoint_details = [node.details.get("checkpoint") for node in result.nodes]
        self.assertTrue(
            any(
                isinstance(details, Mapping) and details["status"] == "matched"
                for details in checkpoint_details
            )
        )

    def test_selected_package_owns_checkpoint_metadata_projection(self) -> None:
        client = Mock(spec=ProjectAdapterClient)
        client.call.return_value = {"hidden_dim": 12}
        selected = SelectedModelPackage(
            ModelPackageReference("linears", "linear", client)
        )

        result = selected.checkpoint_config_overrides({"model.weight": (2, 3)})

        self.assertEqual(result, {"hidden_dim": 12})
        client.call.assert_called_once_with(
            "checkpoint_config_overrides",
            {
                "model_id": "linears/linear",
                "tensor_shapes": {"model.weight": [2, 3]},
            },
        )

    def test_worker_request_and_result_protocol_remain_exact(self) -> None:
        selected = self.selected()
        request = InspectionRequest(
            preset="baseline",
            overrides={"hidden_dim": "12"},
            dataset="Mnist",
            experiment_task="image-classification",
        )
        limits = InspectionWorkerLimits(
            memory_bytes=1024**3,
            cpu_count=2,
            timeout_seconds=7.5,
        )

        encoded = encode_worker_request(selected, request, limits)
        payload = json.loads(encoded)
        self.assertEqual(
            list(payload),
            [
                "modelType",
                "model",
                "preset",
                "overrides",
                "dataset",
                "experimentTask",
                "limits",
            ],
        )
        self.assertEqual(
            payload,
            {
                "modelType": "linears",
                "model": "linear",
                "preset": "baseline",
                "overrides": {"HIDDEN_DIM": 12},
                "dataset": "Mnist",
                "experimentTask": "image-classification",
                "limits": {
                    "memoryBytes": 1024**3,
                    "cpuCount": 2,
                },
            },
        )

        malformed_payload = dict(payload)
        malformed_payload["model"] = "linears/linear"
        with self.assertRaisesRegex(ValueError, "model identity"):
            decode_worker_request(malformed_payload)

        result = InspectionService(InProcessInspectionExecutor()).inspect(
            selected,
            preset="baseline",
            overrides={"hidden_dim": "12"},
            dataset="Mnist",
        )
        decoded = decode_worker_response(
            json.dumps(
                success_envelope(result),
                allow_nan=False,
                separators=(",", ":"),
            ).encode()
        )
        self.assertEqual(
            inspection_result_to_wire(decoded),
            inspection_result_to_wire(result),
        )

    def test_worker_domain_and_malformed_envelopes_are_stable(self) -> None:
        expected = InspectionFailure(
            "bad model input",
            kind=FailureKind.CONFLICT,
        )
        with self.assertRaises(InspectionFailure) as raised:
            decode_worker_response(
                json.dumps(domain_failure_envelope(expected)).encode()
            )
        self.assertEqual(raised.exception.detail, "bad model input")
        self.assertEqual(raised.exception.kind, FailureKind.CONFLICT)

        for raw in (
            b"not-json",
            b"[]",
            b'{"ok":true}',
            b'{"ok":false,"detail":1,"failureKind":"invalid"}',
        ):
            with self.subTest(raw=raw):
                with self.assertRaises(InspectionFailure) as malformed:
                    decode_worker_response(raw)
                self.assertEqual(
                    malformed.exception.detail,
                    "Inspection worker produced an invalid result.",
                )
                self.assertEqual(
                    malformed.exception.kind,
                    FailureKind.UNAVAILABLE,
                )

    def test_canonical_worker_target_and_process_failures(self) -> None:
        self.assertEqual(
            _DEFAULT_WORKER_COMMAND,
            (
                sys.executable,
                "-P",
                "-m",
                "emperor_workbench.inspection.worker",
            ),
        )
        selected = self.selected()
        request = InspectionRequest(preset="baseline", dataset="Mnist")

        with self.assertRaises(InspectionFailure) as crashed:
            SubprocessInspectionExecutor(
                InspectionWorkerLimits(timeout_seconds=5),
                command=(sys.executable, "-c", "raise SystemExit(23)"),
            ).inspect(selected, request)
        self.assertEqual(crashed.exception.detail, "Inspection worker crashed.")
        self.assertEqual(crashed.exception.kind, FailureKind.UNAVAILABLE)

        with self.assertRaises(InspectionFailure) as malformed:
            SubprocessInspectionExecutor(
                InspectionWorkerLimits(timeout_seconds=5),
                command=(sys.executable, "-c", "print('not-json')"),
            ).inspect(selected, request)
        self.assertEqual(
            malformed.exception.detail,
            "Inspection worker produced an invalid result.",
        )

        with patch(
            "emperor_workbench.inspection._subprocess.MAX_WORKER_RESULT_BYTES",
            32,
        ):
            with self.assertRaises(InspectionFailure) as oversized:
                SubprocessInspectionExecutor(
                    InspectionWorkerLimits(timeout_seconds=5),
                    command=(
                        sys.executable,
                        "-c",
                        "import sys; sys.stdout.write('x' * 33)",
                    ),
                ).inspect(selected, request)
        self.assertEqual(
            oversized.exception.detail,
            "Inspection worker result exceeded its size limit.",
        )

    def test_timeout_kills_the_worker_process_group(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            child_pid_path = Path(temporary) / "child.pid"
            worker_code = (
                "import pathlib, subprocess, sys, time; "
                "child=subprocess.Popen([sys.executable, '-c', "
                "'import time; time.sleep(60)']); "
                f"pathlib.Path({str(child_pid_path)!r}).write_text(str(child.pid)); "
                "time.sleep(60)"
            )
            executor = SubprocessInspectionExecutor(
                InspectionWorkerLimits(
                    memory_bytes=4 * 1024**3,
                    cpu_count=1,
                    timeout_seconds=0.5,
                ),
                command=(sys.executable, "-c", worker_code),
            )

            started_at = time.monotonic()
            with self.assertRaises(InspectionFailure) as raised:
                executor.inspect(
                    self.selected(),
                    InspectionRequest(preset="baseline", dataset="Mnist"),
                )
            self.assertEqual(raised.exception.kind, FailureKind.TIMEOUT)
            self.assertIn("exceeded the 0.5 second", raised.exception.detail)
            self.assertLess(time.monotonic() - started_at, 2)
            child_pid = int(child_pid_path.read_text(encoding="utf-8"))
            for _attempt in range(40):
                if not psutil.pid_exists(child_pid):
                    break
                time.sleep(0.05)
            self.assertFalse(psutil.pid_exists(child_pid))

    def test_timeout_covers_a_worker_blocked_before_reading_stdin(self) -> None:
        executor = SubprocessInspectionExecutor(
            InspectionWorkerLimits(
                memory_bytes=4 * 1024**3,
                cpu_count=1,
                timeout_seconds=0.25,
            ),
            command=(
                sys.executable,
                "-c",
                "import time; time.sleep(60)",
            ),
        )

        started_at = time.monotonic()
        with (
            patch(
                "emperor_workbench.inspection._subprocess.encode_worker_request",
                return_value=b"x" * (8 * 1024**2),
            ),
            self.assertRaises(InspectionFailure) as raised,
        ):
            executor.inspect(
                self.selected(),
                InspectionRequest(preset="baseline", dataset="Mnist"),
            )

        self.assertEqual(raised.exception.kind, FailureKind.TIMEOUT)
        self.assertLess(time.monotonic() - started_at, 2)

    def test_canonical_worker_returns_semantic_result(self) -> None:
        result = SubprocessInspectionExecutor(
            InspectionWorkerLimits(
                memory_bytes=4 * 1024**3,
                cpu_count=1,
                timeout_seconds=20,
            )
        ).inspect(
            self.selected(),
            InspectionRequest(preset="baseline", dataset="Mnist"),
        )

        self.assertIsInstance(result, InspectionResult)
        self.assertEqual(result.identity.catalog_key, "linears/linear")
        self.assertGreater(result.parameter_count, 0)


if __name__ == "__main__":
    unittest.main()

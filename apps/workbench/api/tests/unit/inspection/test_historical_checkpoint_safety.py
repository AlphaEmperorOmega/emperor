from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import torch

from emperor_workbench.failures import FailureKind
from emperor_workbench.inspection import (
    InProcessInspectionExecutor,
    InspectionFailure,
)
from emperor_workbench.inspection._historical._checkpoint_shapes import (
    MAX_CHECKPOINT_GRAPH_SHAPE_BYTES,
    CheckpointLoadBudgets,
)
from emperor_workbench.inspection._historical._inspection import (
    HistoricalInspection,
)
from emperor_workbench.model_packages import ModelPackageCatalog
from tests.support.inspection import (
    inspect_model,
    inspection_response,
)
from tests.support.model_packages import (
    project_adapter_client,
)
from tests.support.training_jobs import write_tensorboard_run
from tests.unit.inspection._graph_support import (
    nodes_by_id,
)
from tests.unit.inspection._historical_support import (
    checkpoint_state_dict,
)
from tests.unit.inspection._historical_support import (
    first_run_id as _first_run_id,
)
from tests.unit.inspection._historical_support import (
    http_inspection as _http_inspection,
)
from tests.unit.inspection._historical_support import (
    inspection_service as _inspection_service,
)
from tests.unit.inspection._historical_support import (
    run_history as _run_history,
)


class HistoricalInspectionCheckpointSafetyTests(unittest.TestCase):
    def test_historical_checkpoint_invalid_preset_maps_to_workbench_error(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            run_dir = write_tensorboard_run(
                logs_root,
                [
                    "exp_linear",
                    "linears",
                    "linear",
                    "MISSING",
                    "Mnist",
                    "run_20260601_010203",
                    "version_0",
                ],
            )
            torch.save(
                {
                    "state_dict": checkpoint_state_dict(
                        input_dim=8,
                        hidden_dim=16,
                        output_dim=4,
                        layer_count=1,
                    )
                },
                run_dir / "checkpoints" / "epoch=0-step=1.ckpt",
            )
            run_history = _run_history(logs_root)
            run_id = _first_run_id(run_history)

            with self.assertRaisesRegex(
                InspectionFailure,
                "Unknown preset 'missing'",
            ) as raised:
                _http_inspection(
                    _inspection_service(run_history),
                    model_type="linears",
                    model="linear",
                    preset="missing",
                    dataset="Mnist",
                    overrides={},
                    log_run_id=run_id,
                )

        self.assertEqual(raised.exception.kind, FailureKind.INVALID)

    def test_inspection_service_skips_oversized_checkpoint_shape_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = os.path.join(tmp, "logs")
            run_dir = write_tensorboard_run(
                Path(logs_root),
                [
                    "exp_linear",
                    "linears",
                    "linear",
                    "BASELINE",
                    "Cifar10",
                    "run_20260601_010203",
                    "version_0",
                ],
            )
            (run_dir / "result.json").write_text(
                json.dumps(
                    {
                        "params": {
                            "input_dim": 3072,
                            "output_dim": 10,
                            "hidden_dim": 12,
                        }
                    }
                ),
                encoding="utf-8",
            )
            checkpoint_path = run_dir / "checkpoints" / "epoch=0-step=1.ckpt"
            os.truncate(checkpoint_path, MAX_CHECKPOINT_GRAPH_SHAPE_BYTES + 1)
            run_history = _run_history(logs_root)
            run_id = _first_run_id(run_history)
            service = _inspection_service(run_history)

            with mock.patch(
                "emperor_workbench.inspection._historical._checkpoint_shapes.torch.load",
                side_effect=AssertionError("oversized checkpoint was loaded"),
            ) as torch_load:
                historical_result = _http_inspection(
                    service,
                    model_type="linears",
                    model="linear",
                    preset="baseline",
                    dataset="Cifar10",
                    overrides={},
                    log_run_id=run_id,
                )
            expected_result = inspect_model(
                "linears/linear",
                "baseline",
                {
                    "input_dim": 3072,
                    "output_dim": 10,
                    "hidden_dim": 12,
                },
                dataset="Cifar10",
            )

        torch_load.assert_not_called()
        root_checkpoint = nodes_by_id(historical_result["nodes"])["__root__"][
            "details"
        ]["checkpoint"]
        self.assertEqual(
            historical_result["parameterCount"],
            expected_result["parameterCount"],
        )
        self.assertEqual(root_checkpoint["status"], "missing")
        self.assertEqual(root_checkpoint["tensorCount"], 0)
        self.assertEqual(root_checkpoint["reason"], "structuralFallback")
        self.assertTrue(
            any(
                reason.startswith("checkpointTooLarge:")
                for reason in root_checkpoint["fallbackReasons"]
            )
        )

    def test_historical_inspection_bounds_checkpoint_candidate_attempts(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            run_dir = logs_root.joinpath(
                "exp_linear",
                "linears",
                "linear",
                "BASELINE",
                "Mnist",
                "run_20260601_010203",
                "version_0",
            )
            checkpoint_dir = run_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True)
            for index in range(40):
                checkpoint_dir.joinpath(f"epoch=0-step={index}.ckpt").write_bytes(
                    b"not-a-checkpoint"
                )
            run_history = _run_history(logs_root)
            run_id = _first_run_id(run_history)

            with mock.patch(
                "emperor_workbench.inspection._historical._checkpoint_shapes.torch.load",
                side_effect=RuntimeError("malformed checkpoint"),
            ) as torch_load:
                result = _http_inspection(
                    _inspection_service(run_history),
                    model_type="linears",
                    model="linear",
                    preset="baseline",
                    dataset="Mnist",
                    overrides={},
                    log_run_id=run_id,
                )

        self.assertEqual(torch_load.call_count, 32)
        root_checkpoint = nodes_by_id(result["nodes"])["__root__"]["details"][
            "checkpoint"
        ]
        self.assertEqual(
            root_checkpoint["fallbackReasons"],
            ["checkpointCandidateLimit:32"],
        )

    def test_historical_inspection_bounds_aggregate_checkpoint_bytes(
        self,
    ) -> None:
        malformed_checkpoint = b"bad-checkpoint"
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            run_dir = logs_root.joinpath(
                "exp_linear",
                "linears",
                "linear",
                "BASELINE",
                "Mnist",
                "run_20260601_010203",
                "version_0",
            )
            checkpoint_dir = run_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True)
            for index in range(2):
                checkpoint_dir.joinpath(f"epoch=0-step={index}.ckpt").write_bytes(
                    malformed_checkpoint
                )
            run_history = _run_history(logs_root)
            run_id = _first_run_id(run_history)

            with mock.patch(
                "emperor_workbench.inspection._historical._checkpoint_shapes.torch.load",
                side_effect=RuntimeError("malformed checkpoint"),
            ) as torch_load:
                selected = ModelPackageCatalog(project_adapter_client()).select(
                    "linears/linear"
                )
                semantic_result = HistoricalInspection(
                    selected,
                    executor=InProcessInspectionExecutor(),
                    source=run_history,
                    checkpoint_budgets=CheckpointLoadBudgets(
                        max_candidates=4,
                        max_file_bytes=64,
                        max_aggregate_bytes=len(malformed_checkpoint),
                    ),
                ).inspect(
                    log_run_id=run_id,
                    preset="baseline",
                    request_overrides={},
                    dataset="Mnist",
                )
                result = inspection_response(semantic_result).model_dump(mode="json")

        self.assertEqual(torch_load.call_count, 1)
        root_checkpoint = nodes_by_id(result["nodes"])["__root__"]["details"][
            "checkpoint"
        ]
        self.assertIn(
            "checkpointAggregateTooLarge:",
            root_checkpoint["fallbackReasons"][0],
        )

    def test_historical_inspection_rejects_a_checkpoint_changed_after_context(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            run_dir = write_tensorboard_run(
                logs_root,
                [
                    "exp_linear",
                    "linears",
                    "linear",
                    "BASELINE",
                    "Mnist",
                    "run_20260601_010203",
                    "version_0",
                ],
            )
            run_history = _run_history(logs_root)
            run_id = _first_run_id(run_history)
            context = run_history.inspection_context(run_id)
            checkpoint = run_dir / "checkpoints" / "epoch=0-step=1.ckpt"
            checkpoint.write_bytes(checkpoint.read_bytes() + b"changed")
            frozen_source = mock.Mock()
            frozen_source.inspection_context.return_value = context

            with mock.patch(
                "emperor_workbench.inspection._historical._checkpoint_shapes.torch.load",
                side_effect=AssertionError("changed checkpoint was loaded"),
            ) as torch_load:
                result = _http_inspection(
                    _inspection_service(frozen_source),
                    model_type="linears",
                    model="linear",
                    preset="baseline",
                    dataset="Mnist",
                    overrides={},
                    log_run_id=run_id,
                )

        torch_load.assert_not_called()
        root_checkpoint = nodes_by_id(result["nodes"])["__root__"]["details"][
            "checkpoint"
        ]
        self.assertEqual(root_checkpoint["reason"], "structuralFallback")
        self.assertEqual(
            root_checkpoint["fallbackReasons"],
            ["checkpointChanged"],
        )

    def test_historical_inspection_uses_package_checkpoint_metadata_extension(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            run_dir = write_tensorboard_run(
                logs_root,
                [
                    "exp_linear",
                    "linears",
                    "linear",
                    "BASELINE",
                    "Mnist",
                    "run_20260601_010203",
                    "version_0",
                ],
            )
            torch.save(
                {
                    "state_dict": checkpoint_state_dict(
                        input_dim=8,
                        hidden_dim=16,
                        output_dim=4,
                        layer_count=1,
                    )
                },
                run_dir / "checkpoints" / "epoch=0-step=1.ckpt",
            )
            run_history = _run_history(logs_root)
            run_id = _first_run_id(run_history)

            with mock.patch(
                "emperor_workbench.project_adapter.ModelPackageReference."
                "checkpoint_config_overrides",
                return_value={"hidden_dim": 40},
            ) as interpret_checkpoint:
                result = _http_inspection(
                    _inspection_service(run_history),
                    model_type="linears",
                    model="linear",
                    preset="baseline",
                    dataset="Mnist",
                    overrides={},
                    log_run_id=run_id,
                )
            with mock.patch(
                "emperor_workbench.project_adapter.ModelPackageReference."
                "checkpoint_config_overrides",
                side_effect=RuntimeError("invalid package metadata"),
            ):
                fallback_result = _http_inspection(
                    _inspection_service(run_history),
                    model_type="linears",
                    model="linear",
                    preset="baseline",
                    dataset="Mnist",
                    overrides={},
                    log_run_id=run_id,
                )

        interpret_checkpoint.assert_called_once()
        tensor_shapes = interpret_checkpoint.call_args.args[0]
        self.assertEqual(
            tensor_shapes["input_model.model.weight_params"],
            (8, 16),
        )
        node_by_id = nodes_by_id(result["nodes"])
        self.assertEqual(node_by_id["input_model"]["details"]["dims"], "8 -> 40")
        self.assertEqual(
            node_by_id["input_model.model"]["details"]["weightShape"],
            "8 x 16",
        )
        fallback_nodes = nodes_by_id(fallback_result["nodes"])
        self.assertEqual(
            fallback_nodes["input_model"]["details"]["dims"],
            "8 -> 16",
        )
        self.assertEqual(
            fallback_nodes["__root__"]["details"]["checkpoint"]["fallbackReasons"],
            ["packageCheckpointMetadataUnavailable"],
        )

    def test_historical_inspection_ignores_a_malformed_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            run_dir = write_tensorboard_run(
                logs_root,
                [
                    "exp_linear",
                    "linears",
                    "linear",
                    "BASELINE",
                    "Mnist",
                    "run_20260601_010203",
                    "version_0",
                ],
            )
            (run_dir / "result.json").write_text(
                json.dumps({"params": {"hidden_dim": 12}}),
                encoding="utf-8",
            )
            (run_dir / "checkpoints" / "epoch=0-step=1.ckpt").write_bytes(
                b"not a checkpoint"
            )
            run_history = _run_history(logs_root)
            run_id = _first_run_id(run_history)
            result = _http_inspection(
                _inspection_service(run_history),
                model_type="linears",
                model="linear",
                preset="baseline",
                dataset="Mnist",
                overrides={},
                log_run_id=run_id,
            )

        expected = inspect_model(
            "linears/linear",
            "baseline",
            {"hidden_dim": 12},
            dataset="Mnist",
        )
        self.assertEqual(result["parameterCount"], expected["parameterCount"])
        node_by_id = nodes_by_id(result["nodes"])
        self.assertEqual(node_by_id["input_model"]["details"]["dims"], "784 -> 12")
        self.assertTrue(
            all("checkpoint" not in node["details"] for node in result["nodes"])
        )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import torch
from model_runtime.inspection import (
    InspectionResult,
)

from emperor_workbench.inspection import (
    InspectionFailure,
)
from tests.support.inspection import (
    inspect_model,
    inspection_response,
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
from tests.unit.inspection._historical_support import (
    semantic_inspection as _semantic_inspection,
)


class HistoricalInspectionConfigurationTests(unittest.TestCase):
    def test_inspection_service_uses_saved_log_run_params(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = os.path.join(tmp, "logs")
            run_dir = write_tensorboard_run(
                Path(logs_root),
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
            run_history = _run_history(logs_root)
            run_id = _first_run_id(run_history)
            service = _inspection_service(run_history)

            default_result = _http_inspection(
                service,
                model_type="linears",
                model="linear",
                preset="baseline",
                dataset="Mnist",
                overrides={},
            )
            historical_semantic_result = _semantic_inspection(
                service,
                model_type="linears",
                model="linear",
                preset="baseline",
                dataset="Mnist",
                overrides={},
                log_run_id=run_id,
            )
            historical_result = inspection_response(
                historical_semantic_result
            ).model_dump(mode="json")
            expected_result = inspect_model(
                "linears/linear",
                "baseline",
                {"hidden_dim": 12},
                dataset="Mnist",
            )

        self.assertNotEqual(
            historical_result["parameterCount"],
            default_result["parameterCount"],
        )
        self.assertEqual(
            historical_result["parameterCount"],
            expected_result["parameterCount"],
        )

    def test_inspection_service_uses_checkpoint_shapes_for_log_run_graph(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = os.path.join(tmp, "logs")
            run_dir = write_tensorboard_run(
                Path(logs_root),
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
                json.dumps(
                    {
                        "params": {
                            "input_dim": 20,
                            "output_dim": 9,
                            "hidden_dim": 12,
                            "stack_num_layers": 4,
                        },
                        "metrics": {},
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "checkpoints" / "last.ckpt").write_text(
                "placeholder",
                encoding="utf-8",
            )
            torch.save(
                {
                    "state_dict": checkpoint_state_dict(
                        input_dim=20,
                        hidden_dim=12,
                        output_dim=9,
                        layer_count=4,
                    )
                },
                run_dir / "checkpoints" / "epoch=0-step=1.ckpt",
            )
            torch.save(
                {
                    "state_dict": checkpoint_state_dict(
                        input_dim=14,
                        hidden_dim=32,
                        output_dim=6,
                        layer_count=3,
                    )
                },
                run_dir / "checkpoints" / "epoch=2-step=300.ckpt",
            )
            run_history = _run_history(logs_root)
            run_id = _first_run_id(run_history)
            service = _inspection_service(run_history)

            preset_result = _http_inspection(
                service,
                model_type="linears",
                model="linear",
                preset="baseline",
                dataset="Mnist",
                overrides={},
            )
            historical_semantic_result = _semantic_inspection(
                service,
                model_type="linears",
                model="linear",
                preset="baseline",
                dataset="Mnist",
                overrides={},
                log_run_id=run_id,
            )
            historical_result = inspection_response(
                historical_semantic_result
            ).model_dump(mode="json")
            checkpoint_result = inspect_model(
                "linears/linear",
                "baseline",
                {
                    "input_dim": 14,
                    "output_dim": 6,
                    "hidden_dim": 32,
                    "stack_num_layers": 3,
                },
                dataset="Mnist",
            )
            stale_result = inspect_model(
                "linears/linear",
                "baseline",
                {
                    "input_dim": 20,
                    "output_dim": 9,
                    "hidden_dim": 12,
                    "stack_num_layers": 4,
                },
                dataset="Mnist",
            )

        node_by_id = nodes_by_id(historical_result["nodes"])

        self.assertIsInstance(historical_semantic_result, InspectionResult)
        semantic_input = next(
            node
            for node in historical_semantic_result.nodes
            if node.path == "input_model.model"
        )
        self.assertEqual(semantic_input.details["weight_shape"], "14 x 32")
        self.assertEqual(
            semantic_input.details["checkpoint"]["tensor_count"],
            2,
        )

        self.assertEqual(
            preset_result,
            inspect_model("linears/linear", "baseline", {}, dataset="Mnist"),
        )
        self.assertEqual(
            historical_result["parameterCount"],
            checkpoint_result["parameterCount"],
        )
        self.assertNotEqual(
            historical_result["parameterCount"],
            stale_result["parameterCount"],
        )
        self.assertEqual(node_by_id["input_model"]["details"]["dims"], "14 -> 32")
        self.assertEqual(node_by_id["main_model"]["details"]["numLayers"], 3)
        self.assertEqual(
            node_by_id["main_model.layers.0"]["details"]["dims"],
            "32 -> 32",
        )
        self.assertEqual(node_by_id["output_model"]["details"]["dims"], "32 -> 6")
        self.assertNotIn("main_model.layers.3", node_by_id)
        self.assertEqual(
            node_by_id["input_model.model"]["details"]["weightShape"],
            "14 x 32",
        )
        self.assertEqual(
            node_by_id["input_model.model"]["details"]["biasShape"],
            "32",
        )
        self.assertEqual(
            node_by_id["main_model.layers.2.model"]["details"]["weightShape"],
            "32 x 32",
        )
        self.assertEqual(
            node_by_id["output_model.model"]["details"]["weightShape"],
            "32 x 6",
        )
        self.assertEqual(
            node_by_id["output_model.model"]["details"]["biasShape"],
            "6",
        )

    def test_inspection_service_enables_safe_checkpoint_controller_structure(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = os.path.join(tmp, "logs")
            run_dir = write_tensorboard_run(
                Path(logs_root),
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
            state_dict = checkpoint_state_dict(
                input_dim=8,
                hidden_dim=16,
                output_dim=4,
                layer_count=2,
            )
            for outer_index in range(2):
                for gate_index in range(2):
                    state_dict[
                        "main_model.layers."
                        f"{outer_index}.gate_model.model.layers.{gate_index}."
                        "model.weight_params"
                    ] = torch.zeros(16, 16)
            torch.save(
                {"state_dict": state_dict},
                run_dir / "checkpoints" / "epoch=0-step=1.ckpt",
            )
            run_history = _run_history(logs_root)
            run_id = _first_run_id(run_history)
            service = _inspection_service(run_history)

            result = _http_inspection(
                service,
                model_type="linears",
                model="linear",
                preset="baseline",
                dataset="Mnist",
                overrides={},
                log_run_id=run_id,
            )

        node_by_id = nodes_by_id(result["nodes"])
        self.assertEqual(node_by_id["main_model"]["details"]["numLayers"], 2)
        self.assertTrue(node_by_id["main_model.layers.0"]["details"]["gate"])
        self.assertIn("main_model.layers.0.gate_model", node_by_id)
        self.assertEqual(
            node_by_id["__root__"]["details"]["checkpoint"],
            {"status": "matched", "tensorCount": len(state_dict)},
        )
        self.assertEqual(
            node_by_id["input_model"]["details"]["checkpoint"]["status"],
            "matched",
        )
        self.assertEqual(
            node_by_id["loss_fn"]["details"]["checkpoint"],
            {
                "status": "missing",
                "tensorCount": 0,
                "reason": "noCheckpointTensor",
            },
        )

    def test_inspection_service_request_overrides_beat_checkpoint_structure(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = os.path.join(tmp, "logs")
            run_dir = write_tensorboard_run(
                Path(logs_root),
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
                        layer_count=3,
                    )
                },
                run_dir / "checkpoints" / "epoch=0-step=1.ckpt",
            )
            run_history = _run_history(logs_root)
            run_id = _first_run_id(run_history)
            service = _inspection_service(run_history)

            result = _http_inspection(
                service,
                model_type="linears",
                model="linear",
                preset="baseline",
                dataset="Mnist",
                overrides={"hidden_dim": 32, "stack_num_layers": 1},
                log_run_id=run_id,
            )

        node_by_id = nodes_by_id(result["nodes"])
        self.assertEqual(node_by_id["main_model"]["details"]["numLayers"], 1)
        self.assertNotIn("main_model.layers.1", node_by_id)
        layer_details = node_by_id["main_model.layers.0.model"]["details"]
        self.assertEqual(layer_details["weightShape"], "16 x 16")
        self.assertEqual(layer_details["biasShape"], "16")
        self.assertEqual(layer_details["inputDim"], 16)
        self.assertEqual(layer_details["outputDim"], 16)
        self.assertEqual(layer_details["dims"], "16 -> 16")

    def test_inspection_service_uses_checkpoint_dims_for_controller_linear_node(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = os.path.join(tmp, "logs")
            run_dir = write_tensorboard_run(
                Path(logs_root),
                [
                    "exp_linear",
                    "linears",
                    "linear",
                    "GATING",
                    "Mnist",
                    "run_20260601_010203",
                    "version_0",
                ],
            )
            (run_dir / "result.json").write_text(
                json.dumps(
                    {
                        "params": {
                            "hidden_dim": 32,
                            "stack_num_layers": 1,
                            "submodule_stack_hidden_dim": 256,
                        }
                    }
                ),
                encoding="utf-8",
            )
            state_dict = {
                "main_model.layers.0.gate_model.model.layers.0."
                "model.weight_params": torch.zeros(32, 32),
                "main_model.layers.0.gate_model.model.layers.0."
                "model.bias_params": torch.zeros(32),
                "main_model.layers.0.gate_model.model.layers.1."
                "model.weight_params": torch.zeros(32, 32),
                "main_model.layers.0.gate_model.model.layers.1."
                "model.bias_params": torch.zeros(32),
            }
            torch.save(
                {"state_dict": state_dict},
                run_dir / "checkpoints" / "epoch=0-step=1.ckpt",
            )
            run_history = _run_history(logs_root)
            run_id = _first_run_id(run_history)
            service = _inspection_service(run_history)

            result = _http_inspection(
                service,
                model_type="linears",
                model="linear",
                preset="gating",
                dataset="Mnist",
                overrides={},
                log_run_id=run_id,
            )

        node_by_id = nodes_by_id(result["nodes"])
        gate_details = node_by_id[
            "main_model.layers.0.gate_model.model.layers.0.model"
        ]["details"]
        self.assertEqual(gate_details["weightShape"], "32 x 32")
        self.assertEqual(gate_details["biasShape"], "32")
        self.assertEqual(gate_details["inputDim"], 32)
        self.assertEqual(gate_details["outputDim"], 32)
        self.assertEqual(gate_details["dims"], "32 -> 32")

    def test_inspection_service_marks_locked_checkpoint_override_fallback(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = os.path.join(tmp, "logs")
            run_dir = write_tensorboard_run(
                Path(logs_root),
                [
                    "exp_linear",
                    "linears",
                    "linear",
                    "GATING",
                    "Mnist",
                    "run_20260601_010203",
                    "version_0",
                ],
            )
            state_dict = checkpoint_state_dict(
                input_dim=8,
                hidden_dim=16,
                output_dim=4,
                layer_count=1,
            )
            state_dict[
                "main_model.layers.0.gate_model.model.layers.0.model.weight_params"
            ] = torch.zeros(16, 16)
            torch.save(
                {"state_dict": state_dict},
                run_dir / "checkpoints" / "epoch=0-step=1.ckpt",
            )
            run_history = _run_history(logs_root)
            run_id = _first_run_id(run_history)
            service = _inspection_service(run_history)

            result = _http_inspection(
                service,
                model_type="linears",
                model="linear",
                preset="gating",
                dataset="Mnist",
                overrides={},
                log_run_id=run_id,
            )

        root = nodes_by_id(result["nodes"])["__root__"]
        self.assertEqual(
            root["details"]["checkpoint"],
            {
                "status": "matched",
                "tensorCount": len(state_dict),
                "reason": "structuralFallback",
            },
        )

    def test_inspection_service_ignores_unknown_saved_log_run_params(self) -> None:
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
                            "gather_frequency_flag": False,
                        }
                    }
                ),
                encoding="utf-8",
            )
            run_history = _run_history(logs_root)
            run_id = _first_run_id(run_history)
            service = _inspection_service(run_history)

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

        self.assertEqual(
            historical_result["parameterCount"],
            expected_result["parameterCount"],
        )

    def test_inspection_service_keeps_request_overrides_strict_with_log_run(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = os.path.join(tmp, "logs")
            write_tensorboard_run(
                Path(logs_root),
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
            service = _inspection_service(run_history)

            with self.assertRaises(InspectionFailure) as raised:
                _http_inspection(
                    service,
                    model_type="linears",
                    model="linear",
                    preset="baseline",
                    dataset="Mnist",
                    overrides={"gather_frequency_flag": False},
                    log_run_id=run_id,
                )

        self.assertIn(
            "Unknown override 'gather_frequency_flag'",
            raised.exception.detail,
        )

    def test_inspection_service_rejects_mismatched_log_run_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = os.path.join(tmp, "logs")
            write_tensorboard_run(
                Path(logs_root),
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
            service = _inspection_service(run_history)

            with self.assertRaises(InspectionFailure) as raised:
                _http_inspection(
                    service,
                    model_type="linears",
                    model="linear",
                    preset="baseline",
                    dataset="Cifar10",
                    overrides={},
                    log_run_id=run_id,
                )

        self.assertIn("belongs to dataset 'Mnist'", raised.exception.detail)

    def test_inspection_service_preserves_preset_inspect_without_log_run(self) -> None:
        service = _inspection_service()

        self.assertEqual(
            _http_inspection(
                service,
                model_type="linears",
                model="linear",
                preset="baseline",
                dataset="Mnist",
                overrides={},
            ),
            inspect_model("linears/linear", "baseline", {}, dataset="Mnist"),
        )

    def test_historical_inspection_freezes_saved_checkpoint_request_precedence(
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
            (run_dir / "result.json").write_text(
                json.dumps({"params": {"hidden_dim": 12, "stack_num_layers": 4}}),
                encoding="utf-8",
            )
            torch.save(
                {
                    "state_dict": checkpoint_state_dict(
                        input_dim=784,
                        hidden_dim=32,
                        output_dim=10,
                        layer_count=2,
                    )
                },
                run_dir / "checkpoints" / "epoch=2-step=300.ckpt",
            )
            run_history = _run_history(logs_root)
            run_id = _first_run_id(run_history)
            result = _http_inspection(
                _inspection_service(run_history),
                model_type="linears",
                model="linear",
                preset="baseline",
                dataset="Mnist",
                overrides={"hidden_dim": 48},
                log_run_id=run_id,
            )
            expected = inspect_model(
                "linears/linear",
                "baseline",
                {"hidden_dim": 48, "stack_num_layers": 2},
                dataset="Mnist",
            )

        node_by_id = nodes_by_id(result["nodes"])
        self.assertEqual(result["parameterCount"], expected["parameterCount"])
        self.assertEqual(node_by_id["input_model"]["details"]["dims"], "784 -> 48")
        self.assertEqual(node_by_id["main_model"]["details"]["numLayers"], 2)
        self.assertEqual(
            node_by_id["input_model.model"]["details"]["weightShape"],
            "784 x 32",
        )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import os
import unittest

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from model_runtime.inspection import InspectionRequest, inspect_model_shapes
from models.catalog import model_package


class InspectionShapeTraceTests(unittest.TestCase):
    def test_shape_trace_executes_a_synthetic_input_for_every_experiment_task(
        self,
    ) -> None:
        cases = (
            ("linears/linear", ((1, 1, 28, 28),)),
            ("vit/linear", ((1, 1, 28, 28),)),
            ("mlp_mixer/linear", ((1, 1, 28, 28),)),
            ("bert/linear", ((1, 35),)),
            ("gpt/linear", ((1, 35),)),
            ("transformer/linear", ((1, 64), (1, 63))),
        )

        for model_id, expected_inputs in cases:
            with self.subTest(model=model_id):
                package = model_package(model_id)
                self.assertIsNotNone(package)
                assert package is not None

                result, trace = inspect_model_shapes(
                    package,
                    InspectionRequest(preset="baseline"),
                )

                self.assertEqual(result.identity, package.identity)
                self.assertEqual(
                    tuple(tensor.shape for tensor in trace.sample_inputs),
                    expected_inputs,
                )
                self.assertEqual(trace.batch_size, 1)
                self.assertEqual(trace.modules[0].node_id, "__root__")
                self.assertEqual(len(trace.modules[0].calls), 1)
                self.assertTrue(trace.modules[0].calls[0].outputs)
                self.assertEqual(trace.methods, ())

    def test_variable_trace_records_same_shape_reassignments_and_source_locations(
        self,
    ) -> None:
        package = model_package("linears/linear")
        assert package is not None

        _result, trace = inspect_model_shapes(
            package,
            InspectionRequest(
                preset="baseline",
                overrides={"stack_num_layers": 1},
            ),
            detail="variables",
        )

        model_forward = next(
            method
            for method in trace.methods
            if method.parent_id is None and method.qualified_name == "Model.forward"
        )
        self.assertEqual(model_forward.module_path, "model")
        self.assertEqual(model_forward.source_path, "models/linears/linear/model.py")
        self.assertEqual(model_forward.inputs[0].shape, (1, 1, 28, 28))
        model_variables = {
            (tensor.name, tensor.shape)
            for variable in model_forward.variables
            for tensor in variable.tensors
        }
        self.assertIn(("X", (1, 784)), model_variables)
        self.assertIn(("X", (1, 32)), model_variables)
        self.assertIn(("logits", (1, 10)), model_variables)

        layer_forward = next(
            method
            for method in trace.methods
            if method.qualified_name == "Layer.forward"
            and method.module_path == "main_model.layers.0"
        )
        same_shape_x_assignments = [
            tensor
            for variable in layer_forward.variables
            for tensor in variable.tensors
            if tensor.name == "X" and tensor.shape == (1, 32)
        ]
        self.assertGreaterEqual(len(same_shape_x_assignments), 2)
        self.assertTrue(
            all(
                method.source_path.startswith(("models/", "emperor/"))
                for method in trace.methods
            )
        )

    def test_recurrent_modules_report_each_execution(self) -> None:
        package = model_package("linears/linear")
        assert package is not None

        _result, trace = inspect_model_shapes(
            package,
            InspectionRequest(preset="recurrent"),
        )
        modules = {module.node_id: module for module in trace.modules}

        self.assertEqual(len(modules["main_model.block_model"].calls), 4)
        self.assertEqual(len(modules["main_model.block_model.layers.0"].calls), 4)


if __name__ == "__main__":
    unittest.main()

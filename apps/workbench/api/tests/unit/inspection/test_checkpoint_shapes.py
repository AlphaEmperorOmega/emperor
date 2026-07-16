from __future__ import annotations

import os
import unittest

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import torch

from emperor_workbench.inspection._historical._checkpoint_shapes import (
    checkpoint_graph_shapes_from_state_dict,
)
from tests.unit.inspection._historical_support import (
    checkpoint_state_dict,
)


class CheckpointShapeExtractionTests(unittest.TestCase):
    def test_checkpoint_shape_extractor_counts_recurrent_block_layers(self) -> None:
        checkpoint_shapes = checkpoint_graph_shapes_from_state_dict(
            checkpoint_state_dict(
                input_dim=8,
                hidden_dim=16,
                output_dim=4,
                layer_count=2,
                stack_prefix="main_model.block_model.layers",
            )
        )

        self.assertIsNotNone(checkpoint_shapes)
        if checkpoint_shapes is None:
            self.fail("Expected checkpoint shape extraction to succeed.")
        self.assertEqual(
            checkpoint_shapes.config_overrides,
            {
                "input_dim": 8,
                "output_dim": 4,
                "hidden_dim": 16,
                "stack_num_layers": 2,
            },
        )

    def test_checkpoint_shape_extractor_infers_optional_controller_counts(
        self,
    ) -> None:
        state_dict = checkpoint_state_dict(
            input_dim=8,
            hidden_dim=16,
            output_dim=4,
            layer_count=2,
        )
        for outer_index in range(2):
            for gate_index in range(3):
                state_dict[
                    "main_model.layers."
                    f"{outer_index}.gate_model.model.layers.{gate_index}."
                    "model.weight_params"
                ] = torch.zeros(16, 16)
            for halting_index in range(4):
                state_dict[
                    "main_model.layers."
                    f"{outer_index}.halting_model.halting_gate_model.layers."
                    f"{halting_index}.model.weight_params"
                ] = torch.zeros(16, 2 if halting_index == 3 else 16)
            for memory_index in range(2):
                state_dict[
                    "main_model.layers."
                    f"{outer_index}.memory_model.memory_model.layers."
                    f"{memory_index}.model.weight_params"
                ] = torch.zeros(16, 16)

        checkpoint_shapes = checkpoint_graph_shapes_from_state_dict(state_dict)

        self.assertIsNotNone(checkpoint_shapes)
        if checkpoint_shapes is None:
            self.fail("Expected checkpoint shape extraction to succeed.")
        self.assertEqual(checkpoint_shapes.config_overrides["stack_num_layers"], 2)
        self.assertTrue(checkpoint_shapes.config_overrides["stack_gate_flag"])
        self.assertEqual(
            checkpoint_shapes.config_overrides["gate_stack_num_layers"],
            3,
        )
        self.assertTrue(checkpoint_shapes.config_overrides["stack_halting_flag"])
        self.assertEqual(
            checkpoint_shapes.config_overrides["halting_stack_num_layers"],
            4,
        )
        self.assertTrue(checkpoint_shapes.config_overrides["memory_flag"])
        self.assertEqual(
            checkpoint_shapes.config_overrides["memory_stack_num_layers"],
            2,
        )

    def test_checkpoint_shape_extractor_maps_boundary_generator_counts_to_global(
        self,
    ) -> None:
        state_dict = checkpoint_state_dict(
            input_dim=8,
            hidden_dim=16,
            output_dim=4,
            layer_count=2,
        )
        for prefix in (
            "input_model.model.adaptive_behaviour.weight_generator.model",
            "output_model.model.adaptive_behaviour.weight_generator.model",
        ):
            for layer_index in range(3):
                state_dict[f"{prefix}.layers.{layer_index}.model.weight_params"] = (
                    torch.zeros(16, 16)
                )

        checkpoint_shapes = checkpoint_graph_shapes_from_state_dict(state_dict)

        self.assertIsNotNone(checkpoint_shapes)
        if checkpoint_shapes is None:
            self.fail("Expected checkpoint shape extraction to succeed.")
        self.assertEqual(
            checkpoint_shapes.config_overrides["adaptive_generator_stack_num_layers"],
            3,
        )
        self.assertEqual(
            [
                key
                for key in checkpoint_shapes.config_overrides
                if key.endswith("_layer_adaptive_generator_stack_num_layers")
            ],
            [],
        )

    def test_checkpoint_shape_extractor_infers_controller_hidden_dims_safely(
        self,
    ) -> None:
        state_dict = checkpoint_state_dict(
            input_dim=8,
            hidden_dim=16,
            output_dim=4,
            layer_count=2,
        )
        for outer_index in range(2):
            state_dict[
                "main_model.layers."
                f"{outer_index}.gate_model.model.layers.0.model.weight_params"
            ] = torch.zeros(16, 32)
            state_dict[
                "main_model.layers."
                f"{outer_index}.gate_model.model.layers.1.model.weight_params"
            ] = torch.zeros(32, 16)
            state_dict[
                "main_model.layers."
                f"{outer_index}.memory_model.memory_model.layers.0."
                "model.weight_params"
            ] = torch.zeros(16, 32)
            state_dict[
                "main_model.layers."
                f"{outer_index}.memory_model.memory_model.layers.1."
                "model.weight_params"
            ] = torch.zeros(32, 16)
            state_dict[
                "main_model.layers."
                f"{outer_index}.halting_model.halting_gate_model.layers.0."
                "model.weight_params"
            ] = torch.zeros(16, 24)
            state_dict[
                "main_model.layers."
                f"{outer_index}.halting_model.halting_gate_model.layers.2."
                "model.weight_params"
            ] = torch.zeros(24, 2)

        checkpoint_shapes = checkpoint_graph_shapes_from_state_dict(state_dict)

        self.assertIsNotNone(checkpoint_shapes)
        if checkpoint_shapes is None:
            self.fail("Expected checkpoint shape extraction to succeed.")
        self.assertTrue(
            checkpoint_shapes.config_overrides["gate_stack_independent_flag"]
        )
        self.assertEqual(
            checkpoint_shapes.config_overrides["gate_stack_hidden_dim"],
            32,
        )
        self.assertEqual(checkpoint_shapes.config_overrides["gate_stack_num_layers"], 2)
        self.assertTrue(
            checkpoint_shapes.config_overrides["memory_stack_independent_flag"]
        )
        self.assertEqual(
            checkpoint_shapes.config_overrides["memory_stack_hidden_dim"],
            32,
        )
        self.assertEqual(
            checkpoint_shapes.config_overrides["memory_stack_num_layers"],
            2,
        )
        self.assertEqual(
            checkpoint_shapes.config_overrides["submodule_stack_hidden_dim"],
            32,
        )
        self.assertNotIn(
            "halting_stack_hidden_dim",
            checkpoint_shapes.config_overrides,
        )
        self.assertNotIn(
            "halting_stack_num_layers",
            checkpoint_shapes.config_overrides,
        )
        self.assertIn(
            "halting_stack_num_layers:nonContiguous",
            checkpoint_shapes.diagnostics.structural_fallback_reasons,
        )

    def test_checkpoint_shape_extractor_infers_expert_and_sampler_counts(
        self,
    ) -> None:
        state_dict: dict[str, torch.Tensor] = {}
        for outer_index in range(2):
            for router_index in range(4):
                output_dim = 3 if router_index == 3 else 16
                state_dict[
                    "main_model.expert_stack.layers."
                    f"{outer_index}.model.sampler.router.model.layers."
                    f"{router_index}.model.weight_params"
                ] = torch.zeros(16, output_dim)
            state_dict[
                "main_model.expert_stack.layers."
                f"{outer_index}.model.sampler.router.model.layers.3."
                "model.bias_params"
            ] = torch.zeros(3)
            for expert_index in range(3):
                for layer_index in range(2):
                    state_dict[
                        "main_model.expert_stack.layers."
                        f"{outer_index}.model.expert_modules.{expert_index}."
                        f"layers.{layer_index}.model.weight_params"
                    ] = torch.zeros(16, 16)

        checkpoint_shapes = checkpoint_graph_shapes_from_state_dict(state_dict)

        self.assertIsNotNone(checkpoint_shapes)
        if checkpoint_shapes is None:
            self.fail("Expected checkpoint shape extraction to succeed.")
        self.assertEqual(checkpoint_shapes.config_overrides["stack_num_layers"], 2)
        self.assertEqual(
            checkpoint_shapes.config_overrides["expert_stack_num_layers"],
            2,
        )
        self.assertEqual(
            checkpoint_shapes.config_overrides["router_stack_num_layers"],
            4,
        )
        self.assertNotIn(
            "router_stack_independent_flag",
            checkpoint_shapes.config_overrides,
        )
        self.assertEqual(checkpoint_shapes.config_overrides["expert_num_experts"], 3)

    def test_checkpoint_shape_extractor_counts_transformer_layers(self) -> None:
        checkpoint_shapes = checkpoint_graph_shapes_from_state_dict(
            {
                "transformer.layers.0.model.feed_forward_model.model.layers.0."
                "model.weight_params": torch.zeros(8, 8),
                "transformer.layers.1.model.feed_forward_model.model.layers.0."
                "model.weight_params": torch.zeros(8, 8),
                "transformer.layers.2.model.feed_forward_model.model.layers.0."
                "model.weight_params": torch.zeros(8, 8),
            }
        )

        self.assertIsNotNone(checkpoint_shapes)
        if checkpoint_shapes is None:
            self.fail("Expected checkpoint shape extraction to succeed.")
        self.assertEqual(checkpoint_shapes.config_overrides["stack_num_layers"], 3)

    def test_checkpoint_shape_extractor_omits_non_contiguous_layer_counts(
        self,
    ) -> None:
        checkpoint_shapes = checkpoint_graph_shapes_from_state_dict(
            {
                "main_model.layers.0.model.weight_params": torch.zeros(8, 8),
                "main_model.layers.2.model.weight_params": torch.zeros(8, 8),
            }
        )

        self.assertIsNotNone(checkpoint_shapes)
        if checkpoint_shapes is None:
            self.fail("Expected checkpoint shape extraction to succeed.")
        self.assertNotIn("stack_num_layers", checkpoint_shapes.config_overrides)
        self.assertIn(
            "stack_num_layers:nonContiguous",
            checkpoint_shapes.diagnostics.structural_fallback_reasons,
        )

    def test_checkpoint_shape_extractor_maps_direct_tensor_shape_details(
        self,
    ) -> None:
        checkpoint_shapes = checkpoint_graph_shapes_from_state_dict(
            {
                "linear.model.weight_params": torch.zeros(32, 64),
                "linear.model.bias_params": torch.zeros(128),
                "token_embedding.weight": torch.zeros(11, 7),
                "conv.weight": torch.zeros(5, 3, 2, 2),
                "projection.weight": torch.zeros(3, 4),
                "batch.running_mean": torch.zeros(4),
                "batch.num_batches_tracked": torch.tensor(0),
            }
        )

        self.assertIsNotNone(checkpoint_shapes)
        if checkpoint_shapes is None:
            self.fail("Expected checkpoint shape extraction to succeed.")
        self.assertEqual(
            checkpoint_shapes.parameter_shapes["linear.model"]["weightShape"],
            "32 x 64",
        )
        self.assertEqual(
            checkpoint_shapes.parameter_shapes["linear.model"]["biasShape"],
            "128",
        )
        self.assertEqual(
            checkpoint_shapes.parameter_shapes["linear.model"]["inputDim"],
            32,
        )
        self.assertEqual(
            checkpoint_shapes.parameter_shapes["linear.model"]["outputDim"],
            64,
        )
        self.assertEqual(
            checkpoint_shapes.parameter_shapes["linear.model"]["dims"],
            "32 -> 64",
        )
        self.assertEqual(
            checkpoint_shapes.parameter_shapes["token_embedding"]["weightShape"],
            "11 x 7",
        )
        self.assertNotIn("dims", checkpoint_shapes.parameter_shapes["token_embedding"])
        self.assertEqual(
            checkpoint_shapes.parameter_shapes["token_embedding"]["numEmbeddings"],
            11,
        )
        self.assertEqual(
            checkpoint_shapes.parameter_shapes["token_embedding"]["embeddingDim"],
            7,
        )
        self.assertEqual(
            checkpoint_shapes.parameter_shapes["conv"]["kernelShape"],
            "2 x 2",
        )
        self.assertNotIn("dims", checkpoint_shapes.parameter_shapes["conv"])
        self.assertEqual(
            checkpoint_shapes.parameter_shapes["projection"]["weightShape"],
            "3 x 4",
        )
        self.assertNotIn("dims", checkpoint_shapes.parameter_shapes["projection"])
        self.assertEqual(
            checkpoint_shapes.parameter_shapes["batch"]["tensorShapes"],
            {
                "num_batches_tracked": "scalar",
                "running_mean": "4",
            },
        )


if __name__ == "__main__":
    unittest.main()

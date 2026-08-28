import unittest
from copy import deepcopy

from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerConfig,
    LayerNormPositionOptions,
    LayerStack,
    LayerStackConfig,
)
from emperor.layers._stack.topology import LayerStackTopology
from emperor.linears import LinearLayerConfig
from emperor.nn import Module


class TestLayerStackTopology(unittest.TestCase):
    def test_layer_stack_uses_topology_with_an_immutable_plan(self):
        stack = LayerStack(
            LayerStackConfig(
                input_dim=4,
                hidden_dim=8,
                output_dim=3,
                num_layers=1,
                apply_output_postprocessing_flag=False,
                last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
                layer_config=LayerConfig(
                    activation=ActivationOptions.DISABLED,
                    dropout_probability=0.0,
                    layer_norm_position=LayerNormPositionOptions.DISABLED,
                    residual_config=None,
                    gate_config=None,
                    halting_config=None,
                    memory_config=None,
                    layer_model_config=LinearLayerConfig(bias_flag=True),
                ),
            )
        )

        self.assertIsInstance(
            stack.layer_dimension_resolver,
            LayerStackTopology,
        )
        self.assertIsInstance(stack._layer_dimensions(), tuple)

    def test_resolves_ordinary_layer_dimensions(self):
        cases = (
            (
                "single layer bypasses hidden dimension",
                LayerStackConfig(
                    input_dim=4,
                    hidden_dim=8,
                    output_dim=3,
                    num_layers=1,
                ),
                ((4, 3),),
            ),
            (
                "two layers with a stable input dimension",
                LayerStackConfig(
                    input_dim=8,
                    hidden_dim=8,
                    output_dim=3,
                    num_layers=2,
                ),
                ((8, 8), (8, 3)),
            ),
            (
                "two layers with a separate input projection",
                LayerStackConfig(
                    input_dim=4,
                    hidden_dim=8,
                    output_dim=3,
                    num_layers=2,
                ),
                ((4, 8), (8, 3)),
            ),
            (
                "multiple layers with a stable input dimension",
                LayerStackConfig(
                    input_dim=8,
                    hidden_dim=8,
                    output_dim=3,
                    num_layers=3,
                ),
                ((8, 8), (8, 8), (8, 3)),
            ),
            (
                "multiple layers with a separate input projection",
                LayerStackConfig(
                    input_dim=4,
                    hidden_dim=8,
                    output_dim=3,
                    num_layers=4,
                ),
                ((4, 8), (8, 8), (8, 8), (8, 3)),
            ),
        )

        for description, stack_config, expected_dimensions in cases:
            with self.subTest(description=description):
                topology = LayerStackTopology(stack_config)

                dimensions = topology.resolve()

                self.assertIsInstance(dimensions, tuple)
                self.assertEqual(dimensions, expected_dimensions)

    def test_resolve_does_not_mutate_the_stack_config(self):
        stack_config = LayerStackConfig(
            input_dim=4,
            hidden_dim=8,
            output_dim=3,
            num_layers=3,
        )
        original_stack_config = deepcopy(stack_config)

        LayerStackTopology(stack_config).resolve()

        self.assertEqual(stack_config, original_stack_config)

    def test_is_not_a_registered_model_module(self):
        topology = LayerStackTopology(
            LayerStackConfig(
                input_dim=4,
                hidden_dim=8,
                output_dim=3,
                num_layers=1,
            )
        )

        self.assertNotIsInstance(topology, Module)


if __name__ == "__main__":
    unittest.main()

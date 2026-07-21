import unittest

import torch

from emperor.halting import HaltingHiddenStateModeOptions, StickBreakingConfig
from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    Layer,
    LayerConfig,
    LayerNormPositionOptions,
    LayerStack,
    LayerStackConfig,
    LayerState,
    MirroredLayerStack,
    MirroredLayerStackConfig,
)
from emperor.linears import LinearLayerConfig


def make_layer_config(
    activation: ActivationOptions = ActivationOptions.RELU,
    dropout_probability: float = 0.0,
    halting_config: StickBreakingConfig | None = None,
) -> LayerConfig:
    return LayerConfig(
        activation=activation,
        residual_config=None,
        dropout_probability=dropout_probability,
        layer_norm_position=LayerNormPositionOptions.DISABLED,
        gate_config=None,
        halting_config=halting_config,
        memory_config=None,
        layer_model_config=LinearLayerConfig(bias_flag=True),
    )


def make_mirrored_config(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int,
    *,
    activation: ActivationOptions = ActivationOptions.RELU,
    dropout_probability: float = 0.0,
    halting_config: StickBreakingConfig | None = None,
) -> MirroredLayerStackConfig:
    return MirroredLayerStackConfig(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        apply_output_pipeline_flag=False,
        last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
        shared_gate_config=None,
        shared_halting_config=None,
        shared_memory_config=None,
        layer_config=make_layer_config(
            activation,
            dropout_probability,
            halting_config,
        ),
    )


def make_halting_config(dim: int) -> StickBreakingConfig:
    return StickBreakingConfig(
        input_dim=dim,
        threshold=0.99,
        dropout_probability=0.0,
        hidden_state_mode=HaltingHiddenStateModeOptions.RAW,
        halting_gate_config=LayerStackConfig(
            input_dim=dim,
            hidden_dim=dim,
            output_dim=2,
            num_layers=1,
            apply_output_pipeline_flag=False,
            last_layer_bias_option=LastLayerBiasOptions.DISABLED,
            shared_gate_config=None,
            shared_halting_config=None,
            shared_memory_config=None,
            layer_config=make_layer_config(ActivationOptions.DISABLED),
        ),
    )


class TestMirroredLayerStack(unittest.TestCase):
    def test_one_layer_depth_builds_two_wrapped_layers(self):
        stack = make_mirrored_config(4, 8, 3, 1).build()

        self.assertIsInstance(stack, MirroredLayerStack)
        self.assertIsInstance(stack, LayerStack)
        self.assertEqual(len(stack), 2)
        self.assertTrue(all(isinstance(layer, Layer) for layer in stack))
        self.assertEqual(
            [(layer.input_dim, layer.output_dim) for layer in stack],
            [(4, 8), (8, 3)],
        )

    def test_three_layer_depth_builds_independent_arms(self):
        stack = make_mirrored_config(4, 8, 3, 3).build()

        self.assertEqual(len(stack.expansion_layers), 3)
        self.assertEqual(len(stack.contraction_layers), 3)
        self.assertEqual(
            [(layer.input_dim, layer.output_dim) for layer in stack],
            [
                (4, 8),
                (8, 8),
                (8, 8),
                (8, 8),
                (8, 8),
                (8, 3),
            ],
        )
        expansion_parameters = {
            id(parameter)
            for layer in stack.expansion_layers
            for parameter in layer.parameters()
        }
        contraction_parameters = {
            id(parameter)
            for layer in stack.contraction_layers
            for parameter in layer.parameters()
        }
        self.assertTrue(expansion_parameters)
        self.assertTrue(contraction_parameters)
        self.assertTrue(expansion_parameters.isdisjoint(contraction_parameters))

    def test_only_final_contraction_uses_output_pipeline_policy(self):
        stack = make_mirrored_config(
            4,
            8,
            3,
            3,
            dropout_probability=0.25,
        ).build()

        for layer in stack[:-1]:
            self.assertEqual(layer.activation_function, ActivationOptions.RELU)
            self.assertIsNotNone(layer.dropout_module)
        self.assertEqual(
            stack[-1].activation_function,
            ActivationOptions.DISABLED,
        )
        self.assertIsNone(stack[-1].dropout_module)

    def test_preserves_shape_dtype_and_gradients(self):
        stack = make_mirrored_config(6, 12, 4, 3).build().double()
        input_values = torch.randn(
            2,
            5,
            6,
            dtype=torch.float64,
            requires_grad=True,
        )

        output_state = stack(LayerState(hidden=input_values))
        output_state.hidden.square().mean().backward()

        self.assertEqual(output_state.hidden.shape, (2, 5, 4))
        self.assertEqual(output_state.hidden.dtype, torch.float64)
        self.assertIsNotNone(input_values.grad)
        self.assertTrue(
            all(
                parameter.grad is not None
                for parameter in stack.parameters()
                if parameter.requires_grad
            )
        )

    def test_one_layer_depth_supports_halting_across_both_arms(self):
        dim = 4
        stack = make_mirrored_config(
            dim,
            dim,
            dim,
            1,
            halting_config=make_halting_config(dim),
        ).build()

        self.assertEqual(len(stack), 2)
        self.assertTrue(all(layer.halting_model is not None for layer in stack))


if __name__ == "__main__":
    unittest.main()

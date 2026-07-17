from __future__ import annotations

import copy
import unittest

import torch

import emperor.augmentations as augmentations
import emperor.augmentations.adaptive_parameters as adaptive_parameters
from emperor.augmentations.adaptive_parameters import (
    AdaptiveLinearLayerConfig,
    AdaptiveParameterAugmentationConfig,
    AdditiveDynamicBiasConfig,
    WeightDecayScheduleOptions,
)
from emperor.augmentations.adaptive_parameters._augmentation import (
    AdaptiveParameterAugmentation,
)
from emperor.augmentations.adaptive_parameters._diagonals.base import (
    DynamicDiagonalAbstract,
)
from emperor.augmentations.adaptive_parameters._linear_adapter import (
    AdaptiveLinearLayer,
)
from emperor.augmentations.adaptive_parameters._weights.base import (
    DynamicWeightAbstract,
)
from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerConfig,
    LayerNormPositionOptions,
    LayerStackConfig,
    ResidualConnectionOptions,
)
from emperor.linears import LinearLayer, LinearLayerConfig


def linear_stack_config(input_dim: int, output_dim: int) -> LayerStackConfig:
    return LayerStackConfig(
        input_dim=input_dim,
        hidden_dim=max(input_dim, output_dim),
        output_dim=output_dim,
        num_layers=1,
        last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
        apply_output_pipeline_flag=False,
        layer_config=LayerConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            activation=ActivationOptions.DISABLED,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            residual_connection_option=ResidualConnectionOptions.DISABLED,
            dropout_probability=0.0,
            gate_config=None,
            halting_config=None,
            memory_config=None,
            layer_model_config=LinearLayerConfig(
                input_dim=input_dim,
                output_dim=output_dim,
                bias_flag=True,
            ),
        ),
    )


def adaptive_linear_config(
    *,
    input_dim: int = 2,
    output_dim: int = 2,
    bias_flag: bool = True,
    adaptive_bias: bool = False,
) -> AdaptiveLinearLayerConfig:
    bias_config = None
    if adaptive_bias:
        bias_config = AdditiveDynamicBiasConfig(
            decay_schedule=WeightDecayScheduleOptions.DISABLED,
            decay_rate=0.0,
            decay_warmup_batches=0,
            model_config=linear_stack_config(input_dim, output_dim),
        )
    return AdaptiveLinearLayerConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        bias_flag=bias_flag,
        adaptive_augmentation_config=AdaptiveParameterAugmentationConfig(
            bias_config=bias_config,
        ),
    )


def set_exact_adaptive_linear_parameters(model: AdaptiveLinearLayer) -> None:
    with torch.no_grad():
        model.weight_params.copy_(
            torch.tensor(
                [
                    [1.0, -2.0],
                    [0.5, 3.0],
                ]
            )
        )
        model.bias_params.copy_(torch.tensor([0.25, -0.75]))
        generator_layer = model.adaptive_behaviour.bias_model.model[0].model
        generator_layer.weight_params.copy_(torch.eye(2))
        generator_layer.bias_params.zero_()


class AdaptiveParameterBehavioralContractTests(unittest.TestCase):
    def test_interfaces_use_explicit_exports_without_dynamic_shortcuts(self):
        self.assertIs(augmentations.adaptive_parameters, adaptive_parameters)
        self.assertFalse(hasattr(augmentations, "__getattr__"))
        self.assertFalse(hasattr(adaptive_parameters, "__getattr__"))
        self.assertFalse(hasattr(adaptive_parameters, "_LAZY_EXPORTS"))
        self.assertFalse(hasattr(adaptive_parameters, "AdaptiveLinearLayer"))
        self.assertFalse(hasattr(adaptive_parameters, "AdaptiveParameterAugmentation"))

    def test_model_only_adjustment_path_executes_a_real_emperor_linear_layer(self):
        augmentation = AdaptiveParameterAugmentation(
            AdaptiveParameterAugmentationConfig(input_dim=2, output_dim=2)
        )
        generator = LinearLayer(
            LinearLayerConfig(input_dim=2, output_dim=2, bias_flag=True)
        )
        with torch.no_grad():
            generator.weight_params.copy_(
                torch.tensor(
                    [
                        [2.0, -1.0],
                        [0.5, 3.0],
                    ]
                )
            )
            generator.bias_params.copy_(torch.tensor([0.25, -0.5]))
        inputs = torch.tensor([[1.0, 2.0], [-3.0, 0.5]])

        output = augmentation._AdaptiveParameterAugmentation__call_model(
            generator,
            None,
            inputs,
        )

        torch.testing.assert_close(
            output,
            inputs @ generator.weight_params + generator.bias_params,
        )

    def test_abstract_parameter_generators_report_the_missing_forward_contract(self):
        diagonal = DynamicDiagonalAbstract.__new__(DynamicDiagonalAbstract)
        torch.nn.Module.__init__(diagonal)
        weight = DynamicWeightAbstract.__new__(DynamicWeightAbstract)
        torch.nn.Module.__init__(weight)
        parameters = torch.ones(2, 2)
        inputs = torch.ones(1, 2)

        with self.assertRaisesRegex(
            NotImplementedError,
            r"^DynamicDiagonalAbstract must implement forward\(\)\.$",
        ):
            diagonal(parameters, inputs)
        with self.assertRaisesRegex(
            NotImplementedError,
            r"^DynamicWeightAbstract must implement forward\(\)\.$",
        ):
            weight(parameters, inputs)

    def test_disabled_adaptive_linear_is_exact_for_float64_non_contiguous_input(self):
        model = AdaptiveLinearLayer(
            adaptive_linear_config(input_dim=2, output_dim=3)
        ).double()
        with torch.no_grad():
            model.weight_params.copy_(
                torch.tensor(
                    [
                        [1.0, -2.0, 0.5],
                        [3.0, 0.25, -1.0],
                    ],
                    dtype=torch.float64,
                )
            )
            model.bias_params.copy_(torch.tensor([0.5, -1.0, 2.0], dtype=torch.float64))
        inputs = torch.tensor(
            [
                [1.0, -2.0, 3.0],
                [0.5, 4.0, -1.5],
            ],
            dtype=torch.float64,
        ).transpose(0, 1)
        self.assertFalse(inputs.is_contiguous())

        output = model(inputs)

        expected = inputs @ model.weight_params + model.bias_params
        torch.testing.assert_close(output, expected)
        self.assertEqual(output.dtype, torch.float64)
        self.assertEqual(output.device.type, "cpu")
        self.assertTrue(torch.isfinite(output).all())

    def test_config_builds_disabled_bias_free_layer_with_exact_matrix_product(self):
        config = adaptive_linear_config(
            input_dim=2,
            output_dim=3,
            bias_flag=False,
        )

        model = config.build()
        with torch.no_grad():
            model.weight_params.copy_(
                torch.tensor(
                    [
                        [1.0, 2.0, -1.0],
                        [3.0, -0.5, 4.0],
                    ]
                )
            )
        inputs = torch.tensor([[2.0, -1.0], [0.5, 3.0]])

        self.assertIsInstance(model, AdaptiveLinearLayer)
        self.assertFalse(model.has_adaptive_augmentation)
        self.assertIsNone(model.adaptive_behaviour)
        self.assertIsNone(model.bias_params)
        torch.testing.assert_close(model(inputs), inputs @ model.weight_params)

    def test_per_sample_affine_callback_is_exact_and_isolates_samples(self):
        model = AdaptiveLinearLayer(adaptive_linear_config())
        inputs = torch.tensor([[1.0, 2.0], [-3.0, 0.5]])
        weights = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[-2.0, 0.5], [1.0, 3.0]],
            ]
        )
        bias = torch.tensor([[0.25, -0.5], [2.0, 1.0]])

        output = model._compute_affine_transformation_callback(
            weights,
            bias,
            inputs,
        )

        expected = torch.stack(
            [
                inputs[0] @ weights[0] + bias[0],
                inputs[1] @ weights[1] + bias[1],
            ]
        )
        torch.testing.assert_close(output, expected)
        changed_weights = weights.clone()
        changed_weights[1].add_(100.0)
        changed = model._compute_affine_transformation_callback(
            changed_weights,
            bias,
            inputs,
        )
        torch.testing.assert_close(changed[0], output[0])
        self.assertFalse(torch.equal(changed[1], output[1]))

    def test_active_bias_has_exact_per_sample_math_and_nonzero_gradients(self):
        model = AdaptiveLinearLayer(adaptive_linear_config(adaptive_bias=True))
        set_exact_adaptive_linear_parameters(model)
        inputs = torch.tensor(
            [
                [1.0, 2.0],
                [-3.0, 0.5],
            ],
            requires_grad=True,
        )

        output = model(inputs)

        expected = inputs @ model.weight_params + model.bias_params + inputs
        torch.testing.assert_close(output, expected)
        output.square().sum().backward()
        self.assertIsNotNone(inputs.grad)
        self.assertTrue(torch.isfinite(inputs.grad).all())
        self.assertTrue(torch.any(inputs.grad != 0))
        generator = model.adaptive_behaviour.bias_model.model[0].model
        for parameter in (
            model.weight_params,
            model.bias_params,
            generator.weight_params,
            generator.bias_params,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertTrue(torch.any(parameter.grad != 0))

    def test_adam_state_and_model_state_continue_strictly_after_restore(self):
        source = AdaptiveLinearLayer(adaptive_linear_config(adaptive_bias=True))
        set_exact_adaptive_linear_parameters(source)
        source_optimizer = torch.optim.Adam(source.parameters(), lr=0.01)
        first_batch = torch.tensor([[1.0, 2.0], [-1.0, 0.5]])

        source_optimizer.zero_grad()
        source(first_batch).square().mean().backward()
        source_optimizer.step()

        model_state = copy.deepcopy(source.state_dict())
        optimizer_state = copy.deepcopy(source_optimizer.state_dict())
        restored = AdaptiveLinearLayer(adaptive_linear_config(adaptive_bias=True))
        incompatible = restored.load_state_dict(model_state, strict=True)
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])
        restored_optimizer = torch.optim.Adam(restored.parameters(), lr=0.01)
        restored_optimizer.load_state_dict(optimizer_state)

        continuation_batch = torch.tensor([[0.25, -2.0], [3.0, 1.5]])
        torch.testing.assert_close(
            source(continuation_batch),
            restored(continuation_batch),
        )
        for model, optimizer in (
            (source, source_optimizer),
            (restored, restored_optimizer),
        ):
            optimizer.zero_grad()
            model(continuation_batch).square().mean().backward()
            optimizer.step()

        for name, tensor in source.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[name], tensor)
        self.assertEqual(
            source_optimizer.state_dict()["param_groups"],
            restored_optimizer.state_dict()["param_groups"],
        )
        for source_value, restored_value in zip(
            source_optimizer.state_dict()["state"].values(),
            restored_optimizer.state_dict()["state"].values(),
            strict=True,
        ):
            self.assertEqual(source_value.keys(), restored_value.keys())
            for key, value in source_value.items():
                if torch.is_tensor(value):
                    torch.testing.assert_close(restored_value[key], value)
                else:
                    self.assertEqual(restored_value[key], value)

if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import subprocess
import sys
import unittest

import torch

from emperor.convs import Conv2dLayerConfig
from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerConfig,
    LayerNormPositionOptions,
    LayerStackConfig,
)
from emperor.linears import LinearLayerConfig
from emperor.patch import (
    ConvPatchEmbeddingConfig,
    LinearPatchEmbeddingConfig,
    PatchBase,
    PatchConfig,
    PatchEmbeddingConv,
    PatchEmbeddingLinear,
)
from models.vit.linear.experiment_config import ExperimentConfig


def linear_stack_config(
    *,
    input_dim: int = 101,
    hidden_dim: int = 7,
    output_dim: int = 103,
    bias_flag: bool = True,
) -> LayerStackConfig:
    return LayerStackConfig(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=1,
        last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
        apply_output_pipeline_flag=False,
        layer_config=LayerConfig(
            activation=ActivationOptions.DISABLED,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            residual_config=None,
            dropout_probability=0.0,
            gate_config=None,
            halting_config=None,
            memory_config=None,
            layer_model_config=LinearLayerConfig(bias_flag=bias_flag),
        ),
    )


def conv_stack_config(
    *,
    input_dim: int = 107,
    hidden_dim: int = 11,
    output_dim: int = 109,
    kernel_size: int = 2,
    stride: int = 1,
    padding: int = 0,
    bias_flag: bool = True,
) -> LayerStackConfig:
    return LayerStackConfig(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=1,
        last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
        apply_output_pipeline_flag=False,
        layer_config=LayerConfig(
            activation=ActivationOptions.DISABLED,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            residual_config=None,
            dropout_probability=0.0,
            gate_config=None,
            halting_config=None,
            memory_config=None,
            layer_model_config=Conv2dLayerConfig(
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias_flag=bias_flag,
            ),
        ),
    )


def linear_config(**overrides: object) -> LinearPatchEmbeddingConfig:
    values: dict[str, object] = {
        "embedding_dim": 2,
        "num_input_channels": 1,
        "patch_size": 2,
        "dropout_probability": 0.0,
        "stride": 2,
        "padding": 0,
        "embedding_stack_config": linear_stack_config(),
    }
    values.update(overrides)
    return LinearPatchEmbeddingConfig(**values)


def conv_config(**overrides: object) -> ConvPatchEmbeddingConfig:
    values: dict[str, object] = {
        "embedding_dim": 2,
        "num_input_channels": 1,
        "patch_size": 2,
        "dropout_probability": 0.0,
        "conv_stack_config": conv_stack_config(),
    }
    values.update(overrides)
    return ConvPatchEmbeddingConfig(**values)


class PatchConfigurationBehaviorTests(unittest.TestCase):
    def test_base_config_defaults_metadata_and_build_error_are_exact(self) -> None:
        config = PatchConfig()

        self.assertEqual(
            (
                config.embedding_dim,
                config.num_input_channels,
                config.patch_size,
                config.dropout_probability,
            ),
            (None, None, None, None),
        )
        self.assertEqual(config.get_custom_parameters(), {})
        with self.assertRaises(NotImplementedError) as error:
            config.build()
        self.assertEqual(
            str(error.exception),
            "PatchConfig must implement `_registry_owner` or override `build`",
        )

    def test_concrete_configs_preserve_explicit_zero_and_dispatch_real_owners(
        self,
    ) -> None:
        linear = linear_config(padding=0, dropout_probability=0.0)
        convolutional = conv_config(dropout_probability=0.0)

        self.assertEqual(
            (
                linear.embedding_dim,
                linear.num_input_channels,
                linear.patch_size,
                linear.dropout_probability,
                linear.stride,
                linear.padding,
            ),
            (2, 1, 2, 0.0, 2, 0),
        )
        self.assertEqual(
            (
                convolutional.embedding_dim,
                convolutional.num_input_channels,
                convolutional.patch_size,
                convolutional.dropout_probability,
            ),
            (2, 1, 2, 0.0),
        )
        self.assertEqual(linear.get_custom_parameters(), {})
        self.assertEqual(convolutional.get_custom_parameters(), {})
        self.assertIs(linear.registry_owner(), PatchEmbeddingLinear)
        self.assertIs(convolutional.registry_owner(), PatchEmbeddingConv)
        self.assertIsInstance(linear.build(), PatchEmbeddingLinear)
        self.assertIsInstance(convolutional.build(), PatchEmbeddingConv)

    def test_real_vit_wrapper_and_partial_override_precedence(self) -> None:
        base = linear_config(
            embedding_dim=3,
            stride=2,
            padding=0,
            dropout_probability=0.0,
            embedding_stack_config=linear_stack_config(output_dim=97),
        )
        wrapper = ExperimentConfig(patch_config=base)
        override = LinearPatchEmbeddingConfig(
            embedding_dim=4,
            stride=1,
            padding=1,
            dropout_probability=0.25,
        )

        model = PatchEmbeddingLinear(wrapper, override)

        self.assertEqual(
            (
                model.embedding_dim,
                model.num_input_channels,
                model.patch_size,
                model.stride,
                model.padding,
                model.dropout_probability,
                model.patch_dim,
                model.embedding_model[0].model.input_dim,
                model.embedding_model[0].model.output_dim,
            ),
            (4, 1, 2, 1, 1, 0.25, 4, 4, 4),
        )
        self.assertEqual(
            (
                base.embedding_dim,
                base.stride,
                base.padding,
                base.dropout_probability,
                base.embedding_stack_config.output_dim,
            ),
            (3, 2, 0, 0.0, 97),
        )
        self.assertEqual(
            (
                override.embedding_dim,
                override.num_input_channels,
                override.patch_size,
                override.stride,
                override.padding,
                override.dropout_probability,
            ),
            (4, None, None, 1, 1, 0.25),
        )


class PatchBaseBehaviorTests(unittest.TestCase):
    def test_default_class_token_initialization_matches_torch_rng_exactly(
        self,
    ) -> None:
        config = PatchConfig(
            embedding_dim=3,
            num_input_channels=1,
            patch_size=2,
            dropout_probability=0.0,
        )
        with torch.random.fork_rng():
            torch.manual_seed(314159)
            model = PatchBase(config)
            torch.manual_seed(314159)
            expected = torch.randn(1, 1, 3)

        torch.testing.assert_close(model.class_token, expected, rtol=0, atol=0)
        self.assertTrue(model.class_token.requires_grad)
        self.assertTrue(model.class_token.is_leaf)
        self.assertEqual(model.dropout.p, 0.0)

    def test_custom_class_tokens_preserve_shape_values_and_batch_gradients(
        self,
    ) -> None:
        model = PatchBase(
            PatchConfig(
                embedding_dim=3,
                num_input_channels=1,
                patch_size=2,
                dropout_probability=0.0,
            )
        )
        with torch.random.fork_rng():
            torch.manual_seed(2718)
            custom = model._create_class_token((1, 2, 3))
            torch.manual_seed(2718)
            expected_custom = torch.randn(1, 2, 3)
        model.class_token = custom
        patches = torch.tensor(
            [
                [[10.0, 11.0, 12.0]],
                [[20.0, 21.0, 22.0]],
            ]
        )

        output = model._concatenate_class_token(patches)

        expected = torch.cat(
            [expected_custom.expand(2, -1, -1), patches],
            dim=1,
        )
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        self.assertEqual(output.shape, (2, 3, 3))
        output.sum().backward()
        torch.testing.assert_close(
            custom.grad,
            torch.full_like(custom, 2.0),
            rtol=0,
            atol=0,
        )


class LinearPatchEmbeddingBehaviorTests(unittest.TestCase):
    def test_padded_overlapping_rectangular_patches_are_exact_and_isolated(
        self,
    ) -> None:
        stack_config = linear_stack_config(
            input_dim=91,
            output_dim=93,
        )
        model = PatchEmbeddingLinear(
            linear_config(
                embedding_dim=1,
                patch_size=2,
                stride=1,
                padding=1,
                embedding_stack_config=stack_config,
            )
        )
        model.eval()
        with torch.no_grad():
            model.class_token.fill_(-2.0)
            model.embedding_model[0].model.weight_params.fill_(1.0)
            model.embedding_model[0].model.bias_params.fill_(0.5)
        sample = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
        inputs = torch.stack([sample, sample * 10.0])

        output = model(inputs)

        patch_sums = [1.0, 3.0, 5.0, 3.0, 5.0, 12.0, 16.0, 9.0, 4.0, 9.0, 11.0, 6.0]
        expected = torch.tensor(
            [
                [-2.0, *(value + 0.5 for value in patch_sums)],
                [-2.0, *(value * 10.0 + 0.5 for value in patch_sums)],
            ]
        ).unsqueeze(-1)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        self.assertEqual(output.shape, (2, 13, 1))
        self.assertEqual(model.patch_model.kernel_size, 2)
        self.assertEqual(model.patch_model.stride, 1)
        self.assertEqual(model.patch_model.padding, 1)
        self.assertEqual(model.embedding_model[0].model.input_dim, 4)
        self.assertEqual(model.embedding_model[0].model.output_dim, 1)
        self.assertEqual(
            (stack_config.input_dim, stack_config.output_dim),
            (91, 93),
        )

    def test_multichannel_unfold_order_and_projection_are_hand_calculated(
        self,
    ) -> None:
        model = PatchEmbeddingLinear(
            linear_config(
                embedding_dim=2,
                num_input_channels=2,
                patch_size=2,
                stride=2,
                padding=0,
            )
        )
        model.eval()
        with torch.no_grad():
            model.class_token.copy_(torch.tensor([[[9.0, -9.0]]]))
            model.embedding_model[0].model.weight_params.copy_(
                torch.tensor(
                    [
                        [1.0, 1.0],
                        [2.0, -1.0],
                        [3.0, 1.0],
                        [4.0, -1.0],
                        [10.0, 2.0],
                        [20.0, -2.0],
                        [30.0, 2.0],
                        [40.0, -2.0],
                    ]
                )
            )
            model.embedding_model[0].model.bias_params.copy_(torch.tensor([0.5, 1.0]))
        inputs = torch.tensor(
            [
                [
                    [[1.0, 2.0], [3.0, 4.0]],
                    [[5.0, 6.0], [7.0, 8.0]],
                ]
            ]
        )

        output = model(inputs)

        expected = torch.tensor([[[9.0, -9.0], [730.5, -5.0]]])
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        self.assertEqual(output.dtype, torch.float32)
        self.assertEqual(output.device.type, "cpu")
        self.assertTrue(torch.isfinite(output).all())

    def test_too_small_input_has_the_real_unfold_geometry_error(self) -> None:
        model = PatchEmbeddingLinear(
            linear_config(
                patch_size=4,
                stride=1,
                padding=0,
                embedding_stack_config=linear_stack_config(),
            )
        )

        with self.assertRaises(RuntimeError) as error:
            model(torch.ones(1, 1, 2, 3))

        self.assertIn(
            "calculated shape of the array of sliding blocks", str(error.exception)
        )


class ConvPatchEmbeddingBehaviorTests(unittest.TestCase):
    def test_rectangular_multichannel_convolution_is_exact_and_isolated(
        self,
    ) -> None:
        stack_config = conv_stack_config(
            input_dim=95,
            output_dim=97,
            kernel_size=2,
            stride=1,
        )
        model = PatchEmbeddingConv(
            conv_config(
                embedding_dim=2,
                num_input_channels=2,
                patch_size=2,
                conv_stack_config=stack_config,
            )
        )
        model.eval()
        convolution = model.patch_model[0].model.model
        with torch.no_grad():
            model.class_token.copy_(torch.tensor([[[100.0, 200.0]]]))
            convolution.weight.zero_()
            convolution.weight[0, 0].fill_(1.0)
            convolution.weight[1, 1, 0, 0] = 1.0
            convolution.bias.copy_(torch.tensor([0.5, -1.0]))
        sample = torch.tensor(
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]],
            ]
        )
        inputs = torch.stack([sample, sample * 2.0])

        output = model(inputs)

        expected = torch.tensor(
            [
                [[100.0, 200.0], [12.5, 9.0], [16.5, 19.0]],
                [[100.0, 200.0], [24.5, 19.0], [32.5, 39.0]],
            ]
        )
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        self.assertEqual(output.shape, (2, 3, 2))
        self.assertEqual(model.patch_model[0].model.input_dim, 2)
        self.assertEqual(model.patch_model[0].model.output_dim, 2)
        self.assertEqual(
            (stack_config.input_dim, stack_config.output_dim),
            (95, 97),
        )
        self.assertTrue(torch.isfinite(output).all())

    def test_too_small_input_has_the_real_convolution_geometry_error(self) -> None:
        model = PatchEmbeddingConv(
            conv_config(
                patch_size=4,
                conv_stack_config=conv_stack_config(kernel_size=4),
            )
        )

        with self.assertRaises(RuntimeError) as error:
            model(torch.ones(1, 1, 2, 3))

        self.assertIn(
            "Kernel size can't be greater than actual input size",
            str(error.exception),
        )


class PatchTrainingAndStateBehaviorTests(unittest.TestCase):
    def test_dropout_zero_is_a_noop_and_one_is_train_only_for_both_variants(
        self,
    ) -> None:
        cases = (
            (
                PatchEmbeddingLinear,
                linear_config(dropout_probability=0.0),
                linear_config(dropout_probability=1.0),
                torch.arange(32, dtype=torch.float32).reshape(2, 1, 4, 4),
            ),
            (
                PatchEmbeddingConv,
                conv_config(dropout_probability=0.0),
                conv_config(dropout_probability=1.0),
                torch.arange(32, dtype=torch.float32).reshape(2, 1, 4, 4),
            ),
        )

        for module_type, baseline_config, dropped_config, inputs in cases:
            with self.subTest(module_type=module_type.__name__):
                baseline = module_type(baseline_config)
                dropped = module_type(dropped_config)
                dropped.load_state_dict(baseline.state_dict(), strict=True)

                baseline.train()
                train_baseline = baseline(inputs)
                baseline.eval()
                eval_baseline = baseline(inputs)
                repeated_baseline = baseline(inputs)
                dropped.eval()
                eval_dropped = dropped(inputs)
                dropped.train()
                train_dropped = dropped(inputs)

                torch.testing.assert_close(
                    train_baseline,
                    eval_baseline,
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    repeated_baseline,
                    eval_baseline,
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    eval_dropped,
                    eval_baseline,
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    train_dropped,
                    torch.zeros_like(train_dropped),
                    rtol=0,
                    atol=0,
                )
                self.assertGreater(eval_baseline.abs().sum().item(), 0.0)

    def test_float64_noncontiguous_inputs_preserve_dtype_device_and_values(
        self,
    ) -> None:
        cases = (
            PatchEmbeddingLinear(
                linear_config(
                    num_input_channels=2,
                    stride=1,
                    embedding_stack_config=linear_stack_config(),
                )
            ),
            PatchEmbeddingConv(
                conv_config(
                    num_input_channels=2,
                    conv_stack_config=conv_stack_config(stride=1),
                )
            ),
        )
        source = torch.arange(48, dtype=torch.float64).reshape(2, 2, 4, 3)
        inputs = source.transpose(-1, -2)
        self.assertFalse(inputs.is_contiguous())

        for model in cases:
            with self.subTest(model=type(model).__name__):
                model.double()
                model.eval()
                noncontiguous_output = model(inputs)
                contiguous_output = model(inputs.contiguous())

                torch.testing.assert_close(
                    noncontiguous_output,
                    contiguous_output,
                    rtol=0,
                    atol=0,
                )
                self.assertEqual(noncontiguous_output.shape, (2, 7, 2))
                self.assertEqual(noncontiguous_output.dtype, torch.float64)
                self.assertEqual(noncontiguous_output.device.type, "cpu")
                self.assertTrue(torch.isfinite(noncontiguous_output).all())

    def test_all_trainable_paths_have_finite_nonzero_gradients_and_update(
        self,
    ) -> None:
        cases = (
            (
                PatchEmbeddingLinear(linear_config()),
                torch.arange(32, dtype=torch.float32).reshape(2, 1, 4, 4) + 1.0,
            ),
            (
                PatchEmbeddingConv(conv_config()),
                torch.arange(32, dtype=torch.float32).reshape(2, 1, 4, 4) + 1.0,
            ),
        )

        for model, raw_inputs in cases:
            with self.subTest(model=type(model).__name__):
                with torch.no_grad():
                    model.class_token.fill_(1.0)
                    for name, parameter in model.named_parameters():
                        if name != "class_token":
                            parameter.fill_(0.25)
                inputs = raw_inputs.clone().requires_grad_()
                optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
                before = {
                    name: parameter.detach().clone()
                    for name, parameter in model.named_parameters()
                }

                loss = model(inputs).square().sum()
                loss.backward()

                torch.testing.assert_close(
                    model.class_token.grad,
                    torch.full_like(model.class_token, 4.0),
                    rtol=0,
                    atol=0,
                )
                self.assertIsNotNone(inputs.grad)
                self.assertTrue(torch.isfinite(inputs.grad).all())
                self.assertGreater(inputs.grad.abs().sum().item(), 0.0)
                for name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                    self.assertGreater(parameter.grad.abs().sum().item(), 0.0, name)

                optimizer.step()

                for name, parameter in model.named_parameters():
                    self.assertFalse(torch.equal(parameter, before[name]), name)

    def test_strict_state_roundtrip_continues_exact_outputs(self) -> None:
        cases = (
            (
                PatchEmbeddingLinear,
                linear_config(),
                torch.arange(32, dtype=torch.float32).reshape(2, 1, 4, 4),
            ),
            (
                PatchEmbeddingConv,
                conv_config(),
                torch.arange(32, dtype=torch.float32).reshape(2, 1, 4, 4),
            ),
        )

        for module_type, config, inputs in cases:
            with self.subTest(module_type=module_type.__name__):
                model = module_type(config)
                model.eval()
                with torch.no_grad():
                    for index, parameter in enumerate(model.parameters()):
                        values = torch.arange(
                            parameter.numel(),
                            dtype=parameter.dtype,
                            device=parameter.device,
                        ).reshape_as(parameter)
                        parameter.copy_(values / 10.0 + index)
                expected = model(inputs)
                state = model.state_dict()

                restored = module_type(config)
                incompatible = restored.load_state_dict(state, strict=True)
                restored.eval()
                actual = restored(inputs)

                self.assertEqual(incompatible.missing_keys, [])
                self.assertEqual(incompatible.unexpected_keys, [])
                self.assertEqual(tuple(restored.state_dict()), tuple(state))
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(
                    restored(inputs),
                    expected,
                    rtol=0,
                    atol=0,
                )


class PatchInterfaceBehaviorTests(unittest.TestCase):
    def test_unknown_attribute_has_exact_error_and_keyerror_cause(self) -> None:
        import emperor.patch as patch

        with self.assertRaises(AttributeError) as error:
            patch.__getattr__("DoesNotExist")

        self.assertEqual(
            str(error.exception),
            "module 'emperor.patch' has no attribute 'DoesNotExist'",
        )
        self.assertIsInstance(error.exception.__cause__, KeyError)

    def test_lazy_interface_is_lightweight_cached_and_resolves_real_exports(
        self,
    ) -> None:
        script = """\
import json
import sys

import emperor.patch as patch

private_modules = (
    "emperor.patch._base",
    "emperor.patch._config",
    "emperor.patch._validation",
    "emperor.patch._variants",
    "emperor.patch._variants.convolutional",
    "emperor.patch._variants.linear",
)
before = {name: name in sys.modules for name in private_modules}
config_first = patch.LinearPatchEmbeddingConfig
config_second = patch.LinearPatchEmbeddingConfig
linear = patch.PatchEmbeddingLinear
convolutional = patch.PatchEmbeddingConv
print(json.dumps({
    "all": patch.__all__,
    "before": before,
    "config_cached": config_first is config_second,
    "config_module": config_first.__module__,
    "linear_module": linear.__module__,
    "convolutional_module": convolutional.__module__,
}))
"""

        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(
            json.loads(completed.stdout),
            {
                "all": [
                    "PatchConfig",
                    "LinearPatchEmbeddingConfig",
                    "ConvPatchEmbeddingConfig",
                    "PatchBase",
                    "PatchEmbeddingLinear",
                    "PatchEmbeddingConv",
                ],
                "before": {
                    "emperor.patch._base": False,
                    "emperor.patch._config": False,
                    "emperor.patch._validation": False,
                    "emperor.patch._variants": False,
                    "emperor.patch._variants.convolutional": False,
                    "emperor.patch._variants.linear": False,
                },
                "config_cached": True,
                "config_module": "emperor.patch._config",
                "linear_module": "emperor.patch._variants.linear",
                "convolutional_module": "emperor.patch._variants.convolutional",
            },
        )


if __name__ == "__main__":
    unittest.main()

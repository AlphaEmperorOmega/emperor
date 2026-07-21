from __future__ import annotations

import json
import subprocess
import sys
import unittest
from dataclasses import fields

import torch

from emperor.convs import Conv2dLayerConfig
from emperor.convs._layer import Conv2dLayer
from emperor.convs._validation import Conv2dLayerValidator


def make_config(**overrides: object) -> Conv2dLayerConfig:
    values: dict[str, object] = {
        "input_dim": 3,
        "output_dim": 5,
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
        "bias_flag": True,
    }
    values.update(overrides)
    return Conv2dLayerConfig(**values)


class Conv2dConfigurationTests(unittest.TestCase):
    def test_defaults_metadata_registry_and_explicit_parameter_tracking(
        self,
    ) -> None:
        default = Conv2dLayerConfig()
        explicit = make_config()

        self.assertEqual(
            (
                default.input_dim,
                default.output_dim,
                default.kernel_size,
                default.stride,
                default.padding,
                default.bias_flag,
            ),
            (None, None, None, None, None, None),
        )
        self.assertEqual(default.get_custom_parameters(), {})
        self.assertEqual(
            explicit.get_custom_parameters(),
            {"bias_flag": True},
        )
        self.assertIs(explicit.registry_owner(), Conv2dLayer)
        self.assertEqual(
            {
                config_field.name: config_field.metadata["help"]
                for config_field in fields(default)
            },
            {
                "input_dim": "Input channel count (Conv2d in_channels).",
                "output_dim": "Output channel count (Conv2d out_channels).",
                "kernel_size": "Conv2d kernel size.",
                "stride": "Conv2d stride.",
                "padding": "Conv2d padding.",
                "bias_flag": "Add a learnable bias to the output.",
            },
        )

    def test_build_dispatches_to_real_layer_and_partial_overrides_take_precedence(
        self,
    ) -> None:
        base = make_config(
            input_dim=2,
            output_dim=3,
            kernel_size=2,
            stride=1,
            padding=0,
            bias_flag=True,
        )
        overrides = Conv2dLayerConfig(
            output_dim=4,
            stride=2,
            bias_flag=False,
        )

        unmodified = base.build()
        overridden = base.build(overrides)

        self.assertIsInstance(unmodified, Conv2dLayer)
        self.assertIs(unmodified.cfg, base)
        self.assertIsNot(overridden.cfg, base)
        self.assertEqual(
            (
                overridden.input_dim,
                overridden.output_dim,
                overridden.kernel_size,
                overridden.stride,
                overridden.padding,
                overridden.bias_flag,
            ),
            (2, 4, 2, 2, 0, False),
        )
        self.assertEqual(
            (
                base.input_dim,
                base.output_dim,
                base.kernel_size,
                base.stride,
                base.padding,
                base.bias_flag,
            ),
            (2, 3, 2, 1, 0, True),
        )
        self.assertEqual(
            (
                overrides.input_dim,
                overrides.output_dim,
                overrides.kernel_size,
                overrides.stride,
                overrides.padding,
                overrides.bias_flag,
            ),
            (None, 4, None, 2, None, False),
        )

    def test_every_configuration_field_is_required(self) -> None:
        for field_name in (
            "input_dim",
            "output_dim",
            "kernel_size",
            "stride",
            "padding",
            "bias_flag",
        ):
            with self.subTest(field_name=field_name):
                with self.assertRaises(ValueError) as error:
                    Conv2dLayer(make_config(**{field_name: None}))
                self.assertEqual(
                    str(error.exception),
                    f"{field_name} is required for Conv2dLayerConfig, received None",
                )

    def test_configuration_fields_reject_wrong_runtime_types(self) -> None:
        invalid_values = {
            "input_dim": 3.0,
            "output_dim": "5",
            "kernel_size": 3.0,
            "stride": False,
            "padding": 1.0,
            "bias_flag": 1,
        }
        expected_types = {
            "input_dim": "int",
            "output_dim": "int",
            "kernel_size": "int",
            "stride": "int",
            "padding": "int",
            "bias_flag": "bool",
        }

        for field_name, invalid_value in invalid_values.items():
            with self.subTest(field_name=field_name):
                with self.assertRaises(TypeError) as error:
                    Conv2dLayer(make_config(**{field_name: invalid_value}))
                self.assertEqual(
                    str(error.exception),
                    f"{field_name} must be {expected_types[field_name]} for "
                    "Conv2dLayerConfig, "
                    f"got {type(invalid_value).__name__}",
                )

    def test_numeric_boundaries_and_invalid_values(self) -> None:
        valid = Conv2dLayer(
            make_config(
                input_dim=1,
                output_dim=1,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        )
        self.assertEqual(
            (
                valid.model.in_channels,
                valid.model.out_channels,
                valid.model.kernel_size,
                valid.model.stride,
                valid.model.padding,
            ),
            (1, 1, (1, 1), (1, 1), (0, 0)),
        )

        invalid_cases = (
            ("input_dim", 0, "input_dim must be greater than 0, received 0"),
            (
                "output_dim",
                -1,
                "output_dim must be greater than 0, received -1",
            ),
            (
                "kernel_size",
                0,
                "kernel_size must be >= 1, received 0",
            ),
            ("stride", 0, "stride must be >= 1, received 0"),
            ("padding", -1, "padding must be >= 0, received -1"),
        )
        for field_name, value, message in invalid_cases:
            with self.subTest(field_name=field_name):
                with self.assertRaises(ValueError) as error:
                    Conv2dLayer(make_config(**{field_name: value}))
                self.assertEqual(str(error.exception), message)


class Conv2dBehaviorTests(unittest.TestCase):
    def test_module_exposes_validator_and_exact_torch_configuration(self) -> None:
        model = Conv2dLayer(
            make_config(
                input_dim=2,
                output_dim=4,
                kernel_size=3,
                stride=2,
                padding=1,
                bias_flag=True,
            )
        )

        self.assertIs(Conv2dLayer.VALIDATOR, Conv2dLayerValidator)
        self.assertEqual(model.input_dim, 2)
        self.assertEqual(model.output_dim, 4)
        self.assertEqual(tuple(model.model.weight.shape), (4, 2, 3, 3))
        self.assertEqual(tuple(model.model.bias.shape), (4,))
        self.assertEqual(model.model.stride, (2, 2))
        self.assertEqual(model.model.padding, (1, 1))

    def test_exact_rectangular_multi_channel_convolution_and_batch_isolation(
        self,
    ) -> None:
        model = Conv2dLayer(
            make_config(
                input_dim=1,
                output_dim=2,
                kernel_size=2,
                stride=1,
                padding=0,
                bias_flag=True,
            )
        )
        with torch.no_grad():
            model.model.weight.copy_(
                torch.tensor(
                    [
                        [[[1.0, 0.0], [0.0, -1.0]]],
                        [[[0.0, 2.0], [-1.0, 0.0]]],
                    ]
                )
            )
            model.model.bias.copy_(torch.tensor([0.5, -0.5]))
        first = torch.tensor(
            [[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]]
        )
        inputs = torch.stack((first, first + 10.0))
        expected = torch.tensor(
            [
                [
                    [[-4.5, -4.5, -4.5], [-4.5, -4.5, -4.5]],
                    [[-1.5, -0.5, 0.5], [2.5, 3.5, 4.5]],
                ],
                [
                    [[-4.5, -4.5, -4.5], [-4.5, -4.5, -4.5]],
                    [[8.5, 9.5, 10.5], [12.5, 13.5, 14.5]],
                ],
            ]
        )

        output = model(inputs)
        first_alone = model(inputs[:1])
        second_alone = model(inputs[1:])

        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        torch.testing.assert_close(first_alone, expected[:1], rtol=0, atol=0)
        torch.testing.assert_close(second_alone, expected[1:], rtol=0, atol=0)
        self.assertEqual(output.shape, (2, 2, 2, 3))
        self.assertEqual(output.dtype, torch.float32)
        self.assertEqual(output.device.type, "cpu")
        self.assertTrue(torch.isfinite(output).all())

    def test_stride_padding_and_disabled_bias_have_exact_behavior(self) -> None:
        model = Conv2dLayer(
            make_config(
                input_dim=1,
                output_dim=1,
                kernel_size=1,
                stride=2,
                padding=1,
                bias_flag=False,
            )
        )
        with torch.no_grad():
            model.model.weight.fill_(2.0)
        inputs = torch.tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]])

        output = model(inputs)

        self.assertIsNone(model.model.bias)
        self.assertEqual(list(model.state_dict()), ["model.weight"])
        torch.testing.assert_close(
            output,
            torch.tensor([[[[0.0, 0.0, 0.0], [0.0, 10.0, 0.0]]]]),
            rtol=0,
            atol=0,
        )

    def test_train_eval_repeated_calls_and_strict_state_round_trip(self) -> None:
        config = make_config(
            input_dim=1,
            output_dim=1,
            kernel_size=2,
            stride=1,
            padding=0,
            bias_flag=True,
        )
        source = Conv2dLayer(config)
        target = Conv2dLayer(config)
        with torch.no_grad():
            source.model.weight.copy_(torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]]))
            source.model.bias.copy_(torch.tensor([-2.0]))
            target.model.weight.zero_()
            target.model.bias.zero_()
        inputs = torch.tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]])

        source.train()
        training_output = source(inputs)
        source.eval()
        evaluation_output = source(inputs)
        repeated_output = source(inputs)
        incompatible = target.load_state_dict(source.state_dict(), strict=True)
        restored_output = target(inputs)

        expected = torch.tensor([[[[35.0, 45.0]]]])
        torch.testing.assert_close(training_output, expected, rtol=0, atol=0)
        torch.testing.assert_close(evaluation_output, expected, rtol=0, atol=0)
        torch.testing.assert_close(repeated_output, expected, rtol=0, atol=0)
        torch.testing.assert_close(restored_output, expected, rtol=0, atol=0)
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])

    def test_float64_non_contiguous_backward_and_optimizer_step(self) -> None:
        model = Conv2dLayer(
            make_config(
                input_dim=2,
                output_dim=2,
                kernel_size=1,
                stride=1,
                padding=0,
                bias_flag=True,
            )
        ).double()
        with torch.no_grad():
            model.model.weight.copy_(
                torch.tensor(
                    [[[[1.0]], [[2.0]]], [[[3.0]], [[4.0]]]],
                    dtype=torch.float64,
                )
            )
            model.model.bias.copy_(torch.tensor([0.5, 1.0], dtype=torch.float64))
        inputs = (
            torch.arange(1.0, 25.0, dtype=torch.float64)
            .reshape(1, 2, 3, 4)
            .transpose(2, 3)
            .detach()
            .requires_grad_()
        )
        self.assertFalse(inputs.is_contiguous())
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
        original_weight = model.model.weight.detach().clone()
        original_bias = model.model.bias.detach().clone()

        output = model(inputs)
        loss = output.square().mean()
        loss.backward()

        self.assertEqual(output.dtype, torch.float64)
        for gradient in (
            inputs.grad,
            model.model.weight.grad,
            model.model.bias.grad,
        ):
            assert gradient is not None
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(torch.count_nonzero(gradient).item(), 0)

        optimizer.step()

        self.assertFalse(torch.equal(model.model.weight, original_weight))
        self.assertFalse(torch.equal(model.model.bias, original_bias))

    def test_forward_rejects_non_tensor_rank_channel_and_kernel_mismatches(
        self,
    ) -> None:
        model = Conv2dLayer(
            make_config(
                input_dim=2,
                output_dim=1,
                kernel_size=3,
                stride=1,
                padding=0,
            )
        )

        with self.assertRaises(TypeError) as type_error:
            model([[[[1.0]]]])
        self.assertEqual(
            str(type_error.exception),
            "Input Error: forward input must be a Tensor, received list.",
        )

        for invalid in (
            torch.ones(2, 3, 4),
            torch.ones(1, 2, 3, 4, 5),
        ):
            with self.subTest(shape=tuple(invalid.shape)):
                with self.assertRaises(ValueError) as rank_error:
                    model(invalid)
                self.assertEqual(
                    str(rank_error.exception),
                    "Input Error: Conv2dLayer expects a 4D input tensor "
                    "(batch, channels, height, width), received a "
                    f"{invalid.dim()}D tensor with shape "
                    f"{tuple(invalid.shape)}.",
                )

        with self.assertRaises(RuntimeError) as channel_error:
            model(torch.ones(1, 1, 4, 5))
        self.assertIn("expected input", str(channel_error.exception))
        self.assertIn("to have 2 channels", str(channel_error.exception))

        with self.assertRaises(RuntimeError) as geometry_error:
            model(torch.ones(1, 2, 2, 4))
        self.assertIn(
            "Kernel size can't be greater than actual input size",
            str(geometry_error.exception),
        )


class Conv2dInterfaceTests(unittest.TestCase):
    def test_interface_exports_only_config_without_dynamic_shortcuts(self) -> None:
        import emperor.convs as convs

        self.assertEqual(convs.__all__, ("Conv2dLayerConfig",))
        self.assertIs(convs.Conv2dLayerConfig, Conv2dLayerConfig)
        self.assertFalse(hasattr(convs, "Conv2dLayer"))
        self.assertFalse(hasattr(convs, "__getattr__"))
        self.assertFalse(hasattr(convs, "_LAZY_EXPORTS"))

    def test_explicit_interface_is_lightweight_and_build_loads_implementation(
        self,
    ) -> None:
        script = """\
import json
import sys

import emperor.convs as convs

after_interface_import = sorted(
    name for name in sys.modules if name.startswith("emperor.convs.")
)
runtime_before_build = {
    "layer": "emperor.convs._layer" in sys.modules,
    "validation": "emperor.convs._validation" in sys.modules,
    "torch": "torch" in sys.modules,
}
config = convs.Conv2dLayerConfig(
    input_dim=2,
    output_dim=3,
    kernel_size=1,
    stride=1,
    padding=0,
    bias_flag=True,
)
model = config.build()
print(json.dumps({
    "all": convs.__all__,
    "after_interface_import": after_interface_import,
    "runtime_before_build": runtime_before_build,
    "runtime_after_build": {
        "layer": "emperor.convs._layer" in sys.modules,
        "validation": "emperor.convs._validation" in sys.modules,
        "torch": "torch" in sys.modules,
    },
    "config_module": convs.Conv2dLayerConfig.__module__,
    "model_module": type(model).__module__,
    "shortcut_attributes": {
        "__getattr__": hasattr(convs, "__getattr__"),
        "_LAZY_EXPORTS": hasattr(convs, "_LAZY_EXPORTS"),
    },
}))
"""

        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
        )
        result = json.loads(completed.stdout)

        self.assertEqual(
            result,
            {
                "all": ["Conv2dLayerConfig"],
                "after_interface_import": ["emperor.convs._config"],
                "runtime_before_build": {
                    "layer": False,
                    "validation": False,
                    "torch": False,
                },
                "runtime_after_build": {
                    "layer": True,
                    "validation": True,
                    "torch": True,
                },
                "config_module": "emperor.convs._config",
                "model_module": "emperor.convs._layer",
                "shortcut_attributes": {
                    "__getattr__": False,
                    "_LAZY_EXPORTS": False,
                },
            },
        )

    def test_removed_implementation_import_fails(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                "from emperor.convs import Conv2dLayer",
            ],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("ImportError", completed.stderr)


if __name__ == "__main__":
    unittest.main()

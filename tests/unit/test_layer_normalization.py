import math
import unittest
from copy import deepcopy
from io import BytesIO
from itertools import product

import torch

from emperor.layers import LayerConfig, LayerNormPositionOptions, NormalizationOptions

ELEMENTWISE_OPTIONS = (
    NormalizationOptions.DYNAMIC_TANH,
    NormalizationOptions.DERF,
    NormalizationOptions.DYISRU,
)


def make_normalization(option, input_dim=5, output_dim=5, position=None):
    return LayerConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        layer_norm_position=position or LayerNormPositionOptions.DEFAULT,
        normalization=option,
    ).build_normalization()


def reference_normalization(option, hidden, parameters):
    """Direct equations, independent of the production module implementations."""
    match option:
        case NormalizationOptions.RMS_NORM:
            transformed = (
                hidden / (hidden.square().mean(-1, keepdim=True) + 1e-5).sqrt()
            )
        case NormalizationOptions.LAYER_NORM:
            centered = hidden - hidden.mean(-1, keepdim=True)
            transformed = (
                centered / (centered.square().mean(-1, keepdim=True) + 1e-5).sqrt()
            )
        case NormalizationOptions.DYNAMIC_TANH:
            transformed = (parameters["alpha"] * hidden).tanh()
        case NormalizationOptions.DERF:
            transformed = (parameters["alpha"] * hidden + parameters["shift"]).erf()
        case NormalizationOptions.DYISRU:
            beta = parameters["raw_beta"].exp().log1p()
            transformed = (
                math.sqrt(hidden.shape[-1]) * hidden / (beta + hidden.square()).sqrt()
            )
        case _:
            raise AssertionError(f"Missing reference for {option}")
    return transformed * parameters["weight"] + parameters.get("bias", 0)


class TestLayerNormalization(unittest.TestCase):
    def test_outputs_and_all_parameter_gradients_match_equations(self):
        for option, shape in product(NormalizationOptions, ((5,), (2, 5), (2, 3, 5))):
            with self.subTest(option=option, shape=shape):
                module = make_normalization(option).double()
                with torch.no_grad():
                    module.weight.copy_(torch.linspace(0.75, 1.25, 5))
                    if hasattr(module, "bias"):
                        module.bias.copy_(torch.linspace(-0.2, 0.3, 5))
                    if hasattr(module, "alpha"):
                        module.alpha.fill_(0.7)
                    if hasattr(module, "shift"):
                        module.shift.fill_(-0.3)
                    if hasattr(module, "raw_beta"):
                        module.raw_beta.fill_(math.log(math.expm1(2.0)))
                hidden = torch.linspace(-3, 4, math.prod(shape), dtype=torch.float64)
                hidden = hidden.reshape(shape).requires_grad_()
                reference_hidden = hidden.detach().clone().requires_grad_()
                reference_parameters = {
                    name: parameter.detach().clone().requires_grad_()
                    for name, parameter in module.named_parameters()
                }

                actual = module(hidden)
                expected = reference_normalization(
                    option, reference_hidden, reference_parameters
                )

                torch.testing.assert_close(actual, expected)
                loss_weights = torch.linspace(0.2, 0.9, hidden.numel()).reshape(shape)
                actual_gradients = torch.autograd.grad(
                    (actual * loss_weights).sum(), (hidden, *module.parameters())
                )
                expected_gradients = torch.autograd.grad(
                    (expected * loss_weights).sum(),
                    (reference_hidden, *reference_parameters.values()),
                )
                for actual_gradient, expected_gradient in zip(
                    actual_gradients, expected_gradients, strict=True
                ):
                    torch.testing.assert_close(actual_gradient, expected_gradient)

    def test_elementwise_parameters_start_at_published_values(self):
        expected_scalars = {
            NormalizationOptions.DYNAMIC_TANH: {"alpha": 0.5},
            NormalizationOptions.DERF: {"alpha": 0.5, "shift": 0.0},
            NormalizationOptions.DYISRU: {"raw_beta": math.log(math.expm1(4.0))},
        }
        for option in ELEMENTWISE_OPTIONS:
            with self.subTest(option=option):
                module = make_normalization(option)
                torch.testing.assert_close(module.weight, torch.ones(5))
                torch.testing.assert_close(module.bias, torch.zeros(5))
                self.assertEqual(
                    set(dict(module.named_parameters())),
                    {"weight", "bias", *expected_scalars[option]},
                )
                for name, expected in expected_scalars[option].items():
                    parameter = getattr(module, name)
                    self.assertEqual(parameter.numel(), 1)
                    self.assertTrue(parameter.requires_grad)
                    self.assertAlmostEqual(parameter.item(), expected, places=6)

    def test_rectangular_layers_resolve_the_feature_dimension_for_each_position(self):
        for option, (position, dimension) in product(
            NormalizationOptions,
            (
                (LayerNormPositionOptions.BEFORE, 3),
                (LayerNormPositionOptions.DEFAULT, 5),
                (LayerNormPositionOptions.AFTER, 5),
            ),
        ):
            with self.subTest(option=option, position=position):
                module = make_normalization(option, input_dim=3, position=position)
                self.assertEqual(module.weight.shape, (dimension,))
                hidden = torch.randn(2, 4, dimension)
                self.assertEqual(module(hidden).shape, hidden.shape)

    def test_elementwise_modules_preserve_dtype_with_finite_outputs_and_gradients(self):
        for option, dtype, matching_parameters in product(
            ELEMENTWISE_OPTIONS,
            (torch.float16, torch.bfloat16, torch.float32, torch.float64),
            (False, True),
        ):
            with self.subTest(
                option=option, dtype=dtype, matching_parameters=matching_parameters
            ):
                module = make_normalization(option)
                if matching_parameters:
                    module.to(dtype=dtype)
                # Includes zeros, small values, and large finite outliers.
                hidden = torch.tensor([-10000, -0.25, 0, 0.5, 10000], dtype=dtype)
                hidden.requires_grad_()
                reference = deepcopy(module).double()(hidden.detach().double())

                actual = module(hidden)

                self.assertEqual(actual.dtype, dtype)
                torch.testing.assert_close(actual, reference.to(dtype=dtype))
                gradients = torch.autograd.grad(
                    actual.sum(), (hidden, *module.parameters())
                )
                self.assertTrue(torch.isfinite(actual).all())
                for gradient in gradients:
                    self.assertTrue(torch.isfinite(gradient).all())

    def test_dyisru_handles_extreme_inputs_and_beta_underflow(self):
        for raw_beta in (-1000.0, 0.0, 1000.0):
            with self.subTest(raw_beta=raw_beta):
                module = make_normalization(NormalizationOptions.DYISRU)
                with torch.no_grad():
                    module.raw_beta.fill_(raw_beta)
                hidden = torch.tensor([-1e30, -0.5, 0, 0.5, 1e30], requires_grad=True)

                actual = module(hidden)

                self.assertTrue(torch.isfinite(actual).all())
                self.assertEqual(actual[2].item(), 0.0)
                torch.testing.assert_close(
                    actual[[0, 4]], torch.tensor([-math.sqrt(5), math.sqrt(5)])
                )
                gradients = torch.autograd.grad(
                    actual.sum(), (hidden, *module.parameters())
                )
                for gradient in gradients:
                    self.assertTrue(torch.isfinite(gradient).all())

    def test_elementwise_modules_reject_wrong_feature_shapes_and_nonfloating_inputs(
        self,
    ):
        for option in ELEMENTWISE_OPTIONS:
            module = make_normalization(option)
            for shape in ((), (2, 1), (2, 3, 4)):
                with self.subTest(option=option, shape=shape):
                    with self.assertRaisesRegex(ValueError, "last dimension 5"):
                        module(torch.ones(shape))
            with self.subTest(option=option):
                with self.assertRaisesRegex(TypeError, "floating-point"):
                    module(torch.ones(2, 5, dtype=torch.int64))

    def test_elementwise_modules_do_not_mix_rows_or_tokens(self):
        for option in ELEMENTWISE_OPTIONS:
            with self.subTest(option=option):
                module = make_normalization(option)
                hidden = torch.linspace(-2, 3, 30).reshape(2, 3, 5).transpose(0, 1)
                expected = module(hidden)
                changed = hidden.clone()
                changed[0, 0, 0] = 100.0
                actual = module(changed)
                actual[0, 0, 0] = expected[0, 0, 0]
                torch.testing.assert_close(actual, expected)
                torch.testing.assert_close(
                    module(hidden.reshape(-1, 5)).reshape_as(hidden), expected
                )
                self.assertEqual(module(torch.empty(2, 0, 5)).shape, (2, 0, 5))

    def test_checkpoint_round_trip_preserves_learned_parameters_and_outputs(self):
        for option in NormalizationOptions:
            with self.subTest(option=option):
                module = make_normalization(option).double()
                with torch.no_grad():
                    for parameter in module.parameters():
                        parameter.add_(0.3)
                hidden = torch.randn(2, 3, 5, dtype=torch.float64)
                checkpoint = BytesIO()
                torch.save(module.state_dict(), checkpoint)
                checkpoint.seek(0)
                restored = make_normalization(option).double()
                restored.load_state_dict(
                    torch.load(checkpoint, weights_only=True), strict=True
                )

                torch.testing.assert_close(restored(hidden), module(hidden))
                self.assertEqual(set(restored.state_dict()), set(module.state_dict()))


if __name__ == "__main__":
    unittest.main()

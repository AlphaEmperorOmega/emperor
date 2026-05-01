import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from emperor.base.enums import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
)
from emperor.base.layer import LayerConfig, LayerStackConfig
from emperor.linears.core.config import LinearLayerConfig
from emperor.linears.options import LinearOptions
from emperor.augmentations.adaptive_parameters import DynamicDiagonalOptions
from emperor.augmentations.adaptive_parameters.core.diagonal import (
    AntiDynamicDiagonal,
    CombinedDynamicDiagonal,
    DynamicDiagonalAbstract,
    DynamicDiagonalConfig,
    StandardDynamicDiagonal,
)


class TestDynamicDiagonalHandlers(unittest.TestCase):
    def preset(
        self,
        input_dim: int = 8,
        hidden_dim: int = 16,
        output_dim: int = 4,
        bias_flag: bool = True,
        model_type: DynamicDiagonalOptions = DynamicDiagonalOptions.DIAGONAL,
    ) -> DynamicDiagonalConfig:
        return DynamicDiagonalConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            model_type=model_type,
            model_config=LayerStackConfig(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                num_layers=1,
                last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
                apply_output_pipeline_flag=True,
                layer_config=LayerConfig(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    activation=ActivationOptions.RELU,
                    layer_norm_position=LayerNormPositionOptions.DISABLED,
                    residual_flag=False,
                    dropout_probability=0.0,
                    gate_config=None,
                    halting_config=None,
                    shared_halting_flag=False,
                    layer_model_config=LinearLayerConfig(
                        model_type=LinearOptions.LINEAR,
                        input_dim=input_dim,
                        output_dim=output_dim,
                        bias_flag=bias_flag,
                    ),
                ),
            ),
        )

    def _set_generator_identity(self, generator_model) -> None:
        layers = (
            generator_model
            if isinstance(generator_model, nn.Sequential)
            else [generator_model]
        )
        for layer in layers:
            linear_model = layer.model
            with torch.no_grad():
                linear_model.weight_params.zero_()
                diagonal_dim = min(
                    linear_model.weight_params.size(0),
                    linear_model.weight_params.size(1),
                )
                linear_model.weight_params[:diagonal_dim, :diagonal_dim].copy_(
                    torch.eye(diagonal_dim)
                )
                if linear_model.bias_params is not None:
                    linear_model.bias_params.zero_()

    def test_standard_dynamic_diagonal_forward(self):
        batch_size = 2
        input_dims = [8, 4, 6]
        output_dims = [4, 8, 6]

        for input_dim, output_dim in zip(input_dims, output_dims):
            with self.subTest(input_dim=input_dim, output_dim=output_dim):
                cfg = self.preset(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    model_type=DynamicDiagonalOptions.DIAGONAL,
                )
                model = StandardDynamicDiagonal(cfg)
                logits = torch.randn(batch_size, input_dim)
                weight_params = torch.randn(input_dim, output_dim)
                output = model(weight_params, logits)

                self.assertEqual(output.shape, (batch_size, input_dim, output_dim))
                self.assertIsInstance(output, torch.Tensor)

    def test_anti_dynamic_diagonal_forward(self):
        batch_size = 2
        input_dims = [8, 4, 6]
        output_dims = [4, 8, 6]

        for input_dim, output_dim in zip(input_dims, output_dims):
            with self.subTest(input_dim=input_dim, output_dim=output_dim):
                cfg = self.preset(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    model_type=DynamicDiagonalOptions.ANTI_DIAGONAL,
                )
                model = AntiDynamicDiagonal(cfg)
                logits = torch.randn(batch_size, input_dim)
                weight_params = torch.randn(input_dim, output_dim)
                output = model(weight_params, logits)

                self.assertEqual(output.shape, (batch_size, input_dim, output_dim))
                self.assertIsInstance(output, torch.Tensor)

    def test_combined_dynamic_diagonal_forward(self):
        batch_size = 2
        input_dims = [8, 4, 6]
        output_dims = [4, 8, 6]

        for input_dim, output_dim in zip(input_dims, output_dims):
            with self.subTest(input_dim=input_dim, output_dim=output_dim):
                cfg = self.preset(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    model_type=DynamicDiagonalOptions.DIAGONAL_AND_ANTI_DIAGONAL,
                )
                model = CombinedDynamicDiagonal(cfg)
                logits = torch.randn(batch_size, input_dim)
                weight_params = torch.randn(input_dim, output_dim)
                output = model(weight_params, logits)

                self.assertEqual(output.shape, (batch_size, input_dim, output_dim))
                self.assertIsInstance(output, torch.Tensor)

    def test_standard_dynamic_diagonal_adds_diagonal_matrix(self):
        cfg = self.preset(
            input_dim=3,
            output_dim=3,
            model_type=DynamicDiagonalOptions.DIAGONAL,
        )
        model = StandardDynamicDiagonal(cfg)
        self._set_generator_identity(model.model)
        logits = torch.tensor([[1.0, 2.0, 3.0], [4.0, 0.5, 2.5]])
        weight_params = torch.zeros(3, 3)

        output = model(weight_params, logits)
        expected = torch.diag_embed(logits)

        self.assertTrue(torch.allclose(output, expected, atol=1e-6))

    def test_anti_dynamic_diagonal_adds_flipped_diagonal_matrix(self):
        cfg = self.preset(
            input_dim=3,
            output_dim=3,
            model_type=DynamicDiagonalOptions.ANTI_DIAGONAL,
        )
        model = AntiDynamicDiagonal(cfg)
        self._set_generator_identity(model.model)
        logits = torch.tensor([[1.0, 2.0, 3.0], [4.0, 0.5, 2.5]])
        weight_params = torch.zeros(3, 3)

        output = model(weight_params, logits)
        expected = torch.diag_embed(logits).flip(dims=[2])

        self.assertTrue(torch.allclose(output, expected, atol=1e-6))

    def test_combined_dynamic_diagonal_adds_diagonal_and_anti_diagonal(self):
        cfg = self.preset(
            input_dim=3,
            output_dim=3,
            model_type=DynamicDiagonalOptions.DIAGONAL_AND_ANTI_DIAGONAL,
        )
        model = CombinedDynamicDiagonal(cfg)
        self._set_generator_identity(model.diagonal_model.model)
        self._set_generator_identity(model.anti_diagonal_model.model)
        logits = torch.tensor([[1.0, 2.0, 3.0], [4.0, 0.5, 2.5]])
        weight_params = torch.zeros(3, 3)

        output = model(weight_params, logits)
        expected = torch.diag_embed(logits) + torch.diag_embed(logits).flip(dims=[2])

        self.assertTrue(torch.allclose(output, expected, atol=1e-6))

    def test_compute_diagonal_matrix_pads_rectangular_shape(self):
        cfg = self.preset(
            input_dim=3,
            output_dim=5,
            model_type=DynamicDiagonalOptions.DIAGONAL,
        )
        model = StandardDynamicDiagonal(cfg)
        self._set_generator_identity(model.model)
        logits = torch.tensor([[1.0, 2.0, 3.0], [4.0, 0.5, 2.5]])

        output = model._compute_diagonal_matrix(logits)
        expected = F.pad(torch.diag_embed(logits), (0, 2, 0, 0))

        self.assertEqual(output.shape, (2, 3, 5))
        self.assertTrue(torch.allclose(output, expected, atol=1e-6))

    def test_build_creates_model_for_each_option(self):
        input_dim = 8
        output_dim = 4

        for option in DynamicDiagonalOptions:
            with self.subTest(option=option):
                cfg = self.preset(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    model_type=option,
                )
                if option == DynamicDiagonalOptions.DISABLED:
                    with self.assertRaises(ValueError):
                        cfg.build()
                else:
                    model = cfg.build()
                    self.assertIsInstance(model, DynamicDiagonalAbstract)

    def test_invalid_dimensions_raise(self):
        cfg = DynamicDiagonalConfig(
            input_dim=0,
            output_dim=4,
            model_type=DynamicDiagonalOptions.DIAGONAL,
            model_config=self.preset().model_config,
        )

        with self.assertRaises(ValueError):
            cfg.build()

    def test_validate_generator_model_raises_on_unknown_generator_type(self):
        class InvalidGeneratorConfig:
            def build(self, overrides):
                return nn.Identity()

        cfg = self.preset(
            input_dim=3,
            output_dim=3,
            model_type=DynamicDiagonalOptions.DIAGONAL,
        )
        model = StandardDynamicDiagonal(cfg)
        model.model_config = InvalidGeneratorConfig()

        with self.assertRaises(TypeError):
            model._init_model()

    def test_gradients_flow(self):
        batch_size = 2
        input_dim = 8
        output_dim = 4

        for option in DynamicDiagonalOptions:
            with self.subTest(option=option):
                cfg = self.preset(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    model_type=option,
                )
                if option == DynamicDiagonalOptions.DISABLED:
                    with self.assertRaises(ValueError):
                        cfg.build()
                    continue

                model = cfg.build()
                logits = torch.randn(batch_size, input_dim, requires_grad=True)
                weight_params = torch.randn(input_dim, output_dim)
                output = model(weight_params, logits)
                output.sum().backward()

                grads = [param.grad for param in model.parameters() if param.requires_grad]
                non_none_grads = [grad for grad in grads if grad is not None]
                self.assertTrue(len(non_none_grads) > 0)

    def test_option_matrix_forward_shapes(self):
        batch_size = 2
        input_dim = 8
        output_dim = 4
        weight_params = torch.randn(input_dim, output_dim)

        for option in DynamicDiagonalOptions:
            if option == DynamicDiagonalOptions.DISABLED:
                continue

            with self.subTest(option=option):
                cfg = self.preset(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    model_type=option,
                )
                model = cfg.build()
                logits = torch.randn(batch_size, input_dim)
                output = model(weight_params, logits)
                self.assertEqual(output.shape, (batch_size, input_dim, output_dim))

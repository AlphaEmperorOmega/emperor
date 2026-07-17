import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from emperor.augmentations.adaptive_parameters import (
    AntiDynamicDiagonalConfig,
    CombinedDynamicDiagonalConfig,
    DynamicDiagonalConfig,
    StandardDynamicDiagonalConfig,
)
from emperor.augmentations.adaptive_parameters._diagonals.base import (
    DynamicDiagonalAbstract,
)
from emperor.augmentations.adaptive_parameters._diagonals.variants.anti import (
    AntiDynamicDiagonal,
)
from emperor.augmentations.adaptive_parameters._diagonals.variants.combined import (
    CombinedDynamicDiagonal,
)
from emperor.augmentations.adaptive_parameters._diagonals.variants.standard import (
    StandardDynamicDiagonal,
)
from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerConfig,
    LayerNormPositionOptions,
    LayerStack,
    LayerStackConfig,
    ResidualConnectionOptions,
)
from emperor.linears import LinearLayerConfig


class TestDynamicDiagonalHandlers(unittest.TestCase):
    def preset(
        self,
        config_cls: type[DynamicDiagonalConfig] = StandardDynamicDiagonalConfig,
        input_dim: int = 8,
        hidden_dim: int = 16,
        output_dim: int = 4,
        bias_flag: bool = True,
    ) -> DynamicDiagonalConfig:
        return config_cls(
            input_dim=input_dim,
            output_dim=output_dim,
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
                    residual_connection_option=ResidualConnectionOptions.DISABLED,
                    dropout_probability=0.0,
                    gate_config=None,
                    halting_config=None,
                    layer_model_config=LinearLayerConfig(
                        input_dim=input_dim,
                        output_dim=output_dim,
                        bias_flag=bias_flag,
                    ),
                ),
            ),
        )

    def diagonal_cases(
        self,
    ) -> list[tuple[type[DynamicDiagonalConfig], type]]:
        return [
            (StandardDynamicDiagonalConfig, StandardDynamicDiagonal),
            (AntiDynamicDiagonalConfig, AntiDynamicDiagonal),
            (CombinedDynamicDiagonalConfig, CombinedDynamicDiagonal),
        ]

    def _set_generator_identity(self, generator_model) -> None:
        layers = (
            generator_model
            if isinstance(generator_model, (nn.Sequential, LayerStack))
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
                    config_cls=StandardDynamicDiagonalConfig,
                    input_dim=input_dim,
                    output_dim=output_dim,
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
                    config_cls=AntiDynamicDiagonalConfig,
                    input_dim=input_dim,
                    output_dim=output_dim,
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
                    config_cls=CombinedDynamicDiagonalConfig,
                    input_dim=input_dim,
                    output_dim=output_dim,
                )
                model = CombinedDynamicDiagonal(cfg)
                logits = torch.randn(batch_size, input_dim)
                weight_params = torch.randn(input_dim, output_dim)
                output = model(weight_params, logits)

                self.assertEqual(output.shape, (batch_size, input_dim, output_dim))
                self.assertIsInstance(output, torch.Tensor)

    def test_standard_dynamic_diagonal_adds_diagonal_matrix(self):
        cfg = self.preset(
            config_cls=StandardDynamicDiagonalConfig,
            input_dim=3,
            output_dim=3,
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
            config_cls=AntiDynamicDiagonalConfig,
            input_dim=3,
            output_dim=3,
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
            config_cls=CombinedDynamicDiagonalConfig,
            input_dim=3,
            output_dim=3,
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
        cases = [
            (
                "wide",
                3,
                5,
                torch.tensor([[1.0, 2.0, 3.0], [4.0, 0.5, 2.5]]),
                (0, 2, 0, 0),
            ),
            (
                "tall",
                5,
                3,
                torch.tensor(
                    [[1.0, 2.0, 3.0, 9.0, 8.0], [4.0, 0.5, 2.5, 7.0, 6.0]]
                ),
                (0, 0, 0, 2),
            ),
        ]

        for name, input_dim, output_dim, logits, padding in cases:
            with self.subTest(case=name):
                cfg = self.preset(
                    config_cls=StandardDynamicDiagonalConfig,
                    input_dim=input_dim,
                    output_dim=output_dim,
                )
                model = StandardDynamicDiagonal(cfg)
                self._set_generator_identity(model.model)
                output = model._compute_diagonal_matrix(logits)
                diagonal_values = logits[:, : min(input_dim, output_dim)]
                expected = F.pad(torch.diag_embed(diagonal_values), padding)

                self.assertEqual(output.shape, (2, input_dim, output_dim))
                self.assertTrue(torch.allclose(output, expected, atol=1e-6))

    def test_build_creates_model_for_each_leaf_config(self):
        input_dim = 8
        output_dim = 4

        for config_cls, model_cls in self.diagonal_cases():
            with self.subTest(config_cls=config_cls.__name__):
                cfg = self.preset(
                    config_cls=config_cls,
                    input_dim=input_dim,
                    output_dim=output_dim,
                )
                model = cfg.build()
                self.assertIsInstance(model, model_cls)
                self.assertIsInstance(model, DynamicDiagonalAbstract)

    def test_abstract_config_cannot_build(self):
        cfg = self.preset(config_cls=DynamicDiagonalConfig)
        with self.assertRaises(ValueError):
            cfg.build()

    def test_invalid_dimensions_raise(self):
        cfg = StandardDynamicDiagonalConfig(
            input_dim=0,
            output_dim=4,
            model_config=self.preset().model_config,
        )

        with self.assertRaises(ValueError):
            cfg.build()

    def test_validate_generator_model_raises_on_unknown_generator_type(self):
        class InvalidGeneratorConfig:
            def build(self, overrides):
                return nn.Identity()

        cfg = self.preset(
            config_cls=StandardDynamicDiagonalConfig,
            input_dim=3,
            output_dim=3,
        )
        model = StandardDynamicDiagonal(cfg)
        model.model_config = InvalidGeneratorConfig()

        with self.assertRaises(TypeError):
            model._init_model()

    def test_gradients_flow(self):
        batch_size = 2
        input_dim = 8
        output_dim = 4

        for config_cls, _ in self.diagonal_cases():
            with self.subTest(config_cls=config_cls.__name__):
                cfg = self.preset(
                    config_cls=config_cls,
                    input_dim=input_dim,
                    output_dim=output_dim,
                )
                model = cfg.build()
                logits = torch.randn(batch_size, input_dim, requires_grad=True)
                weight_params = torch.randn(input_dim, output_dim)
                output = model(weight_params, logits)
                output.sum().backward()

                grads = [
                    param.grad for param in model.parameters() if param.requires_grad
                ]
                non_none_grads = [grad for grad in grads if grad is not None]
                self.assertTrue(len(non_none_grads) > 0)

    def test_option_matrix_forward_shapes(self):
        batch_size = 2
        input_dim = 8
        output_dim = 4
        weight_params = torch.randn(input_dim, output_dim)

        for config_cls, _ in self.diagonal_cases():
            with self.subTest(config_cls=config_cls.__name__):
                cfg = self.preset(
                    config_cls=config_cls,
                    input_dim=input_dim,
                    output_dim=output_dim,
                )
                model = cfg.build()
                logits = torch.randn(batch_size, input_dim)
                output = model(weight_params, logits)
                self.assertEqual(output.shape, (batch_size, input_dim, output_dim))


if __name__ == "__main__":
    unittest.main()

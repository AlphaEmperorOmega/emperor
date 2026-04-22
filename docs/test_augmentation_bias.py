import unittest
from unittest import mock

import torch
import torch.nn as nn

from emperor.base.enums import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
)
from emperor.base.layer import LayerConfig, LayerStackConfig
from emperor.linears.core.config import LinearLayerConfig
from emperor.linears.options import LinearOptions
from emperor.augmentations.adaptive_parameters.options import DynamicBiasOptions
from emperor.augmentations.adaptive_parameters.core.bias import (
    AffineTransformDynamicBias,
    DynamicBiasAbstract,
    DynamicBiasConfig,
    ElementwiseDynamicBias,
    GatedDynamicBias,
    GeneratorDynamicBias,
    WeightedBankDynamicBias,
)


class ConstantGenerator(nn.Module):
    def __init__(self, output: torch.Tensor):
        super().__init__()
        self.register_buffer("output", output)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        batch_size = logits.size(0)
        if self.output.size(0) != batch_size:
            raise ValueError(
                f"ConstantGenerator expected batch size {self.output.size(0)}, received {batch_size}."
            )
        return self.output


def _mock_init_generator_model(
    model: DynamicBiasAbstract,
    overrides: LayerStackConfig | None = None,
) -> nn.Module:
    return nn.Sequential(nn.Linear(model.input_dim, overrides.output_dim))


class TestDynamicBiasHandlers(unittest.TestCase):
    def setUp(self):
        self.generator_patcher = mock.patch.object(
            DynamicBiasAbstract,
            "_init_generator_model",
            new=_mock_init_generator_model,
        )
        self.generator_patcher.start()

    def tearDown(self):
        self.generator_patcher.stop()

    def preset(
        self,
        input_dim: int = 8,
        hidden_dim: int = 16,
        output_dim: int = 4,
        bias_flag: bool = True,
        model_type: DynamicBiasOptions = DynamicBiasOptions.SCALE_AND_OFFSET,
        bank_expansion_factor: int | None = None,
    ) -> DynamicBiasConfig:
        return DynamicBiasConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            bias_flag=bias_flag,
            model_type=model_type,
            bank_expansion_factor=bank_expansion_factor,
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

    def test_affine_transform_dynamic_bias_forward(self):
        batch_size = 2
        input_dim = 8
        output_dim = 4
        cfg = self.preset(
            input_dim=input_dim,
            output_dim=output_dim,
            model_type=DynamicBiasOptions.SCALE_AND_OFFSET,
        )
        model = AffineTransformDynamicBias(cfg)
        logits = torch.randn(batch_size, input_dim)
        bias_params = torch.randn(output_dim)
        output = model(bias_params, logits)

        self.assertEqual(output.shape, (batch_size, output_dim))
        self.assertIsInstance(output, torch.Tensor)

    def test_elementwise_dynamic_bias_forward(self):
        batch_size = 2
        input_dim = 8
        output_dim = 4
        cfg = self.preset(
            input_dim=input_dim,
            output_dim=output_dim,
            model_type=DynamicBiasOptions.ELEMENT_WISE_OFFSET,
        )
        model = ElementwiseDynamicBias(cfg)
        logits = torch.randn(batch_size, input_dim)
        bias_params = torch.randn(output_dim)
        output = model(bias_params, logits)

        self.assertEqual(output.shape, (batch_size, output_dim))
        self.assertIsInstance(output, torch.Tensor)

    def test_gated_dynamic_bias_forward(self):
        batch_size = 2
        input_dim = 8
        output_dim = 4
        cfg = self.preset(
            input_dim=input_dim,
            output_dim=output_dim,
            model_type=DynamicBiasOptions.GATED,
        )
        model = GatedDynamicBias(cfg)
        logits = torch.randn(batch_size, input_dim)
        bias_params = torch.randn(output_dim)
        output = model(bias_params, logits)

        self.assertEqual(output.shape, (batch_size, output_dim))
        self.assertIsInstance(output, torch.Tensor)

    def test_generator_dynamic_bias_forward(self):
        batch_size = 2
        input_dim = 8
        output_dim = 4
        cfg = self.preset(
            input_dim=input_dim,
            output_dim=output_dim,
            model_type=DynamicBiasOptions.DYNAMIC_PARAMETERS,
        )
        model = GeneratorDynamicBias(cfg)
        logits = torch.randn(batch_size, input_dim)
        output = model(None, logits)

        self.assertEqual(output.shape, (batch_size, output_dim))
        self.assertIsInstance(output, torch.Tensor)

    def test_weighted_bank_dynamic_bias_forward(self):
        batch_size = 2
        input_dim = 8
        output_dim = 4
        cfg = self.preset(
            input_dim=input_dim,
            output_dim=output_dim,
            model_type=DynamicBiasOptions.WEIGHTED_BANK,
            bank_expansion_factor=3,
        )
        model = WeightedBankDynamicBias(cfg)
        logits = torch.randn(batch_size, input_dim)
        output = model(None, logits)

        self.assertEqual(output.shape, (batch_size, output_dim))
        self.assertIsInstance(output, torch.Tensor)

    def test_affine_transform_dynamic_bias_mathematical_correctness(self):
        cfg = self.preset(
            input_dim=3,
            output_dim=2,
            model_type=DynamicBiasOptions.SCALE_AND_OFFSET,
        )
        model = AffineTransformDynamicBias(cfg)
        logits = torch.zeros(2, 3)
        bias_params = torch.tensor([1.0, -2.0])
        affine_parameters = torch.tensor([[2.0, 0.5], [-1.5, 1.0]])
        model.scalar_offset_generator = ConstantGenerator(affine_parameters)

        output = model(bias_params, logits)
        expected = torch.tensor([[2.5, -3.5], [-0.5, 4.0]])

        self.assertTrue(torch.allclose(output, expected, atol=1e-6))

    def test_elementwise_dynamic_bias_adds_generated_offset(self):
        cfg = self.preset(
            input_dim=3,
            output_dim=3,
            model_type=DynamicBiasOptions.ELEMENT_WISE_OFFSET,
        )
        model = ElementwiseDynamicBias(cfg)
        logits = torch.zeros(2, 3)
        bias_params = torch.tensor([1.0, 2.0, 3.0])
        generated_bias_offset = torch.tensor(
            [[0.5, -1.0, 2.0], [-0.5, 1.5, -2.0]]
        )
        model.generator_model = ConstantGenerator(generated_bias_offset)

        output = model(bias_params, logits)
        expected = torch.tensor([[1.5, 1.0, 5.0], [0.5, 3.5, 1.0]])

        self.assertTrue(torch.allclose(output, expected, atol=1e-6))

    def test_gated_dynamic_bias_applies_sigmoid_gate(self):
        cfg = self.preset(
            input_dim=3,
            output_dim=2,
            model_type=DynamicBiasOptions.GATED,
        )
        model = GatedDynamicBias(cfg)
        logits = torch.zeros(2, 3)
        bias_params = torch.tensor([2.0, -4.0])
        gate_logits = torch.tensor([[0.0, 0.0], [2.0, -2.0]])
        model.gate_generator = ConstantGenerator(gate_logits)

        output = model(bias_params, logits)
        expected = bias_params.unsqueeze(0) * torch.sigmoid(gate_logits)

        self.assertTrue(torch.allclose(output, expected, atol=1e-6))

    def test_generator_dynamic_bias_returns_generated_parameters(self):
        cfg = self.preset(
            input_dim=3,
            output_dim=2,
            model_type=DynamicBiasOptions.DYNAMIC_PARAMETERS,
        )
        model = GeneratorDynamicBias(cfg)
        logits = torch.zeros(2, 3)
        generated_bias = torch.tensor([[1.0, -1.0], [0.5, 2.5]])
        model.bias_generator = ConstantGenerator(generated_bias)

        output = model(None, logits)

        self.assertTrue(torch.equal(output, generated_bias))

    def test_weighted_bank_dynamic_bias_uses_bank_distribution(self):
        cfg = self.preset(
            input_dim=3,
            output_dim=2,
            model_type=DynamicBiasOptions.WEIGHTED_BANK,
            bank_expansion_factor=2,
        )
        model = WeightedBankDynamicBias(cfg)
        logits = torch.zeros(2, 3)
        bank_logits = torch.tensor([[12.0, -12.0], [-12.0, 12.0]])
        model.distribution_generator = ConstantGenerator(bank_logits)
        with torch.no_grad():
            model.weight_bank.copy_(torch.tensor([[1.0, 0.0], [0.0, 2.0]]))

        output = model(None, logits)
        expected_distribution = torch.softmax(bank_logits, dim=-1)
        expected = torch.matmul(expected_distribution, model.weight_bank)

        self.assertTrue(torch.allclose(output, expected, atol=1e-6))
        self.assertFalse(torch.allclose(output[0], output[1], atol=1e-6))

    def test_bias_required_strategies_raise_when_bias_params_missing(self):
        logits = torch.randn(2, 8)
        required_options = [
            DynamicBiasOptions.SCALE_AND_OFFSET,
            DynamicBiasOptions.ELEMENT_WISE_OFFSET,
            DynamicBiasOptions.GATED,
        ]

        for option in required_options:
            with self.subTest(option=option):
                cfg = self.preset(model_type=option)
                model = cfg.build()
                with self.assertRaises(ValueError):
                    model(None, logits)

    def test_build_creates_model_for_each_option(self):
        input_dim = 8
        output_dim = 4

        for option in DynamicBiasOptions:
            with self.subTest(option=option):
                cfg = self.preset(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    model_type=option,
                    bank_expansion_factor=3
                    if option == DynamicBiasOptions.WEIGHTED_BANK
                    else None,
                )
                if option == DynamicBiasOptions.DISABLED:
                    with self.assertRaises(ValueError):
                        cfg.build()
                else:
                    model = cfg.build()
                    self.assertIsInstance(model, DynamicBiasAbstract)

    def test_gradients_flow(self):
        batch_size = 2
        input_dim = 8
        output_dim = 4

        for option in DynamicBiasOptions:
            with self.subTest(option=option):
                cfg = self.preset(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    model_type=option,
                    bank_expansion_factor=3
                    if option == DynamicBiasOptions.WEIGHTED_BANK
                    else None,
                )
                if option == DynamicBiasOptions.DISABLED:
                    with self.assertRaises(ValueError):
                        cfg.build()
                    continue

                model = cfg.build()
                logits = torch.randn(batch_size, input_dim, requires_grad=True)
                bias_params = (
                    None
                    if option
                    in {
                        DynamicBiasOptions.DYNAMIC_PARAMETERS,
                        DynamicBiasOptions.WEIGHTED_BANK,
                    }
                    else torch.randn(output_dim)
                )
                output = model(bias_params, logits)
                output.sum().backward()

                grads = [param.grad for param in model.parameters() if param.requires_grad]
                non_none_grads = [grad for grad in grads if grad is not None]
                self.assertTrue(len(non_none_grads) > 0)

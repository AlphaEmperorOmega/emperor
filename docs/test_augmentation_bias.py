import unittest

import torch
import torch.nn as nn

from emperor.base.enums import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
)
from emperor.base.layer import LayerConfig, LayerStackConfig
from emperor.base.layer.state import LayerState
from emperor.linears.core.config import LinearLayerConfig
from emperor.linears.options import LinearOptions
from emperor.augmentations.adaptive_parameters.options import (
    DynamicBiasOptions,
    WeightDecayScheduleOptions,
)
from emperor.augmentations.adaptive_parameters.core.bias import (
    AdditiveDynamicBias,
    AffineTransformDynamicBias,
    DynamicBiasAbstract,
    DynamicBiasConfig,
    GeneratorDynamicBias,
    MultiplicativeDynamicBias,
    SigmoidGatedDynamicBias,
    TanhGatedDynamicBias,
    WeightedBankDynamicBias,
)


class ConstantGenerator(nn.Module):
    def __init__(self, output: torch.Tensor):
        super().__init__()
        self.register_buffer("output", output)

    def forward(self, logits: torch.Tensor | LayerState) -> torch.Tensor | LayerState:
        if isinstance(logits, LayerState):
            input_tensor = logits.hidden
        else:
            input_tensor = logits

        batch_size = input_tensor.size(0)
        if self.output.size(0) != batch_size:
            raise ValueError(
                f"ConstantGenerator expected batch size {self.output.size(0)}, received {batch_size}."
            )
        if isinstance(logits, LayerState):
            logits.hidden = self.output
            return logits
        return self.output


class TestDynamicBiasHandlers(unittest.TestCase):
    def preset(
        self,
        input_dim: int = 8,
        hidden_dim: int = 16,
        output_dim: int = 4,
        bias_flag: bool = True,
        model_type: DynamicBiasOptions = DynamicBiasOptions.SCALE_AND_OFFSET,
        bank_expansion_factor: int | None = None,
        decay_schedule: WeightDecayScheduleOptions | None = None,
        decay_rate: float | None = None,
        decay_warmup_batches: int | None = None,
    ) -> DynamicBiasConfig:
        return DynamicBiasConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            bias_flag=bias_flag,
            model_type=model_type,
            bank_expansion_factor=bank_expansion_factor,
            decay_schedule=decay_schedule,
            decay_rate=decay_rate,
            decay_warmup_batches=decay_warmup_batches,
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

    def test_additive_dynamic_bias_forward(self):
        batch_size = 2
        input_dim = 8
        output_dim = 4
        cfg = self.preset(
            input_dim=input_dim,
            output_dim=output_dim,
            model_type=DynamicBiasOptions.ADDITIVE,
        )
        model = AdditiveDynamicBias(cfg)
        logits = torch.randn(batch_size, input_dim)
        bias_params = torch.randn(output_dim)
        output = model(bias_params, logits)

        self.assertEqual(output.shape, (batch_size, output_dim))
        self.assertIsInstance(output, torch.Tensor)

    def test_sigmoid_multiplicative_dynamic_bias_forward(self):
        batch_size = 2
        input_dim = 8
        output_dim = 4
        cfg = self.preset(
            input_dim=input_dim,
            output_dim=output_dim,
            model_type=DynamicBiasOptions.SIGMOID_MULTIPLICATIVE,
        )
        model = SigmoidGatedDynamicBias(cfg)
        logits = torch.randn(batch_size, input_dim)
        bias_params = torch.randn(output_dim)
        output = model(bias_params, logits)

        self.assertEqual(output.shape, (batch_size, output_dim))
        self.assertIsInstance(output, torch.Tensor)

    def test_tanh_multiplicative_dynamic_bias_forward(self):
        batch_size = 2
        input_dim = 8
        output_dim = 4
        cfg = self.preset(
            input_dim=input_dim,
            output_dim=output_dim,
            model_type=DynamicBiasOptions.TANH_MULTIPLICATIVE,
        )
        model = TanhGatedDynamicBias(cfg)
        logits = torch.randn(batch_size, input_dim)
        bias_params = torch.randn(output_dim)
        output = model(bias_params, logits)

        self.assertEqual(output.shape, (batch_size, output_dim))
        self.assertIsInstance(output, torch.Tensor)

    def test_multiplicative_dynamic_bias_forward(self):
        batch_size = 2
        input_dim = 8
        output_dim = 4
        cfg = self.preset(
            input_dim=input_dim,
            output_dim=output_dim,
            model_type=DynamicBiasOptions.MULTIPLICATIVE,
        )
        model = MultiplicativeDynamicBias(cfg)
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
        bias_params = torch.randn(output_dim)
        output = model(bias_params, logits)

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
        bias_params = torch.randn(output_dim)
        output = model(bias_params, logits)

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

    def test_additive_dynamic_bias_adds_generated_offset(self):
        cfg = self.preset(
            input_dim=3,
            output_dim=3,
            model_type=DynamicBiasOptions.ADDITIVE,
        )
        model = AdditiveDynamicBias(cfg)
        logits = torch.zeros(2, 3)
        bias_params = torch.tensor([1.0, 2.0, 3.0])
        generated_bias_offset = torch.tensor([[0.5, -1.0, 2.0], [-0.5, 1.5, -2.0]])
        model.generator_model = ConstantGenerator(generated_bias_offset)

        output = model(bias_params, logits)
        expected = torch.tensor([[1.5, 1.0, 5.0], [0.5, 3.5, 1.0]])

        self.assertTrue(torch.allclose(output, expected, atol=1e-6))

    def test_sigmoid_multiplicative_dynamic_bias_applies_sigmoid_gate(self):
        cfg = self.preset(
            input_dim=3,
            output_dim=2,
            model_type=DynamicBiasOptions.SIGMOID_MULTIPLICATIVE,
        )
        model = SigmoidGatedDynamicBias(cfg)
        logits = torch.zeros(2, 3)
        bias_params = torch.tensor([2.0, -4.0])
        gate_logits = torch.tensor([[0.0, 0.0], [2.0, -2.0]])
        model.gate_generator = ConstantGenerator(gate_logits)

        output = model(bias_params, logits)
        expected = bias_params.unsqueeze(0) * torch.sigmoid(gate_logits)

        self.assertTrue(torch.allclose(output, expected, atol=1e-6))

    def test_tanh_multiplicative_dynamic_bias_applies_tanh_gate(self):
        cfg = self.preset(
            input_dim=3,
            output_dim=2,
            model_type=DynamicBiasOptions.TANH_MULTIPLICATIVE,
        )
        model = TanhGatedDynamicBias(cfg)
        logits = torch.zeros(2, 3)
        bias_params = torch.tensor([2.0, -4.0])
        gate_logits = torch.tensor([[0.0, 0.0], [2.0, -2.0]])
        model.gate_generator = ConstantGenerator(gate_logits)

        output = model(bias_params, logits)
        expected = bias_params.unsqueeze(0) * torch.tanh(gate_logits)

        self.assertTrue(torch.allclose(output, expected, atol=1e-6))

    def test_multiplicative_dynamic_bias_applies_raw_scale(self):
        cfg = self.preset(
            input_dim=3,
            output_dim=2,
            model_type=DynamicBiasOptions.MULTIPLICATIVE,
        )
        model = MultiplicativeDynamicBias(cfg)
        logits = torch.zeros(2, 3)
        bias_params = torch.tensor([2.0, -4.0])
        bias_scale = torch.tensor([[0.5, -1.0], [2.0, 3.0]])
        model.scale_generator = ConstantGenerator(bias_scale)

        output = model(bias_params, logits)
        expected = bias_params.unsqueeze(0) * bias_scale

        self.assertTrue(torch.allclose(output, expected, atol=1e-6))

    def test_additive_dynamic_bias_applies_bias_decay(self):
        cfg = self.preset(
            input_dim=3,
            output_dim=2,
            model_type=DynamicBiasOptions.ADDITIVE,
            decay_schedule=WeightDecayScheduleOptions.MULTIPLICATIVE,
            decay_rate=0.5,
            decay_warmup_batches=0,
        )
        model = AdditiveDynamicBias(cfg)
        logits = torch.zeros(2, 3)
        bias_params = torch.tensor([2.0, -4.0])
        generated_bias_offset = torch.zeros(2, 2)
        model.generator_model = ConstantGenerator(generated_bias_offset)

        first_output = model(bias_params, logits)
        second_output = model(bias_params, logits)

        expected_first_output = bias_params.unsqueeze(0) + generated_bias_offset
        expected_second_output = 0.5 * bias_params.unsqueeze(0) + generated_bias_offset

        self.assertTrue(torch.allclose(first_output, expected_first_output, atol=1e-6))
        self.assertTrue(
            torch.allclose(second_output, expected_second_output, atol=1e-6)
        )

    def test_generator_dynamic_bias_returns_generated_parameters(self):
        cfg = self.preset(
            input_dim=3,
            output_dim=2,
            model_type=DynamicBiasOptions.DYNAMIC_PARAMETERS,
        )
        model = GeneratorDynamicBias(cfg)
        logits = torch.zeros(2, 3)
        bias_params = torch.tensor([7.0, 9.0])
        generated_bias = torch.tensor([[1.0, -1.0], [0.5, 2.5]])
        model.bias_generator = ConstantGenerator(generated_bias)

        output = model(bias_params, logits)

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
        bias_params = torch.tensor([3.0, -5.0])
        bank_logits = torch.tensor([[12.0, -12.0], [-12.0, 12.0]])
        model.distribution_generator = ConstantGenerator(bank_logits)
        with torch.no_grad():
            model.weight_bank.copy_(torch.tensor([[1.0, 0.0], [0.0, 2.0]]))

        output = model(bias_params, logits)
        expected_distribution = torch.softmax(bank_logits, dim=-1)
        expected = torch.matmul(expected_distribution, model.weight_bank)

        self.assertTrue(torch.allclose(output, expected, atol=1e-6))
        self.assertFalse(torch.allclose(output[0], output[1], atol=1e-6))

    def test_generator_dynamic_bias_ignores_passed_bias_tensor(self):
        cfg = self.preset(
            input_dim=3,
            output_dim=2,
            model_type=DynamicBiasOptions.DYNAMIC_PARAMETERS,
        )
        model = GeneratorDynamicBias(cfg)
        logits = torch.zeros(2, 3)
        generated_bias = torch.tensor([[1.0, -1.0], [0.5, 2.5]])
        model.bias_generator = ConstantGenerator(generated_bias)

        output_a = model(torch.tensor([7.0, 9.0]), logits)
        output_b = model(torch.tensor([-3.0, 4.0]), logits)

        self.assertTrue(torch.equal(output_a, generated_bias))
        self.assertTrue(torch.equal(output_b, generated_bias))

    def test_weighted_bank_dynamic_bias_ignores_passed_bias_tensor(self):
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

        output_a = model(torch.tensor([3.0, -5.0]), logits)
        output_b = model(torch.tensor([-1.0, 8.0]), logits)

        self.assertTrue(torch.allclose(output_a, output_b, atol=1e-6))

    def test_bias_required_strategies_raise_when_bias_params_missing(self):
        logits = torch.randn(2, 8)
        required_options = [
            DynamicBiasOptions.SCALE_AND_OFFSET,
            DynamicBiasOptions.ADDITIVE,
            DynamicBiasOptions.MULTIPLICATIVE,
            DynamicBiasOptions.SIGMOID_MULTIPLICATIVE,
            DynamicBiasOptions.TANH_MULTIPLICATIVE,
        ]

        for option in required_options:
            with self.subTest(option=option):
                cfg = self.preset(model_type=option)
                model = cfg.build()
                with self.assertRaises(ValueError):
                    model(None, logits)

    def test_non_bank_bias_strategies_reject_bank_expansion_factor(self):
        non_bank_options = [
            DynamicBiasOptions.SCALE_AND_OFFSET,
            DynamicBiasOptions.ADDITIVE,
            DynamicBiasOptions.MULTIPLICATIVE,
            DynamicBiasOptions.DYNAMIC_PARAMETERS,
            DynamicBiasOptions.SIGMOID_MULTIPLICATIVE,
            DynamicBiasOptions.TANH_MULTIPLICATIVE,
        ]

        for option in non_bank_options:
            with self.subTest(option=option):
                cfg = self.preset(model_type=option, bank_expansion_factor=2)
                with self.assertRaises(ValueError):
                    cfg.build()

    def test_weighted_bank_requires_positive_bank_expansion_factor(self):
        invalid_factors = [None, 0, -1]

        for factor in invalid_factors:
            with self.subTest(bank_expansion_factor=factor):
                cfg = self.preset(
                    model_type=DynamicBiasOptions.WEIGHTED_BANK,
                    bank_expansion_factor=factor,
                )
                with self.assertRaises(ValueError):
                    cfg.build()

    def test_bank_expansion_factor_variants(self):
        valid_factors = [1, 2, 3]
        invalid_factors = [0, -1]

        for option in DynamicBiasOptions:
            if option == DynamicBiasOptions.DISABLED:
                continue

            is_bank_type = option == DynamicBiasOptions.WEIGHTED_BANK
            for factor in [*invalid_factors, *valid_factors]:
                with self.subTest(option=option, bank_expansion_factor=factor):
                    cfg = self.preset(
                        model_type=option,
                        bank_expansion_factor=factor,
                    )
                    if not is_bank_type:
                        if factor is None:
                            model = cfg.build()
                            self.assertIsInstance(model, DynamicBiasAbstract)
                        else:
                            with self.assertRaises(ValueError):
                                cfg.build()
                    elif factor in invalid_factors or factor is None:
                        with self.assertRaises(ValueError):
                            cfg.build()
                    else:
                        model = cfg.build()
                        logits = torch.randn(2, 8)
                        bias_params = torch.randn(4)
                        output = model(bias_params, logits)
                        self.assertEqual(output.shape, (2, 4))

    def test_build_creates_model_for_each_option(self):
        input_dim = 8
        output_dim = 4

        for option in DynamicBiasOptions:
            with self.subTest(option=option):
                cfg = self.preset(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    model_type=option,
                    bank_expansion_factor=(
                        3 if option == DynamicBiasOptions.WEIGHTED_BANK else None
                    ),
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
                    bank_expansion_factor=(
                        3 if option == DynamicBiasOptions.WEIGHTED_BANK else None
                    ),
                )
                if option == DynamicBiasOptions.DISABLED:
                    with self.assertRaises(ValueError):
                        cfg.build()
                    continue

                model = cfg.build()
                logits = torch.randn(batch_size, input_dim, requires_grad=True)
                bias_params = torch.randn(output_dim)
                output = model(bias_params, logits)
                output.sum().backward()

                grads = [
                    param.grad for param in model.parameters() if param.requires_grad
                ]
                non_none_grads = [grad for grad in grads if grad is not None]
                self.assertTrue(len(non_none_grads) > 0)

    def test_bias_decay_schedule_options(self):
        output_dim = 4
        bias_params = torch.randn(output_dim)

        for schedule in WeightDecayScheduleOptions:
            with self.subTest(decay_schedule=schedule):
                cfg = self.preset(
                    output_dim=output_dim,
                    model_type=DynamicBiasOptions.ADDITIVE,
                    decay_schedule=schedule,
                    decay_rate=0.4,
                )
                model = cfg.build()

                if schedule == WeightDecayScheduleOptions.DISABLED:
                    result = model._maybe_apply_bias_decay(bias_params)
                    self.assertTrue(torch.equal(result, bias_params))
                else:
                    result = model._maybe_apply_bias_decay(bias_params)
                    self.assertTrue(torch.equal(result, bias_params))
                    result = model._maybe_apply_bias_decay(bias_params)
                    self.assertEqual(result.shape, bias_params.shape)
                    self.assertFalse(torch.equal(result, bias_params))

    def test_bias_decay_schedule_mathematical_correctness(self):
        output_dim = 4
        decay_rate = 0.3
        num_steps = 5
        bias_params = torch.randn(output_dim)

        schedule_expected_factor = {
            WeightDecayScheduleOptions.EXPONENTIAL: lambda step: torch.exp(
                torch.tensor(-decay_rate * step)
            ),
            WeightDecayScheduleOptions.LINEAR: lambda step: torch.clamp(
                torch.tensor(1.0 - decay_rate * step), min=0.0
            ),
            WeightDecayScheduleOptions.MULTIPLICATIVE: lambda step: torch.pow(
                torch.tensor(1.0 - decay_rate), torch.tensor(float(step))
            ),
        }

        for schedule, expected_factor_fn in schedule_expected_factor.items():
            with self.subTest(schedule=schedule):
                cfg = self.preset(
                    output_dim=output_dim,
                    model_type=DynamicBiasOptions.ADDITIVE,
                    decay_schedule=schedule,
                    decay_rate=decay_rate,
                )
                model = cfg.build()

                for step in range(num_steps):
                    result = model._maybe_apply_bias_decay(bias_params)
                    expected_factor = expected_factor_fn(step)
                    expected = bias_params * expected_factor
                    self.assertTrue(torch.allclose(result, expected, atol=1e-6))

    def test_bias_decay_warmup_delays_decay(self):
        output_dim = 4
        warmup_batches = 3
        bias_params = torch.randn(output_dim)
        cfg = self.preset(
            output_dim=output_dim,
            model_type=DynamicBiasOptions.ADDITIVE,
            decay_schedule=WeightDecayScheduleOptions.EXPONENTIAL,
            decay_rate=0.1,
            decay_warmup_batches=warmup_batches,
        )
        model = cfg.build()

        for step in range(warmup_batches):
            with self.subTest(warmup_step=step):
                result = model._maybe_apply_bias_decay(bias_params)
                self.assertTrue(torch.equal(result, bias_params))

        result = model._maybe_apply_bias_decay(bias_params)
        self.assertTrue(torch.equal(result, bias_params))

        result = model._maybe_apply_bias_decay(bias_params)
        self.assertFalse(torch.equal(result, bias_params))

    def test_bias_decay_schedule_raises_on_unknown_schedule(self):
        cfg = self.preset(
            model_type=DynamicBiasOptions.ADDITIVE,
            decay_schedule=WeightDecayScheduleOptions.EXPONENTIAL,
            decay_rate=0.1,
        )
        model = cfg.build()
        setattr(model, "decay_schedule_option", "invalid_schedule")
        bias_params = torch.randn(4)

        with self.assertRaises(ValueError):
            model._maybe_apply_bias_decay(bias_params)

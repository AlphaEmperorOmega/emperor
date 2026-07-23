import unittest

import torch
import torch.nn as nn

from emperor.augmentations.adaptive_parameters import (
    AdditiveDynamicBiasConfig,
    AffineTransformDynamicBiasConfig,
    BankExpansionFactorOptions,
    DynamicBiasConfig,
    GeneratorDynamicBiasConfig,
    MultiplicativeDynamicBiasConfig,
    SigmoidGatedDynamicBiasConfig,
    TanhGatedDynamicBiasConfig,
    WeightDecayScheduleOptions,
    WeightedBankDynamicBiasConfig,
)
from emperor.augmentations.adaptive_parameters._biases.variants.additive import (
    AdditiveDynamicBias,
)
from emperor.augmentations.adaptive_parameters._biases.variants.affine import (
    AffineTransformDynamicBias,
)
from emperor.augmentations.adaptive_parameters._biases.variants.gated import (
    SigmoidGatedDynamicBias,
    TanhGatedDynamicBias,
)
from emperor.augmentations.adaptive_parameters._biases.variants.generator import (
    GeneratorDynamicBias,
)
from emperor.augmentations.adaptive_parameters._biases.variants.multiplicative import (
    MultiplicativeDynamicBias,
)
from emperor.augmentations.adaptive_parameters._biases.variants.weighted_bank import (
    WeightedBankDynamicBias,
)
from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerConfig,
    LayerNormPositionOptions,
    LayerStackConfig,
    LayerState,
)
from emperor.linears import LinearLayerConfig


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
                "ConstantGenerator expected batch size "
                f"{self.output.size(0)}, received {batch_size}."
            )
        if isinstance(logits, LayerState):
            logits.hidden = self.output
            return logits
        return self.output


class TestDynamicBiasHandlers(unittest.TestCase):
    def preset(
        self,
        config_cls: type[DynamicBiasConfig] = AffineTransformDynamicBiasConfig,
        input_dim: int = 8,
        hidden_dim: int = 16,
        output_dim: int = 4,
        bias_flag: bool = True,
        bank_expansion_factor: BankExpansionFactorOptions | None = None,
        decay_schedule: WeightDecayScheduleOptions = (
            WeightDecayScheduleOptions.DISABLED
        ),
        decay_rate: float = 0.0,
        decay_warmup_batches: int = 0,
    ) -> DynamicBiasConfig:
        common_kwargs = dict(
            input_dim=input_dim,
            output_dim=output_dim,
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
                    residual_config=None,
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
        if config_cls is WeightedBankDynamicBiasConfig:
            return config_cls(
                **common_kwargs,
                bank_expansion_factor=bank_expansion_factor,
            )
        return config_cls(**common_kwargs)

    def bias_cases(self) -> list[tuple[type[DynamicBiasConfig], type]]:
        return [
            (AffineTransformDynamicBiasConfig, AffineTransformDynamicBias),
            (AdditiveDynamicBiasConfig, AdditiveDynamicBias),
            (MultiplicativeDynamicBiasConfig, MultiplicativeDynamicBias),
            (SigmoidGatedDynamicBiasConfig, SigmoidGatedDynamicBias),
            (TanhGatedDynamicBiasConfig, TanhGatedDynamicBias),
            (GeneratorDynamicBiasConfig, GeneratorDynamicBias),
            (WeightedBankDynamicBiasConfig, WeightedBankDynamicBias),
        ]

    def test_affine_transform_dynamic_bias_forward(self):
        batch_size = 2
        input_dim = 8
        output_dim = 4
        cfg = self.preset(
            config_cls=AffineTransformDynamicBiasConfig,
            input_dim=input_dim,
            output_dim=output_dim,
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
            config_cls=AdditiveDynamicBiasConfig,
            input_dim=input_dim,
            output_dim=output_dim,
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
            config_cls=SigmoidGatedDynamicBiasConfig,
            input_dim=input_dim,
            output_dim=output_dim,
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
            config_cls=TanhGatedDynamicBiasConfig,
            input_dim=input_dim,
            output_dim=output_dim,
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
            config_cls=MultiplicativeDynamicBiasConfig,
            input_dim=input_dim,
            output_dim=output_dim,
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
            config_cls=GeneratorDynamicBiasConfig,
            input_dim=input_dim,
            output_dim=output_dim,
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
            config_cls=WeightedBankDynamicBiasConfig,
            input_dim=input_dim,
            output_dim=output_dim,
            bank_expansion_factor=BankExpansionFactorOptions.FACTOR_OF_THREE,
        )
        model = WeightedBankDynamicBias(cfg)
        logits = torch.randn(batch_size, input_dim)
        bias_params = torch.randn(output_dim)
        output = model(bias_params, logits)

        self.assertEqual(output.shape, (batch_size, output_dim))
        self.assertIsInstance(output, torch.Tensor)

    def test_affine_transform_dynamic_bias_mathematical_correctness(self):
        cfg = self.preset(
            config_cls=AffineTransformDynamicBiasConfig,
            input_dim=3,
            output_dim=2,
        )
        model = AffineTransformDynamicBias(cfg)
        logits = torch.zeros(2, 3)
        bias_params = torch.tensor([1.0, -2.0])
        affine_parameters = torch.tensor([[2.0, 0.5], [-1.5, 1.0]])
        model.model = ConstantGenerator(affine_parameters)

        output = model(bias_params, logits)
        expected = torch.tensor([[2.5, -3.5], [-0.5, 4.0]])

        self.assertTrue(torch.allclose(output, expected, atol=1e-6))

    def test_additive_dynamic_bias_adds_generated_offset(self):
        cfg = self.preset(
            config_cls=AdditiveDynamicBiasConfig,
            input_dim=3,
            output_dim=3,
        )
        model = AdditiveDynamicBias(cfg)
        logits = torch.zeros(2, 3)
        bias_params = torch.tensor([1.0, 2.0, 3.0])
        generated_bias_offset = torch.tensor([[0.5, -1.0, 2.0], [-0.5, 1.5, -2.0]])
        model.model = ConstantGenerator(generated_bias_offset)

        output = model(bias_params, logits)
        expected = torch.tensor([[1.5, 1.0, 5.0], [0.5, 3.5, 1.0]])

        self.assertTrue(torch.allclose(output, expected, atol=1e-6))

    def test_sigmoid_multiplicative_dynamic_bias_applies_sigmoid_gate(self):
        cfg = self.preset(
            config_cls=SigmoidGatedDynamicBiasConfig,
            input_dim=3,
            output_dim=2,
        )
        model = SigmoidGatedDynamicBias(cfg)
        logits = torch.zeros(2, 3)
        bias_params = torch.tensor([2.0, -4.0])
        gate_logits = torch.tensor([[0.0, 0.0], [2.0, -2.0]])
        model.model = ConstantGenerator(gate_logits)

        output = model(bias_params, logits)
        expected = bias_params.unsqueeze(0) * torch.sigmoid(gate_logits)

        self.assertTrue(torch.allclose(output, expected, atol=1e-6))

    def test_tanh_multiplicative_dynamic_bias_applies_tanh_gate(self):
        cfg = self.preset(
            config_cls=TanhGatedDynamicBiasConfig,
            input_dim=3,
            output_dim=2,
        )
        model = TanhGatedDynamicBias(cfg)
        logits = torch.zeros(2, 3)
        bias_params = torch.tensor([2.0, -4.0])
        gate_logits = torch.tensor([[0.0, 0.0], [2.0, -2.0]])
        model.model = ConstantGenerator(gate_logits)

        output = model(bias_params, logits)
        expected = bias_params.unsqueeze(0) * torch.tanh(gate_logits)

        self.assertTrue(torch.allclose(output, expected, atol=1e-6))

    def test_multiplicative_dynamic_bias_applies_raw_scale(self):
        cfg = self.preset(
            config_cls=MultiplicativeDynamicBiasConfig,
            input_dim=3,
            output_dim=2,
        )
        model = MultiplicativeDynamicBias(cfg)
        logits = torch.zeros(2, 3)
        bias_params = torch.tensor([2.0, -4.0])
        bias_scale = torch.tensor([[0.5, -1.0], [2.0, 3.0]])
        model.model = ConstantGenerator(bias_scale)

        output = model(bias_params, logits)
        expected = bias_params.unsqueeze(0) * bias_scale

        self.assertTrue(torch.allclose(output, expected, atol=1e-6))

    def test_bias_dependent_variants_transform_each_batched_bias_independently(self):
        logits = torch.zeros(2, 3)
        bias_params = torch.tensor([[2.0, -4.0], [3.0, 5.0]])
        generated = torch.tensor([[0.5, -1.0], [2.0, 3.0]])
        shared_cases = (
            (
                AdditiveDynamicBias,
                AdditiveDynamicBiasConfig,
                bias_params + generated,
            ),
            (
                MultiplicativeDynamicBias,
                MultiplicativeDynamicBiasConfig,
                bias_params * generated,
            ),
            (
                SigmoidGatedDynamicBias,
                SigmoidGatedDynamicBiasConfig,
                bias_params * torch.sigmoid(generated),
            ),
            (
                TanhGatedDynamicBias,
                TanhGatedDynamicBiasConfig,
                bias_params * torch.tanh(generated),
            ),
        )

        for model_cls, config_cls, expected in shared_cases:
            with self.subTest(model_cls=model_cls.__name__):
                model = model_cls(
                    self.preset(config_cls=config_cls, input_dim=3, output_dim=2)
                )
                model.model = ConstantGenerator(generated)
                torch.testing.assert_close(model(bias_params, logits), expected)

        affine = AffineTransformDynamicBias(
            self.preset(
                config_cls=AffineTransformDynamicBiasConfig,
                input_dim=3,
                output_dim=2,
            )
        )
        affine_parameters = torch.tensor([[2.0, 0.5], [-1.0, 3.0]])
        affine.model = ConstantGenerator(affine_parameters)
        bias_scale, bias_offset = affine_parameters.chunk(2, dim=-1)
        torch.testing.assert_close(
            affine(bias_params, logits),
            bias_scale * bias_params + bias_offset,
        )

    def test_additive_dynamic_bias_applies_bias_decay(self):
        cfg = self.preset(
            config_cls=AdditiveDynamicBiasConfig,
            input_dim=3,
            output_dim=2,
            decay_schedule=WeightDecayScheduleOptions.MULTIPLICATIVE,
            decay_rate=0.5,
            decay_warmup_batches=0,
        )
        model = AdditiveDynamicBias(cfg)
        logits = torch.zeros(2, 3)
        bias_params = torch.tensor([2.0, -4.0])
        generated_bias_offset = torch.zeros(2, 2)
        model.model = ConstantGenerator(generated_bias_offset)

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
            config_cls=GeneratorDynamicBiasConfig,
            input_dim=3,
            output_dim=2,
        )
        model = GeneratorDynamicBias(cfg)
        logits = torch.zeros(2, 3)
        bias_params = torch.tensor([7.0, 9.0])
        generated_bias = torch.tensor([[1.0, -1.0], [0.5, 2.5]])
        model.model = ConstantGenerator(generated_bias)

        output = model(bias_params, logits)

        self.assertTrue(torch.equal(output, generated_bias))

    def test_weighted_bank_dynamic_bias_uses_bank_distribution(self):
        cfg = self.preset(
            config_cls=WeightedBankDynamicBiasConfig,
            input_dim=3,
            output_dim=2,
            bank_expansion_factor=BankExpansionFactorOptions.FACTOR_OF_TWO,
        )
        model = WeightedBankDynamicBias(cfg)
        logits = torch.zeros(2, 3)
        bias_params = torch.tensor([3.0, -5.0])
        bank_logits = torch.tensor([[12.0, -12.0], [-12.0, 12.0]])
        model.model = ConstantGenerator(bank_logits)
        with torch.no_grad():
            model.weight_bank.copy_(torch.tensor([[1.0, 0.0], [0.0, 2.0]]))

        output = model(bias_params, logits)
        expected_distribution = torch.softmax(bank_logits, dim=-1)
        expected = torch.matmul(expected_distribution, model.weight_bank)

        self.assertTrue(torch.allclose(output, expected, atol=1e-6))
        self.assertFalse(torch.allclose(output[0], output[1], atol=1e-6))

    def test_generator_dynamic_bias_ignores_passed_bias_tensor(self):
        cfg = self.preset(
            config_cls=GeneratorDynamicBiasConfig,
            input_dim=3,
            output_dim=2,
        )
        model = GeneratorDynamicBias(cfg)
        logits = torch.zeros(2, 3)
        generated_bias = torch.tensor([[1.0, -1.0], [0.5, 2.5]])
        model.model = ConstantGenerator(generated_bias)

        output_a = model(torch.tensor([7.0, 9.0]), logits)
        output_b = model(torch.tensor([-3.0, 4.0]), logits)

        self.assertTrue(torch.equal(output_a, generated_bias))
        self.assertTrue(torch.equal(output_b, generated_bias))

    def test_weighted_bank_dynamic_bias_ignores_passed_bias_tensor(self):
        cfg = self.preset(
            config_cls=WeightedBankDynamicBiasConfig,
            input_dim=3,
            output_dim=2,
            bank_expansion_factor=BankExpansionFactorOptions.FACTOR_OF_TWO,
        )
        model = WeightedBankDynamicBias(cfg)
        logits = torch.zeros(2, 3)
        bank_logits = torch.tensor([[12.0, -12.0], [-12.0, 12.0]])
        model.model = ConstantGenerator(bank_logits)
        with torch.no_grad():
            model.weight_bank.copy_(torch.tensor([[1.0, 0.0], [0.0, 2.0]]))

        output_a = model(torch.tensor([3.0, -5.0]), logits)
        output_b = model(torch.tensor([-1.0, 8.0]), logits)

        self.assertTrue(torch.allclose(output_a, output_b, atol=1e-6))

    def test_bias_required_strategies_raise_when_bias_params_missing(self):
        logits = torch.randn(2, 8)
        required_config_classes = [
            AffineTransformDynamicBiasConfig,
            AdditiveDynamicBiasConfig,
            MultiplicativeDynamicBiasConfig,
            SigmoidGatedDynamicBiasConfig,
            TanhGatedDynamicBiasConfig,
        ]

        for config_cls in required_config_classes:
            with self.subTest(config_cls=config_cls.__name__):
                cfg = self.preset(config_cls=config_cls)
                model = cfg.build()
                with self.assertRaises(ValueError):
                    model(None, logits)

    def test_build_creates_model_for_each_leaf_config(self):
        input_dim = 8
        output_dim = 4
        for config_cls, model_cls in self.bias_cases():
            with self.subTest(config_cls=config_cls.__name__):
                bank_factor = (
                    BankExpansionFactorOptions.FACTOR_OF_THREE
                    if config_cls is WeightedBankDynamicBiasConfig
                    else None
                )
                cfg = self.preset(
                    config_cls=config_cls,
                    input_dim=input_dim,
                    output_dim=output_dim,
                    bank_expansion_factor=bank_factor,
                )
                model = cfg.build()
                self.assertIsInstance(model, model_cls)

    def test_abstract_config_cannot_build(self):
        cfg = self.preset(config_cls=DynamicBiasConfig)
        with self.assertRaisesRegex(
            ValueError,
            "^DynamicBiasConfig is abstract and has no registered "
            "DynamicBias class; instantiate a concrete leaf config instead\\.$",
        ):
            cfg.build()

    def test_validate_generator_model_raises_on_unknown_generator_type(self):
        class InvalidGeneratorConfig:
            def build(self, overrides):
                return nn.Identity()

        cfg = self.preset(
            config_cls=AdditiveDynamicBiasConfig,
            input_dim=3,
            output_dim=2,
        )
        model = AdditiveDynamicBias(cfg)
        model.model_config = InvalidGeneratorConfig()

        with self.assertRaisesRegex(
            TypeError,
            "^Expected model_config\\.build\\(\\.\\.\\.\\) to return a Layer, "
            "Sequential, or LayerStack, received Identity\\.$",
        ):
            model._init_model(model.output_dim)

    def test_bank_expansion_factor_field_absent_on_non_bank_leaves(self):
        non_bank_leaf_classes = [
            AffineTransformDynamicBiasConfig,
            AdditiveDynamicBiasConfig,
            MultiplicativeDynamicBiasConfig,
            SigmoidGatedDynamicBiasConfig,
            TanhGatedDynamicBiasConfig,
            GeneratorDynamicBiasConfig,
        ]
        for leaf_cls in non_bank_leaf_classes:
            with self.subTest(leaf=leaf_cls.__name__):
                with self.assertRaises(TypeError):
                    leaf_cls(
                        bank_expansion_factor=BankExpansionFactorOptions.FACTOR_OF_TWO
                    )

    def test_weighted_bank_requires_bank_expansion_factor_enum(self):
        invalid_factors = [None, 0, -1]

        for factor in invalid_factors:
            with self.subTest(bank_expansion_factor=factor):
                cfg = self.preset(
                    config_cls=WeightedBankDynamicBiasConfig,
                    bank_expansion_factor=factor,
                )
                with self.assertRaises(ValueError):
                    cfg.build()

    def test_bank_expansion_factor_variants(self):
        batch_size = 2
        input_dim = 8
        output_dim = 4
        logits = torch.randn(batch_size, input_dim)
        bias_params = torch.randn(output_dim)

        for factor in BankExpansionFactorOptions:
            with self.subTest(bank_expansion_factor=factor):
                cfg = self.preset(
                    config_cls=WeightedBankDynamicBiasConfig,
                    input_dim=input_dim,
                    output_dim=output_dim,
                    bank_expansion_factor=factor,
                )
                if factor == BankExpansionFactorOptions.DISABLED:
                    with self.assertRaises(ValueError):
                        cfg.build()
                else:
                    model = cfg.build()
                    output = model(bias_params, logits)
                    self.assertEqual(output.shape, (batch_size, output_dim))

    def test_gradients_flow(self):
        torch.manual_seed(0)
        batch_size = 2
        input_dim = 8
        output_dim = 4

        for config_cls, _ in self.bias_cases():
            with self.subTest(config_cls=config_cls.__name__):
                bank_factor = (
                    BankExpansionFactorOptions.FACTOR_OF_THREE
                    if config_cls is WeightedBankDynamicBiasConfig
                    else None
                )
                cfg = self.preset(
                    config_cls=config_cls,
                    input_dim=input_dim,
                    output_dim=output_dim,
                    bank_expansion_factor=bank_factor,
                )
                cfg.model_config.layer_config.activation = ActivationOptions.DISABLED
                model = cfg.build()
                logits = torch.randn(batch_size, input_dim, requires_grad=True)
                bias_params = torch.randn(output_dim, requires_grad=True)
                output = model(bias_params, logits)
                output.sum().backward()

                grads = [
                    param.grad for param in model.parameters() if param.requires_grad
                ]
                self.assertTrue(grads)
                self.assertTrue(all(grad is not None for grad in grads))
                self.assertTrue(all(torch.isfinite(grad).all() for grad in grads))
                self.assertGreater(
                    sum(torch.count_nonzero(grad).item() for grad in grads),
                    0,
                )
                self.assertIsNotNone(logits.grad)
                self.assertTrue(torch.isfinite(logits.grad).all())
                self.assertGreater(torch.count_nonzero(logits.grad).item(), 0)
                if config_cls in {
                    GeneratorDynamicBiasConfig,
                    WeightedBankDynamicBiasConfig,
                }:
                    self.assertIsNone(bias_params.grad)
                else:
                    self.assertIsNotNone(bias_params.grad)
                    self.assertTrue(torch.isfinite(bias_params.grad).all())
                    self.assertGreater(torch.count_nonzero(bias_params.grad).item(), 0)

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
                    config_cls=AdditiveDynamicBiasConfig,
                    output_dim=output_dim,
                    decay_schedule=schedule,
                    decay_rate=decay_rate,
                )
                model = cfg.build()

                for step in range(num_steps):
                    result = model._maybe_apply_bias_decay(bias_params)
                    expected_factor = expected_factor_fn(step)
                    expected = bias_params * expected_factor
                    self.assertTrue(torch.allclose(result, expected, atol=1e-6))

    def test_exponential_bias_decay_saturates_huge_rate_to_active_dtype(self):
        bias_params = torch.tensor([1.0, -2.0, 3.0])
        cfg = self.preset(
            config_cls=AdditiveDynamicBiasConfig,
            output_dim=bias_params.numel(),
            decay_schedule=WeightDecayScheduleOptions.EXPONENTIAL,
            decay_rate=1.0e100,
        )
        model = cfg.build()

        initial = model._maybe_apply_bias_decay(bias_params)
        decayed = model._maybe_apply_bias_decay(bias_params)

        torch.testing.assert_close(initial, bias_params)
        torch.testing.assert_close(decayed, torch.zeros_like(bias_params))
        self.assertEqual(initial.dtype, bias_params.dtype)
        self.assertEqual(decayed.dtype, bias_params.dtype)
        self.assertTrue(torch.isfinite(initial).all())
        self.assertTrue(torch.isfinite(decayed).all())

    def test_bias_decay_schedule_disabled_leaves_bias_unchanged(self):
        output_dim = 4
        bias_params = torch.randn(output_dim)
        cfg = self.preset(
            config_cls=AdditiveDynamicBiasConfig,
            output_dim=output_dim,
            decay_schedule=WeightDecayScheduleOptions.DISABLED,
            decay_rate=0.3,
        )
        model = cfg.build()
        baseline_decay_step = model.decay_step.clone()
        baseline_warmup_step = model.warmup_step.clone()

        for _ in range(3):
            result = model._maybe_apply_bias_decay(bias_params)
            self.assertTrue(torch.equal(result, bias_params))

        self.assertTrue(torch.equal(model.decay_step, baseline_decay_step))
        self.assertTrue(torch.equal(model.warmup_step, baseline_warmup_step))

    def test_invalid_decay_parameters_raise(self):
        invalid_cases = [
            ("missing_rate", WeightDecayScheduleOptions.EXPONENTIAL, None, 0),
            ("zero_rate", WeightDecayScheduleOptions.EXPONENTIAL, 0.0, 0),
            ("negative_rate", WeightDecayScheduleOptions.EXPONENTIAL, -0.1, 0),
            ("linear_rate_too_large", WeightDecayScheduleOptions.LINEAR, 1.0, 0),
            (
                "multiplicative_rate_too_large",
                WeightDecayScheduleOptions.MULTIPLICATIVE,
                1.0,
                0,
            ),
            ("negative_warmup", WeightDecayScheduleOptions.EXPONENTIAL, 0.1, -1),
        ]

        for name, schedule, rate, warmup_batches in invalid_cases:
            with self.subTest(case=name):
                cfg = self.preset(
                    config_cls=AdditiveDynamicBiasConfig,
                    decay_schedule=schedule,
                    decay_rate=rate,
                    decay_warmup_batches=warmup_batches,
                )
                with self.assertRaises(ValueError):
                    cfg.build()

    def test_bias_decay_warmup_delays_decay(self):
        output_dim = 4
        warmup_batches = 3
        bias_params = torch.randn(output_dim)
        cfg = self.preset(
            config_cls=AdditiveDynamicBiasConfig,
            output_dim=output_dim,
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

    def test_bias_decay_counters_frozen_in_eval_mode(self):
        output_dim = 4
        warmup_batches = 2
        bias_params = torch.randn(output_dim)
        cfg = self.preset(
            config_cls=AdditiveDynamicBiasConfig,
            output_dim=output_dim,
            decay_schedule=WeightDecayScheduleOptions.EXPONENTIAL,
            decay_rate=0.1,
            decay_warmup_batches=warmup_batches,
        )
        model = cfg.build()

        model.eval()
        for _ in range(3):
            model._maybe_apply_bias_decay(bias_params)
        self.assertEqual(model.warmup_step.item(), 0)
        self.assertEqual(model.decay_step.item(), 0)

        model.train()
        for _ in range(warmup_batches):
            model._maybe_apply_bias_decay(bias_params)
        model._maybe_apply_bias_decay(bias_params)
        frozen_decay_step = model.decay_step.clone()
        frozen_warmup_step = model.warmup_step.clone()

        model.eval()
        baseline = model._maybe_apply_bias_decay(bias_params)
        for _ in range(3):
            result = model._maybe_apply_bias_decay(bias_params)
            self.assertTrue(torch.equal(result, baseline))
        self.assertTrue(torch.equal(model.decay_step, frozen_decay_step))
        self.assertTrue(torch.equal(model.warmup_step, frozen_warmup_step))

    def test_bias_decay_schedule_raises_on_unknown_schedule(self):
        cfg = self.preset(
            config_cls=AdditiveDynamicBiasConfig,
            decay_schedule=WeightDecayScheduleOptions.EXPONENTIAL,
            decay_rate=0.1,
        )
        model = cfg.build()
        model.decay_schedule_option = "invalid_schedule"
        bias_params = torch.randn(4)

        with self.assertRaises(ValueError):
            model._maybe_apply_bias_decay(bias_params)


if __name__ == "__main__":
    unittest.main()

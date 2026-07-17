from __future__ import annotations

import unittest

import torch

from emperor.augmentations.adaptive_parameters import (
    AdditiveDynamicBiasConfig,
    AffineTransformDynamicBiasConfig,
    AntiDynamicDiagonalConfig,
    BankExpansionFactorOptions,
    CombinedDynamicDiagonalConfig,
    GeneratorDynamicBiasConfig,
    MultiplicativeDynamicBiasConfig,
    SigmoidGatedDynamicBiasConfig,
    StandardDynamicDiagonalConfig,
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
from emperor.linears import LinearLayer, LinearLayerConfig


def linear_stack_config(
    input_dim: int,
    output_dim: int,
) -> LayerStackConfig:
    return LayerStackConfig(
        input_dim=input_dim,
        hidden_dim=max(input_dim, output_dim),
        output_dim=output_dim,
        num_layers=1,
        apply_output_pipeline_flag=False,
        last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
        shared_gate_config=None,
        shared_halting_config=None,
        shared_memory_config=None,
        layer_config=LayerConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            activation=ActivationOptions.DISABLED,
            residual_connection_option=ResidualConnectionOptions.DISABLED,
            dropout_probability=0.0,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
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


def common_bias_arguments(
    input_dim: int,
    output_dim: int,
) -> dict[str, object]:
    return {
        "input_dim": input_dim,
        "output_dim": output_dim,
        "decay_schedule": WeightDecayScheduleOptions.DISABLED,
        "decay_rate": 0.0,
        "decay_warmup_batches": 0,
        "model_config": linear_stack_config(7, 11),
    }


def assign_generator(
    stack: LayerStack,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> None:
    linear = stack[0].model
    if not isinstance(linear, LinearLayer):
        raise TypeError(f"Expected LinearLayer, received {type(linear).__name__}.")
    with torch.no_grad():
        linear.weight_params.copy_(weight)
        if linear.bias_params is None:
            raise TypeError("Expected generator bias parameters.")
        linear.bias_params.copy_(bias)


def generator_dimensions(stack: LayerStack) -> tuple[int, int]:
    linear = stack[0].model
    if not isinstance(linear, LinearLayer):
        raise TypeError(f"Expected LinearLayer, received {type(linear).__name__}.")
    return linear.input_dim, linear.output_dim


class AdaptiveParameterBiasDiagonalMutationContractTests(unittest.TestCase):
    def test_each_bias_leaf_applies_overrides_to_itself_and_real_generator(
        self,
    ) -> None:
        cases = (
            (
                AdditiveDynamicBias,
                AdditiveDynamicBiasConfig,
                5,
            ),
            (
                MultiplicativeDynamicBias,
                MultiplicativeDynamicBiasConfig,
                5,
            ),
            (
                SigmoidGatedDynamicBias,
                SigmoidGatedDynamicBiasConfig,
                5,
            ),
            (
                TanhGatedDynamicBias,
                TanhGatedDynamicBiasConfig,
                5,
            ),
            (
                GeneratorDynamicBias,
                GeneratorDynamicBiasConfig,
                5,
            ),
            (
                AffineTransformDynamicBias,
                AffineTransformDynamicBiasConfig,
                2,
            ),
        )

        for model_type, config_type, expected_generator_output in cases:
            with self.subTest(model_type=model_type.__name__):
                config = config_type(**common_bias_arguments(2, 3))
                overrides = config_type(input_dim=4, output_dim=5)
                model = model_type(config, overrides)
                self.assertEqual((model.input_dim, model.output_dim), (4, 5))
                self.assertEqual(
                    generator_dimensions(model.model),
                    (4, expected_generator_output),
                )

        weighted = WeightedBankDynamicBias(
            WeightedBankDynamicBiasConfig(
                **common_bias_arguments(2, 3),
                bank_expansion_factor=BankExpansionFactorOptions.FACTOR_OF_ONE,
            ),
            WeightedBankDynamicBiasConfig(
                input_dim=4,
                output_dim=5,
                bank_expansion_factor=BankExpansionFactorOptions.FACTOR_OF_TWO,
            ),
        )
        self.assertEqual((weighted.input_dim, weighted.output_dim), (4, 5))
        self.assertEqual(weighted.bank_expansion_factor, 2)
        self.assertEqual(tuple(weighted.weight_bank.shape), (2, 5))
        self.assertEqual(generator_dimensions(weighted.model), (4, 2))

    def test_every_bias_strategy_has_exact_real_generator_math(self) -> None:
        logits = torch.tensor([[1.0, 2.0], [-1.0, 0.5]])
        bias_params = torch.tensor([1.0, -2.0, 3.0])
        projection = torch.tensor(
            [
                [1.0, -2.0, 0.5],
                [0.25, 3.0, -1.0],
            ]
        )
        projection_bias = torch.tensor([0.5, -0.25, 2.0])
        generated = logits @ projection + projection_bias
        shared_cases = (
            (
                AdditiveDynamicBias(
                    AdditiveDynamicBiasConfig(**common_bias_arguments(2, 3))
                ),
                bias_params + generated,
            ),
            (
                MultiplicativeDynamicBias(
                    MultiplicativeDynamicBiasConfig(
                        **common_bias_arguments(2, 3)
                    )
                ),
                bias_params * generated,
            ),
            (
                SigmoidGatedDynamicBias(
                    SigmoidGatedDynamicBiasConfig(
                        **common_bias_arguments(2, 3)
                    )
                ),
                bias_params * torch.sigmoid(generated),
            ),
            (
                TanhGatedDynamicBias(
                    TanhGatedDynamicBiasConfig(**common_bias_arguments(2, 3))
                ),
                bias_params * torch.tanh(generated),
            ),
            (
                GeneratorDynamicBias(
                    GeneratorDynamicBiasConfig(**common_bias_arguments(2, 3))
                ),
                generated,
            ),
        )
        for model, expected in shared_cases:
            with self.subTest(model_type=type(model).__name__):
                assign_generator(model.model, projection, projection_bias)
                self.assertTrue(
                    torch.equal(model(bias_params, logits), expected)
                )

        affine = AffineTransformDynamicBias(
            AffineTransformDynamicBiasConfig(**common_bias_arguments(2, 3))
        )
        affine_projection = torch.tensor([[1.0, -0.5], [2.0, 0.25]])
        affine_bias = torch.tensor([0.5, -1.0])
        assign_generator(affine.model, affine_projection, affine_bias)
        affine_parameters = logits @ affine_projection + affine_bias
        expected_affine = (
            affine_parameters[:, :1] * bias_params
            + affine_parameters[:, 1:]
        )
        self.assertTrue(
            torch.equal(affine(bias_params, logits), expected_affine)
        )

        weighted = WeightedBankDynamicBias(
            WeightedBankDynamicBiasConfig(
                **common_bias_arguments(2, 3),
                bank_expansion_factor=BankExpansionFactorOptions.FACTOR_OF_TWO,
            )
        )
        bank_projection = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
        bank_bias = torch.tensor([0.0, -1.0])
        bank = torch.tensor([[1.0, -2.0, 3.0], [4.0, 0.5, -1.0]])
        assign_generator(weighted.model, bank_projection, bank_bias)
        with torch.no_grad():
            weighted.weight_bank.copy_(bank)
        bank_logits = logits @ bank_projection + bank_bias
        expected_weighted = torch.softmax(bank_logits, dim=-1) @ bank
        actual_weighted = weighted(bias_params, logits)
        self.assertTrue(torch.equal(actual_weighted, expected_weighted))
        wrong_batch_softmax = torch.softmax(bank_logits, dim=0) @ bank
        self.assertFalse(torch.allclose(actual_weighted, wrong_batch_softmax))

    def test_invalid_runtime_decay_schedule_fails_through_public_forward(self) -> None:
        model = AdditiveDynamicBias(
            AdditiveDynamicBiasConfig(**common_bias_arguments(2, 3))
        )
        model.decay_schedule_option = "invalid"

        with self.assertRaisesRegex(
            ValueError,
            r"^Unsupported decay_schedule value: 'invalid'\.$",
        ):
            model(torch.ones(3), torch.ones(1, 2))

    def test_diagonal_leaves_apply_overrides_and_generator_dimensions(self) -> None:
        cases = (
            (StandardDynamicDiagonal, StandardDynamicDiagonalConfig),
            (AntiDynamicDiagonal, AntiDynamicDiagonalConfig),
            (CombinedDynamicDiagonal, CombinedDynamicDiagonalConfig),
        )
        for model_type, config_type in cases:
            with self.subTest(model_type=model_type.__name__):
                model = model_type(
                    config_type(
                        input_dim=2,
                        output_dim=3,
                        model_config=linear_stack_config(7, 11),
                    ),
                    config_type(input_dim=4, output_dim=3),
                )
                self.assertEqual((model.input_dim, model.output_dim), (4, 3))
                self.assertEqual(model.padding_shape, (0, 0, 0, 1))
                generator = (
                    model.diagonal_model.model
                    if isinstance(model, CombinedDynamicDiagonal)
                    else model.model
                )
                self.assertEqual(generator_dimensions(generator), (4, 3))

    def test_standard_and_anti_diagonals_have_exact_rectangular_geometry(
        self,
    ) -> None:
        standard = StandardDynamicDiagonal(
            StandardDynamicDiagonalConfig(
                input_dim=2,
                output_dim=3,
                model_config=linear_stack_config(5, 7),
            )
        )
        assign_generator(standard.model, torch.eye(2), torch.zeros(2))
        standard_logits = torch.tensor([[1.0, 2.0], [-1.0, 0.5]])
        standard_base = torch.tensor([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
        expected_standard = standard_base + torch.tensor(
            [
                [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
                [[-1.0, 0.0, 0.0], [0.0, 0.5, 0.0]],
            ]
        )
        self.assertTrue(
            torch.equal(
                standard(standard_base, standard_logits),
                expected_standard,
            )
        )

        anti = AntiDynamicDiagonal(
            AntiDynamicDiagonalConfig(
                input_dim=3,
                output_dim=2,
                model_config=linear_stack_config(5, 7),
            )
        )
        assign_generator(
            anti.model,
            torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]),
            torch.zeros(2),
        )
        anti_logits = torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.5, 4.0]])
        anti_base = torch.tensor([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])
        expected_anti = anti_base + torch.tensor(
            [
                [[0.0, 1.0], [2.0, 0.0], [0.0, 0.0]],
                [[0.0, -1.0], [0.5, 0.0], [0.0, 0.0]],
            ]
        )
        self.assertTrue(torch.equal(anti(anti_base, anti_logits), expected_anti))

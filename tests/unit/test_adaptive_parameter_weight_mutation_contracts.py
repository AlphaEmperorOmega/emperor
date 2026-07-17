from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from emperor.augmentations.adaptive_parameters import (
    BankExpansionFactorOptions,
    DualModelDynamicWeightConfig,
    DynamicDepthOptions,
    HypernetworkDynamicWeightConfig,
    LayeredWeightedBankDynamicWeightConfig,
    LowRankDynamicWeightConfig,
    SingleModelDynamicWeightConfig,
    SoftWeightedBankDynamicWeightConfig,
    WeightDecayScheduleOptions,
    WeightNormalizationOptions,
    WeightNormalizationPositionOptions,
)
from emperor.augmentations.adaptive_parameters._weights.depth_mapping import (
    DepthMappingLayer,
    DepthMappingLayerStack,
)
from emperor.augmentations.adaptive_parameters._weights.variants.dual_model import (
    DualModelDynamicWeight,
)
from emperor.augmentations.adaptive_parameters._weights.variants.hypernetwork import (
    HypernetworkDynamicWeight,
)
from emperor.augmentations.adaptive_parameters._weights.variants.layered_weighted_bank import (
    LayeredWeightedBankDynamicWeight,
)
from emperor.augmentations.adaptive_parameters._weights.variants.low_rank import (
    LowRankDynamicWeight,
)
from emperor.augmentations.adaptive_parameters._weights.variants.single_model import (
    SingleModelDynamicWeight,
)
from emperor.augmentations.adaptive_parameters._weights.variants.soft_weighted_bank import (
    SoftWeightedBankDynamicWeight,
)
from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerConfig,
    LayerNormPositionOptions,
    LayerStackConfig,
    ResidualConnectionOptions,
)
from emperor.linears import LinearLayerConfig


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


def common_weight_arguments(
    input_dim: int,
    output_dim: int,
) -> dict[str, object]:
    return {
        "input_dim": input_dim,
        "output_dim": output_dim,
        "generator_depth": DynamicDepthOptions.DEPTH_OF_ONE,
        "decay_schedule": WeightDecayScheduleOptions.DISABLED,
        "decay_rate": 0.0,
        "decay_warmup_batches": 0,
        "model_config": linear_stack_config(7, 11),
    }


def single_config(
    input_dim: int = 2,
) -> SingleModelDynamicWeightConfig:
    return SingleModelDynamicWeightConfig(
        **common_weight_arguments(input_dim, input_dim),
        normalization_option=WeightNormalizationOptions.DISABLED,
        normalization_position_option=WeightNormalizationPositionOptions.DISABLED,
    )


def dual_config(
    input_dim: int = 2,
    output_dim: int = 3,
) -> DualModelDynamicWeightConfig:
    return DualModelDynamicWeightConfig(
        **common_weight_arguments(input_dim, output_dim),
        normalization_option=WeightNormalizationOptions.DISABLED,
        normalization_position_option=WeightNormalizationPositionOptions.DISABLED,
    )


def low_rank_config(
    input_dim: int = 2,
    output_dim: int = 3,
) -> LowRankDynamicWeightConfig:
    return LowRankDynamicWeightConfig(
        **common_weight_arguments(input_dim, output_dim),
        normalization_option=WeightNormalizationOptions.DISABLED,
    )


def hypernetwork_config(
    input_dim: int = 2,
    output_dim: int = 3,
) -> HypernetworkDynamicWeightConfig:
    return HypernetworkDynamicWeightConfig(
        **common_weight_arguments(input_dim, output_dim),
        normalization_option=WeightNormalizationOptions.DISABLED,
    )


def layered_bank_config(
    input_dim: int = 2,
    output_dim: int = 3,
    factor: BankExpansionFactorOptions = BankExpansionFactorOptions.FACTOR_OF_ONE,
) -> LayeredWeightedBankDynamicWeightConfig:
    return LayeredWeightedBankDynamicWeightConfig(
        **common_weight_arguments(input_dim, output_dim),
        bank_expansion_factor=factor,
    )


def soft_bank_config(
    input_dim: int = 2,
    output_dim: int = 3,
    factor: BankExpansionFactorOptions = BankExpansionFactorOptions.FACTOR_OF_ONE,
) -> SoftWeightedBankDynamicWeightConfig:
    return SoftWeightedBankDynamicWeightConfig(
        **common_weight_arguments(input_dim, output_dim),
        bank_expansion_factor=factor,
    )


def assign_depth_mapping(
    stack: DepthMappingLayerStack,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> None:
    depth_layer = stack.model[0].model
    if not isinstance(depth_layer, DepthMappingLayer):
        raise TypeError(
            f"Expected DepthMappingLayer, received {type(depth_layer).__name__}."
        )
    with torch.no_grad():
        depth_layer.weight_params.copy_(weight.unsqueeze(0))
        depth_layer.bias_params.copy_(bias.unsqueeze(0))


class AdaptiveParameterWeightMutationContractTests(unittest.TestCase):
    def test_each_leaf_constructor_applies_explicit_overrides(self) -> None:
        cases = (
            (
                SingleModelDynamicWeight,
                single_config(2),
                SingleModelDynamicWeightConfig(input_dim=3, output_dim=3),
                (3, 3, None),
            ),
            (
                DualModelDynamicWeight,
                dual_config(2, 3),
                DualModelDynamicWeightConfig(input_dim=3, output_dim=4),
                (3, 4, None),
            ),
            (
                LowRankDynamicWeight,
                low_rank_config(2, 3),
                LowRankDynamicWeightConfig(input_dim=3, output_dim=4),
                (3, 4, None),
            ),
            (
                HypernetworkDynamicWeight,
                hypernetwork_config(2, 3),
                HypernetworkDynamicWeightConfig(input_dim=3, output_dim=4),
                (3, 4, None),
            ),
            (
                LayeredWeightedBankDynamicWeight,
                layered_bank_config(2, 3),
                LayeredWeightedBankDynamicWeightConfig(
                    input_dim=3,
                    output_dim=4,
                    bank_expansion_factor=BankExpansionFactorOptions.FACTOR_OF_TWO,
                ),
                (3, 4, 2),
            ),
            (
                SoftWeightedBankDynamicWeight,
                soft_bank_config(2, 3),
                SoftWeightedBankDynamicWeightConfig(
                    input_dim=3,
                    output_dim=4,
                    bank_expansion_factor=BankExpansionFactorOptions.FACTOR_OF_TWO,
                ),
                (3, 4, 2),
            ),
        )

        for model_type, config, overrides, expected in cases:
            with self.subTest(model_type=model_type.__name__):
                model = model_type(config, overrides)
                actual_factor = getattr(model, "bank_expansion_factor", None)
                self.assertEqual(
                    (model.input_dim, model.output_dim, actual_factor),
                    expected,
                )

    def test_single_dual_low_rank_and_hypernetwork_add_exact_updates(self) -> None:
        inputs = torch.tensor([[1.0, 2.0], [-1.0, 0.5]])
        base = torch.tensor(
            [
                [10.0, 20.0, 30.0],
                [40.0, 50.0, 60.0],
            ]
        )
        output_projection = torch.tensor(
            [
                [1.0, -1.0, 0.5],
                [0.25, 2.0, -2.0],
            ]
        )
        identity = torch.eye(2)

        single = SingleModelDynamicWeight(single_config())
        assign_depth_mapping(single.model, identity, torch.zeros(2))
        single_base = base[:, :2]
        single_expected = (
            single_base.unsqueeze(0)
            + torch.einsum("bi,bj->bij", inputs, inputs)
        )
        self.assertTrue(
            torch.equal(single(single_base, inputs), single_expected)
        )

        dual = DualModelDynamicWeight(dual_config())
        assign_depth_mapping(dual.input_model, identity, torch.zeros(2))
        assign_depth_mapping(
            dual.output_model,
            output_projection,
            torch.zeros(3),
        )
        projected = inputs @ output_projection
        expected = base.unsqueeze(0) + torch.einsum(
            "bi,bj->bij",
            inputs,
            projected,
        )
        self.assertTrue(torch.equal(dual(base, inputs), expected))

        low_rank = LowRankDynamicWeight(low_rank_config())
        assign_depth_mapping(low_rank.input_model, identity, torch.zeros(2))
        assign_depth_mapping(
            low_rank.output_model,
            output_projection,
            torch.zeros(3),
        )
        self.assertTrue(torch.equal(low_rank(base, inputs), expected))

        hypernetwork = HypernetworkDynamicWeight(hypernetwork_config())
        flat_projection = torch.tensor(
            [
                [1.0, -1.0, 0.5, 0.25, 2.0, -2.0],
                [-0.5, 1.5, 2.0, 1.0, -1.0, 0.75],
            ]
        )
        assign_depth_mapping(
            hypernetwork.model,
            flat_projection,
            torch.zeros(6),
        )
        hyper_update = (inputs @ flat_projection).reshape(2, 2, 3)
        self.assertTrue(
            torch.equal(
                hypernetwork(base, inputs),
                base.unsqueeze(0) + hyper_update,
            )
        )

    def test_normalization_defaults_and_every_numeric_transform_are_exact(
        self,
    ) -> None:
        model = SingleModelDynamicWeight(single_config()).double()
        self.assertEqual(model.scale.item(), 1.0)
        self.assertEqual(model.clamp_limit.item(), 1.0)
        with torch.no_grad():
            model.scale.fill_(1.5)
            model.clamp_limit.fill_(2.0)
        vectors = torch.tensor(
            [
                [[1.0e-8, 2.0e-8, -3.0e-8], [3.0, -4.0, 1.0]],
                [[-2.0, 0.5, 4.0], [1.0e-9, -4.0e-9, 2.0e-9]],
            ],
            dtype=torch.float64,
        )
        cases = (
            (
                WeightNormalizationOptions.CLAMP,
                vectors.clamp(-2.0, 2.0),
            ),
            (
                WeightNormalizationOptions.L2_SCALE,
                F.normalize(vectors, dim=-1) * 1.5,
            ),
            (
                WeightNormalizationOptions.SOFT_CLAMP,
                2.0 * torch.tanh(vectors / 2.0),
            ),
            (
                WeightNormalizationOptions.RMS,
                vectors
                / (vectors.pow(2).mean(dim=-1, keepdim=True).sqrt() + 1.0e-8)
                * 1.5,
            ),
            (
                WeightNormalizationOptions.SIGMOID_SCALE,
                (torch.sigmoid(vectors) * 2.0 - 1.0) * 1.5,
            ),
            (
                WeightNormalizationOptions.DISABLED,
                vectors,
            ),
        )

        for option, expected in cases:
            with self.subTest(option=option):
                model.normalization_option = option
                actual = model._apply_normalization_transform(vectors)
                self.assertTrue(
                    torch.allclose(actual, expected, atol=1e-15, rtol=1e-12)
                )

    def test_unsupported_normalization_and_decay_errors_are_exact(self) -> None:
        model = SingleModelDynamicWeight(single_config())
        vectors = torch.ones(1, 1, 2)

        model.normalization_position_option = "invalid"
        with self.assertRaises(ValueError) as position_error:
            model._compute_outer_product(vectors, vectors)
        self.assertEqual(
            str(position_error.exception),
            "Unsupported normalization_position_option value: 'invalid'.",
        )

        model.normalization_option = "invalid"
        with self.assertRaises(ValueError) as normalization_error:
            model._apply_normalization_transform(vectors)
        self.assertEqual(
            str(normalization_error.exception),
            "Unsupported normalization_option value: 'invalid'.",
        )

        with self.assertRaises(ValueError) as decay_error:
            model._DynamicWeightAbstract__compute_decay_factor_by_schedule("invalid")
        self.assertEqual(
            str(decay_error.exception),
            "Unsupported decay_schedule value: 'invalid'.",
        )


if __name__ == "__main__":
    unittest.main()

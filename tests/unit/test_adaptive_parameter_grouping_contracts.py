from __future__ import annotations

import unittest
from inspect import signature
from unittest.mock import patch

import torch

from emperor.augmentations.adaptive_parameters import (
    AdaptiveLinearLayerConfig,
    AdaptiveParameterAugmentationConfig,
    AdaptiveParameterGroupingScopeOptions,
    AdditiveDynamicBiasConfig,
    AffineTransformDynamicBiasConfig,
    AntiDynamicDiagonalConfig,
    BankExpansionFactorOptions,
    CombinedDynamicDiagonalConfig,
    DiagonalAxisMaskConfig,
    DualModelDynamicWeightConfig,
    DynamicDepthOptions,
    GeneratorDynamicBiasConfig,
    HypernetworkDynamicWeightConfig,
    LayeredWeightedBankDynamicWeightConfig,
    LowRankDynamicWeightConfig,
    MaskDimensionOptions,
    MultiplicativeDynamicBiasConfig,
    OuterProductMaskConfig,
    PerAxisScoreMaskConfig,
    SigmoidGatedDynamicBiasConfig,
    SingleModelDynamicWeightConfig,
    SoftWeightedBankDynamicWeightConfig,
    StandardDynamicDiagonalConfig,
    TanhGatedDynamicBiasConfig,
    TopSliceAxisMaskConfig,
    WeightDecayScheduleOptions,
    WeightedBankDynamicBiasConfig,
    WeightInformedScoreAxisMaskConfig,
    WeightNormalizationOptions,
    WeightNormalizationPositionOptions,
)
from emperor.augmentations.adaptive_parameters._augmentation import (
    AdaptiveParameterAugmentation,
)
from emperor.augmentations.adaptive_parameters._grouping import (
    build_adaptive_group_plan,
)
from emperor.augmentations.adaptive_parameters._linear_adapter import (
    AdaptiveLinearLayer,
)
from emperor.augmentations.adaptive_parameters._validation import (
    AdaptiveLinearValidator,
    AdaptiveParameterAugmentationValidator,
)
from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerConfig,
    LayerNormPositionOptions,
    LayerStackConfig,
    RowLayout,
)
from emperor.linears import LinearLayerConfig


def grouping_fields(
    scope: AdaptiveParameterGroupingScopeOptions,
    group_count: int | None,
) -> dict[str, object]:
    return {"grouping_scope": scope, "group_count": group_count}


def linear_stack_config(input_dim: int, output_dim: int) -> LayerStackConfig:
    return LayerStackConfig(
        input_dim=input_dim,
        hidden_dim=max(input_dim, output_dim),
        output_dim=output_dim,
        num_layers=1,
        last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
        apply_output_pipeline_flag=False,
        layer_config=LayerConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            activation=ActivationOptions.DISABLED,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            residual_config=None,
            dropout_probability=0.0,
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


def grouped_adaptive_bias_linear(
    scope: AdaptiveParameterGroupingScopeOptions,
    group_count: int,
) -> AdaptiveLinearLayer:
    input_dim = 2
    output_dim = 2
    model = AdaptiveLinearLayer(
        AdaptiveLinearLayerConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            bias_flag=True,
            adaptive_augmentation_config=AdaptiveParameterAugmentationConfig(
                bias_config=AdditiveDynamicBiasConfig(
                    decay_schedule=WeightDecayScheduleOptions.DISABLED,
                    decay_rate=0.0,
                    decay_warmup_batches=0,
                    model_config=linear_stack_config(input_dim, output_dim),
                ),
                **grouping_fields(scope, group_count),
            ),
        )
    )
    with torch.no_grad():
        model.weight_params.zero_()
        model.bias_params.zero_()
        generator = model.adaptive_behaviour.bias_model.model[0].model
        generator.weight_params.copy_(torch.eye(2))
        generator.bias_params.zero_()
    return model


class DenseWeightShapeSpy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conditioning_shape = None
        self.generated_weight_shape = None

    def forward(self, weight_params, conditioning_input):
        del weight_params
        self.conditioning_shape = tuple(conditioning_input.shape)
        generated_weights = torch.diag_embed(conditioning_input)
        self.generated_weight_shape = tuple(generated_weights.shape)
        return generated_weights


class DeterministicDynamicWeightGenerator(torch.nn.Module):
    def __init__(self, dtype: torch.dtype) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(
            torch.tensor([[0.5, -1.0], [1.5, 2.0]], dtype=dtype)
        )
        self.offset = torch.nn.Parameter(
            torch.tensor([[0.25, 0.5], [-0.5, 1.0]], dtype=dtype)
        )
        self.generated_weights = None
        self.conditioning_input = None

    def forward(self, weight_params, conditioning_input):
        del weight_params
        self.conditioning_input = conditioning_input.detach().clone()
        generated_weights = conditioning_input.unsqueeze(-1) * self.scale.unsqueeze(
            0
        ) + self.offset.unsqueeze(0)
        generated_weights.retain_grad()
        self.generated_weights = generated_weights
        return generated_weights


class AdaptiveParameterGroupingPrimitiveTests(unittest.TestCase):
    def test_rows_grouping_preserves_members_and_restores_original_order(self):
        inputs = torch.tensor(
            [
                [1.0, 10.0],
                [2.0, 20.0],
                [3.0, 30.0],
                [4.0, 40.0],
            ]
        )
        plan = build_adaptive_group_plan(
            inputs,
            AdaptiveParameterGroupingScopeOptions.ROWS,
            2,
            RowLayout.rows(4, context_sharing_restricted=False),
        )

        self.assertEqual(tuple(plan.grouped_members.shape), (2, 2, 2))
        torch.testing.assert_close(
            plan.grouped_members,
            torch.tensor(
                [
                    [[1.0, 10.0], [2.0, 20.0]],
                    [[3.0, 30.0], [4.0, 40.0]],
                ]
            ),
        )
        self.assertIsNone(plan.valid_members)

        grouped_outputs = torch.tensor(
            [
                [[100.0], [101.0]],
                [[200.0], [201.0]],
            ]
        )
        torch.testing.assert_close(
            plan.restore(grouped_outputs),
            torch.tensor([[100.0], [101.0], [200.0], [201.0]]),
        )

    def test_batch_and_sequence_major_layouts_have_identical_logical_groups(self):
        logical_inputs = torch.tensor(
            [
                [[1.0], [2.0], [3.0], [4.0]],
                [[10.0], [20.0], [30.0], [40.0]],
            ]
        )
        batch_major_plan = build_adaptive_group_plan(
            logical_inputs.reshape(-1, 1),
            AdaptiveParameterGroupingScopeOptions.SEQUENCE,
            2,
            RowLayout.sequence(
                leading_shape=(2, 4),
                batch_axis=0,
                sequence_axis=1,
                context_sharing_restricted=False,
            ),
        )
        sequence_major_inputs = logical_inputs.transpose(0, 1).reshape(-1, 1)
        sequence_major_plan = build_adaptive_group_plan(
            sequence_major_inputs,
            AdaptiveParameterGroupingScopeOptions.SEQUENCE,
            2,
            RowLayout.sequence(
                leading_shape=(4, 2),
                batch_axis=1,
                sequence_axis=0,
                context_sharing_restricted=False,
            ),
        )

        expected_grouped_members = torch.tensor(
            [
                [[1.0], [2.0]],
                [[3.0], [4.0]],
                [[10.0], [20.0]],
                [[30.0], [40.0]],
            ]
        )
        torch.testing.assert_close(
            batch_major_plan.grouped_members,
            expected_grouped_members,
        )
        torch.testing.assert_close(
            sequence_major_plan.grouped_members,
            expected_grouped_members,
        )

        logical_grouped_outputs = torch.tensor(
            [
                [[101.0], [102.0]],
                [[103.0], [104.0]],
                [[201.0], [202.0]],
                [[203.0], [204.0]],
            ]
        )
        expected_logical_output = torch.tensor(
            [
                [[101.0], [102.0], [103.0], [104.0]],
                [[201.0], [202.0], [203.0], [204.0]],
            ]
        )
        torch.testing.assert_close(
            batch_major_plan.restore(logical_grouped_outputs),
            expected_logical_output.reshape(-1, 1),
        )
        torch.testing.assert_close(
            sequence_major_plan.restore(logical_grouped_outputs),
            expected_logical_output.transpose(0, 1).reshape(-1, 1),
        )

    def test_non_contiguous_rows_are_grouped_without_changing_logical_order(self):
        inputs = torch.arange(8.0).reshape(2, 4).transpose(0, 1)
        self.assertFalse(inputs.is_contiguous())

        plan = build_adaptive_group_plan(
            inputs,
            AdaptiveParameterGroupingScopeOptions.ROWS,
            2,
            RowLayout.rows(4, context_sharing_restricted=False),
        )

        torch.testing.assert_close(
            plan.grouped_members,
            torch.tensor(
                [
                    [[0.0, 4.0], [1.0, 5.0]],
                    [[2.0, 6.0], [3.0, 7.0]],
                ]
            ),
        )

    def test_padding_mask_aligns_without_compacting_absolute_positions(self):
        inputs = torch.tensor(
            [
                [[1.0], [2.0], [1000.0], [2000.0]],
                [[10.0], [100.0], [1000.0], [10000.0]],
            ]
        )
        valid_rows = torch.tensor([True, True, False, False, True, False, False, False])

        plan = build_adaptive_group_plan(
            inputs.reshape(-1, 1),
            AdaptiveParameterGroupingScopeOptions.SEQUENCE,
            2,
            RowLayout.sequence(
                leading_shape=(2, 4),
                batch_axis=0,
                sequence_axis=1,
                valid_rows=valid_rows,
                context_sharing_restricted=False,
            ),
        )

        torch.testing.assert_close(
            plan.grouped_members,
            torch.tensor(
                [
                    [[1.0], [2.0]],
                    [[1000.0], [2000.0]],
                    [[10.0], [100.0]],
                    [[1000.0], [10000.0]],
                ]
            ),
        )
        torch.testing.assert_close(
            plan.valid_members,
            torch.tensor(
                [
                    [True, True],
                    [False, False],
                    [True, False],
                    [False, False],
                ]
            ),
        )

    def test_masking_excludes_non_finite_padding_without_zero_times_infinity(self):
        model = grouped_adaptive_bias_linear(
            AdaptiveParameterGroupingScopeOptions.SEQUENCE,
            2,
        )
        inputs = torch.tensor(
            [
                [
                    [1.0, 10.0],
                    [2.0, 20.0],
                    [torch.inf, torch.inf],
                    [-torch.inf, -torch.inf],
                ]
            ]
        )

        output = model(
            inputs.reshape(-1, 2),
            row_layout=RowLayout.sequence(
                leading_shape=(1, 4),
                batch_axis=0,
                sequence_axis=1,
                valid_rows=torch.tensor([True, True, False, False]),
                context_sharing_restricted=False,
            ),
        ).reshape(1, 4, 2)

        torch.testing.assert_close(
            output[:, :2],
            torch.tensor([[[3.0, 30.0], [3.0, 30.0]]]),
        )

    def test_grouping_rejects_restricted_or_mismatched_layout_before_reduction(self):
        inputs = torch.ones(8, 2)

        with self.assertRaisesRegex(
            ValueError,
            "context sharing is restricted",
        ):
            build_adaptive_group_plan(
                inputs,
                AdaptiveParameterGroupingScopeOptions.SEQUENCE,
                2,
                RowLayout.sequence(
                    leading_shape=(2, 4),
                    batch_axis=0,
                    sequence_axis=1,
                    context_sharing_restricted=True,
                ),
            )

        with self.assertRaisesRegex(
            ValueError,
            "row_count=6 does not match input row count 8",
        ):
            build_adaptive_group_plan(
                inputs,
                AdaptiveParameterGroupingScopeOptions.SEQUENCE,
                2,
                RowLayout.sequence(
                    leading_shape=(2, 3),
                    batch_axis=0,
                    sequence_axis=1,
                    context_sharing_restricted=False,
                ),
            )

    def test_scope_count_and_physical_divisibility_are_validated(self):
        inputs = torch.ones(8, 2)
        layout = RowLayout.sequence(
            leading_shape=(2, 4),
            batch_axis=0,
            sequence_axis=1,
            context_sharing_restricted=False,
        )
        invalid_cases = (
            (
                AdaptiveParameterGroupingScopeOptions.DISABLED,
                2,
                "Cannot build an adaptive group plan for DISABLED grouping",
            ),
            (
                AdaptiveParameterGroupingScopeOptions.SEQUENCE,
                True,
                "group_count must be a positive integer",
            ),
            (
                AdaptiveParameterGroupingScopeOptions.SEQUENCE,
                0,
                "group_count must be a positive integer",
            ),
            (
                AdaptiveParameterGroupingScopeOptions.SEQUENCE,
                3,
                "sequence length 4 must be divisible by group_count=3",
            ),
            (
                AdaptiveParameterGroupingScopeOptions.SEQUENCE,
                5,
                "group_count=5 cannot exceed sequence length 4",
            ),
        )

        for grouping_scope, group_count, message in invalid_cases:
            with self.subTest(
                grouping_scope=grouping_scope,
                group_count=group_count,
            ):
                with self.assertRaisesRegex((TypeError, ValueError), message):
                    build_adaptive_group_plan(
                        inputs,
                        grouping_scope,
                        group_count,
                        layout,
                    )

    def test_restore_rejects_non_tensor_rank_and_leading_shape_before_reshape(self):
        inputs = torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
        valid_rows = torch.tensor([True, False, False, True])
        inputs_before = inputs.clone()
        valid_rows_before = valid_rows.clone()
        layout = RowLayout.rows(
            4,
            valid_rows=valid_rows,
            context_sharing_restricted=False,
        )
        plan = build_adaptive_group_plan(
            inputs,
            AdaptiveParameterGroupingScopeOptions.ROWS,
            2,
            layout,
        )
        cases = (
            (
                [1.0, 2.0],
                TypeError,
                "grouped_output must be a Tensor, received list.",
            ),
            (
                torch.ones(2, 2),
                ValueError,
                "grouped_output must have shape (context_count, members_per_group, "
                "output_dim), received (2, 2).",
            ),
            (
                torch.ones(2, 3, 1),
                ValueError,
                "grouped_output leading dimensions must equal (2, 2), received (2, 3).",
            ),
        )

        for grouped_output, exception_type, message in cases:
            with self.subTest(grouped_output=grouped_output):
                with self.assertRaises(exception_type) as caught:
                    plan.restore(grouped_output)
                self.assertEqual(str(caught.exception), message)

        torch.testing.assert_close(
            plan.valid_members,
            torch.tensor([[True, False], [False, True]]),
        )
        torch.testing.assert_close(inputs, inputs_before)
        torch.testing.assert_close(valid_rows, valid_rows_before)

    def test_plan_input_scope_and_layout_guards_report_the_failing_value(self):
        inputs = torch.ones(4, 2)
        rows_layout = RowLayout.rows(4, context_sharing_restricted=False)
        meta_mask_layout = RowLayout.rows(
            4,
            valid_rows=torch.ones(4, dtype=torch.bool, device="meta"),
            context_sharing_restricted=False,
        )
        cases = (
            (
                [1.0, 2.0],
                AdaptiveParameterGroupingScopeOptions.ROWS,
                rows_layout,
                TypeError,
                "input_rows must be a Tensor, received list.",
            ),
            (
                torch.ones(2, 2, 1),
                AdaptiveParameterGroupingScopeOptions.ROWS,
                rows_layout,
                ValueError,
                "input_rows must be a two-dimensional matrix, received shape "
                "(2, 2, 1).",
            ),
            (
                inputs,
                "ROWS",
                rows_layout,
                TypeError,
                "grouping_scope must be an AdaptiveParameterGroupingScopeOptions "
                "value, received 'ROWS'.",
            ),
            (
                inputs,
                AdaptiveParameterGroupingScopeOptions.ROWS,
                object(),
                TypeError,
                "row_layout must be a RowLayout, received object.",
            ),
            (
                inputs,
                AdaptiveParameterGroupingScopeOptions.ROWS,
                meta_mask_layout,
                ValueError,
                "row_layout.valid_rows must be on the same device as input_rows, "
                "received meta and cpu.",
            ),
        )

        for input_rows, scope, layout, exception_type, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(exception_type) as caught:
                    build_adaptive_group_plan(input_rows, scope, 2, layout)
                self.assertEqual(str(caught.exception), message)

        torch.testing.assert_close(inputs, torch.ones(4, 2))

    def test_grouping_scope_requires_matching_layout_semantics(self):
        inputs = torch.ones(4, 2)
        rows_layout = RowLayout.rows(4, context_sharing_restricted=False)
        sequence_layout = RowLayout.sequence(
            leading_shape=(2, 2),
            batch_axis=0,
            sequence_axis=1,
            context_sharing_restricted=False,
        )
        cases = (
            (
                AdaptiveParameterGroupingScopeOptions.ROWS,
                sequence_layout,
                "ROWS grouping requires a one-axis row layout.",
            ),
            (
                AdaptiveParameterGroupingScopeOptions.SEQUENCE,
                rows_layout,
                "SEQUENCE grouping requires a two-axis sequence layout.",
            ),
        )

        for scope, layout, message in cases:
            with self.subTest(scope=scope):
                with self.assertRaises(ValueError) as caught:
                    build_adaptive_group_plan(inputs, scope, 2, layout)
                self.assertEqual(str(caught.exception), message)

        future_scope = object.__new__(AdaptiveParameterGroupingScopeOptions)
        object.__setattr__(future_scope, "_name_", "FUTURE")
        object.__setattr__(future_scope, "_value_", 99)

        with self.assertRaises(ValueError) as caught:
            build_adaptive_group_plan(inputs, future_scope, 2, rows_layout)

        self.assertEqual(
            str(caught.exception),
            "Unsupported adaptive parameter grouping scope "
            "<AdaptiveParameterGroupingScopeOptions.FUTURE: 99>.",
        )

    def test_generated_parameter_context_and_grouped_bias_guards_are_exact(self):
        input_batch = torch.ones(3, 2)
        weights = torch.ones(2, 2, 2)
        bias = torch.ones(4, 2)
        grouped_bias = torch.ones(2, 2)
        cases = (
            (
                lambda: AdaptiveLinearValidator.validate_weight_context_count(
                    input_batch,
                    weights,
                ),
                "Dynamic weights context count must match affine input, received "
                "2 and 3.",
            ),
            (
                lambda: AdaptiveLinearValidator.validate_bias_context_count(
                    input_batch,
                    bias,
                ),
                "Dynamic bias context count must match affine input, received 4 and 3.",
            ),
            (
                lambda: (
                    AdaptiveParameterAugmentationValidator.validate_grouped_base_parameters(
                        torch.ones(2, 2),
                        grouped_bias,
                    )
                ),
                "Adaptive parameter grouping requires a shared one-dimensional base "
                "bias; row-specific base bias is not supported.",
            ),
        )

        for action, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(ValueError) as caught:
                    action()
                self.assertEqual(str(caught.exception), message)


class GroupedAdaptiveLinearTests(unittest.TestCase):
    def test_buildable_augmentation_owns_grouping_and_restoration(self):
        linear = grouped_adaptive_bias_linear(
            AdaptiveParameterGroupingScopeOptions.ROWS,
            2,
        )
        augmentation = linear.adaptive_behaviour
        inputs = torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])

        output = augmentation(
            linear._compute_affine_transformation_callback,
            linear.weight_params,
            linear.bias_params,
            inputs,
            row_layout=RowLayout.rows(
                4,
                context_sharing_restricted=False,
            ),
        )

        self.assertIs(
            augmentation.cfg.registry_owner(),
            AdaptiveParameterAugmentation,
        )
        torch.testing.assert_close(
            output,
            torch.tensor([[3.0, 30.0], [3.0, 30.0], [7.0, 70.0], [7.0, 70.0]]),
        )

    def test_grouped_augmentation_rejects_row_specific_base_parameters_first(self):
        linear = grouped_adaptive_bias_linear(
            AdaptiveParameterGroupingScopeOptions.ROWS,
            2,
        )
        augmentation = linear.adaptive_behaviour
        generator_calls = []
        hook = augmentation.bias_model.register_forward_hook(
            lambda *_args: generator_calls.append(True)
        )
        inputs = torch.ones(4, 2)
        row_specific_weights = linear.weight_params.unsqueeze(0).expand(4, -1, -1)

        try:
            with self.assertRaisesRegex(
                ValueError,
                "requires shared two-dimensional base weights",
            ):
                augmentation(
                    linear._compute_affine_transformation_callback,
                    row_specific_weights,
                    linear.bias_params,
                    inputs,
                    row_layout=RowLayout.rows(
                        4,
                        context_sharing_restricted=False,
                    ),
                )
        finally:
            hook.remove()

        self.assertEqual(generator_calls, [])

    def test_rows_generate_one_bias_per_group_and_apply_it_to_each_member(self):
        model = grouped_adaptive_bias_linear(
            AdaptiveParameterGroupingScopeOptions.ROWS,
            2,
        )
        inputs = torch.tensor(
            [
                [1.0, 10.0],
                [2.0, 20.0],
                [3.0, 30.0],
                [4.0, 40.0],
            ],
            requires_grad=True,
        )

        output = model(
            inputs,
            row_layout=RowLayout.rows(
                4,
                context_sharing_restricted=False,
            ),
        )

        expected = torch.tensor(
            [
                [3.0, 30.0],
                [3.0, 30.0],
                [7.0, 70.0],
                [7.0, 70.0],
            ]
        )
        torch.testing.assert_close(output, expected)
        output.sum().backward()
        torch.testing.assert_close(inputs.grad, torch.full_like(inputs, 2.0))

    def test_sequence_grouping_does_not_mix_batches(self):
        model = grouped_adaptive_bias_linear(
            AdaptiveParameterGroupingScopeOptions.SEQUENCE,
            2,
        )
        logical_inputs = torch.tensor(
            [
                [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]],
                [
                    [100.0, 1000.0],
                    [200.0, 2000.0],
                    [300.0, 3000.0],
                    [400.0, 4000.0],
                ],
            ]
        )

        output = model(
            logical_inputs.reshape(-1, 2),
            row_layout=RowLayout.sequence(
                leading_shape=(2, 4),
                batch_axis=0,
                sequence_axis=1,
                context_sharing_restricted=False,
            ),
        ).reshape(2, 4, 2)

        expected = torch.tensor(
            [
                [[3.0, 30.0], [3.0, 30.0], [7.0, 70.0], [7.0, 70.0]],
                [
                    [300.0, 3000.0],
                    [300.0, 3000.0],
                    [700.0, 7000.0],
                    [700.0, 7000.0],
                ],
            ]
        )
        torch.testing.assert_close(output, expected)

    def test_dense_grouped_weights_stay_compact_during_application(self):
        model = grouped_adaptive_bias_linear(
            AdaptiveParameterGroupingScopeOptions.ROWS,
            2,
        )
        shape_spy = DenseWeightShapeSpy()
        model.adaptive_behaviour.weight_model = shape_spy
        model.adaptive_behaviour.bias_model = None
        inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])

        with patch(
            "emperor.augmentations.adaptive_parameters._linear_adapter.torch.bmm",
            wraps=torch.bmm,
        ) as grouped_matrix_multiply:
            output = model(
                inputs,
                row_layout=RowLayout.rows(
                    4,
                    context_sharing_restricted=False,
                ),
            )

        contexts_repeated = torch.tensor(
            [[4.0, 6.0], [4.0, 6.0], [12.0, 14.0], [12.0, 14.0]]
        )
        torch.testing.assert_close(output, inputs * contexts_repeated)
        self.assertEqual(shape_spy.conditioning_shape, (2, 2))
        self.assertEqual(shape_spy.generated_weight_shape, (2, 2, 2))
        grouped_members, generated_weights = grouped_matrix_multiply.call_args.args
        self.assertEqual(tuple(grouped_members.shape), (2, 2, 2))
        self.assertEqual(tuple(generated_weights.shape), (2, 2, 2))

    def test_generated_weight_gradients_drive_exact_generator_updates(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                model = grouped_adaptive_bias_linear(
                    AdaptiveParameterGroupingScopeOptions.ROWS,
                    2,
                ).to(dtype=dtype)
                generator = DeterministicDynamicWeightGenerator(dtype)
                model.adaptive_behaviour.weight_model = generator
                model.adaptive_behaviour.bias_model = None
                inputs = torch.tensor(
                    [[1.0, 2.0], [0.0, 2.0], [4.0, 0.0], [5.0, 6.0]],
                    dtype=dtype,
                )
                loss_coefficients = torch.tensor(
                    [[1.0, 2.0], [3.0, -2.0], [-1.0, 1.0], [2.0, -2.0]],
                    dtype=dtype,
                )
                layout = RowLayout.rows(4, context_sharing_restricted=False)
                optimizer = torch.optim.SGD(generator.parameters(), lr=0.05)
                scale_before = generator.scale.detach().clone()
                offset_before = generator.offset.detach().clone()

                output = model(inputs, row_layout=layout)
                loss = (output * loss_coefficients).sum()
                loss.backward()

                expected_generated_gradient = torch.tensor(
                    [
                        [[1.0, 2.0], [8.0, 0.0]],
                        [[6.0, -6.0], [12.0, -12.0]],
                    ],
                    dtype=dtype,
                )
                expected_scale_gradient = torch.tensor(
                    [[55.0, -52.0], [104.0, -72.0]],
                    dtype=dtype,
                )
                expected_offset_gradient = torch.tensor(
                    [[7.0, -4.0], [20.0, -12.0]],
                    dtype=dtype,
                )

                torch.testing.assert_close(
                    generator.conditioning_input,
                    torch.tensor([[1.0, 4.0], [9.0, 6.0]], dtype=dtype),
                )
                self.assertIsNotNone(generator.generated_weights.grad)
                torch.testing.assert_close(
                    generator.generated_weights.grad,
                    expected_generated_gradient,
                )
                torch.testing.assert_close(
                    generator.scale.grad,
                    expected_scale_gradient,
                )
                torch.testing.assert_close(
                    generator.offset.grad,
                    expected_offset_gradient,
                )

                optimizer.step()

                torch.testing.assert_close(
                    generator.scale,
                    scale_before - 0.05 * expected_scale_gradient,
                )
                torch.testing.assert_close(
                    generator.offset,
                    offset_before - 0.05 * expected_offset_gradient,
                )

    def test_missing_or_restricted_layout_fails_before_generator_execution(self):
        model = grouped_adaptive_bias_linear(
            AdaptiveParameterGroupingScopeOptions.ROWS,
            2,
        )
        generator = model.adaptive_behaviour.bias_model.model[0].model
        calls = []
        hook = generator.register_forward_hook(lambda *_args: calls.append(True))
        inputs = torch.ones(4, 2)

        try:
            with self.assertRaisesRegex(ValueError, "requires an explicit RowLayout"):
                model(inputs)
            with self.assertRaisesRegex(ValueError, "context sharing is restricted"):
                model(
                    inputs,
                    row_layout=RowLayout.rows(
                        4,
                        context_sharing_restricted=True,
                    ),
                )
        finally:
            hook.remove()

        self.assertEqual(calls, [])

    def test_enabled_grouping_requires_an_active_adaptive_component(self):
        config = AdaptiveLinearLayerConfig(
            input_dim=2,
            output_dim=2,
            bias_flag=True,
            adaptive_augmentation_config=AdaptiveParameterAugmentationConfig(
                grouping_scope=AdaptiveParameterGroupingScopeOptions.ROWS,
                group_count=2,
            ),
        )

        with self.assertRaisesRegex(
            ValueError,
            "enabled grouping requires at least one adaptive parameter component",
        ):
            AdaptiveLinearLayer(config)

    def test_resolved_grouping_scope_is_required(self):
        for group_count in (None, 2):
            with self.subTest(group_count=group_count):
                config = AdaptiveLinearLayerConfig(
                    input_dim=2,
                    output_dim=2,
                    bias_flag=True,
                    adaptive_augmentation_config=(
                        AdaptiveParameterAugmentationConfig(
                            group_count=group_count,
                        )
                    ),
                )

                with self.assertRaisesRegex(
                    ValueError,
                    "grouping_scope is required for a resolved",
                ):
                    AdaptiveLinearLayer(config)

    def test_direct_augmentation_requires_a_resolved_grouping_scope(self):
        with self.assertRaisesRegex(
            ValueError,
            "grouping_scope is required for a resolved",
        ):
            AdaptiveParameterAugmentation(
                AdaptiveParameterAugmentationConfig(
                    input_dim=2,
                    output_dim=2,
                )
            )

    def test_invalid_scope_without_components_is_not_silently_treated_as_static(self):
        config = AdaptiveLinearLayerConfig(
            input_dim=2,
            output_dim=2,
            bias_flag=True,
            adaptive_augmentation_config=AdaptiveParameterAugmentationConfig(
                grouping_scope="ROWS",
                group_count=2,
            ),
        )

        with self.assertRaisesRegex(
            TypeError,
            "grouping_scope must be an AdaptiveParameterGroupingScopeOptions value",
        ):
            AdaptiveLinearLayer(config)

    def test_disabled_scope_rejects_a_malformed_dormant_count(self):
        for invalid_count in (True, 0, -1, "two"):
            with self.subTest(invalid_count=invalid_count):
                config = AdaptiveLinearLayerConfig(
                    input_dim=2,
                    output_dim=2,
                    bias_flag=True,
                    adaptive_augmentation_config=(
                        AdaptiveParameterAugmentationConfig(
                            grouping_scope=(
                                AdaptiveParameterGroupingScopeOptions.DISABLED
                            ),
                            group_count=invalid_count,
                        )
                    ),
                )

                with self.assertRaisesRegex(
                    ValueError,
                    "group_count must be a positive integer",
                ):
                    AdaptiveLinearLayer(config)

    def test_public_augmentation_does_not_expose_application_input(self):
        self.assertNotIn(
            "application_input",
            signature(AdaptiveParameterAugmentation.forward).parameters,
        )

    def test_same_group_jacobian_is_coupled_but_other_groups_are_isolated(self):
        model = grouped_adaptive_bias_linear(
            AdaptiveParameterGroupingScopeOptions.ROWS,
            2,
        )
        inputs = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
            requires_grad=True,
        )
        layout = RowLayout.rows(4, context_sharing_restricted=False)

        jacobian = torch.autograd.functional.jacobian(
            lambda value: model(value, row_layout=layout)[0, 0],
            inputs,
        )

        torch.testing.assert_close(
            jacobian,
            torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
        )

    def test_grouped_application_supports_finite_second_order_input_gradients(self):
        model = grouped_adaptive_bias_linear(
            AdaptiveParameterGroupingScopeOptions.ROWS,
            2,
        )
        inputs = torch.randn(4, 2, requires_grad=True)
        layout = RowLayout.rows(4, context_sharing_restricted=False)

        output = model(inputs, row_layout=layout)
        first_gradient = torch.autograd.grad(
            output.pow(3).sum(),
            inputs,
            create_graph=True,
        )[0]
        second_gradient = torch.autograd.grad(first_gradient.sum(), inputs)[0]

        self.assertTrue(torch.isfinite(first_gradient).all())
        self.assertTrue(torch.isfinite(second_gradient).all())

    def test_one_group_per_row_matches_legacy_per_row_adaptivity(self):
        grouped = grouped_adaptive_bias_linear(
            AdaptiveParameterGroupingScopeOptions.ROWS,
            4,
        )
        legacy_config = grouped.cfg
        legacy_config.adaptive_augmentation_config.grouping_scope = (
            AdaptiveParameterGroupingScopeOptions.DISABLED
        )
        legacy_config.adaptive_augmentation_config.group_count = None
        legacy = AdaptiveLinearLayer(legacy_config)
        legacy.load_state_dict(grouped.state_dict(), strict=True)
        inputs = torch.randn(4, 2)

        grouped_output = grouped(
            inputs,
            row_layout=RowLayout.rows(
                4,
                context_sharing_restricted=False,
            ),
        )
        legacy_output = legacy(inputs)

        torch.testing.assert_close(grouped_output, legacy_output)
        self.assertEqual(tuple(grouped.state_dict()), tuple(legacy.state_dict()))

    def test_explicit_disabled_scope_restores_legacy_call_contract(self):
        grouped = grouped_adaptive_bias_linear(
            AdaptiveParameterGroupingScopeOptions.ROWS,
            2,
        )
        disabled_augmentation = grouped.adaptive_behaviour.cfg.build(
            AdaptiveParameterAugmentationConfig(
                grouping_scope=AdaptiveParameterGroupingScopeOptions.DISABLED,
            )
        )
        inputs = torch.randn(4, 2)

        output = disabled_augmentation(
            grouped._compute_affine_transformation_callback,
            grouped.weight_params,
            grouped.bias_params,
            inputs,
        )

        self.assertEqual(tuple(output.shape), (4, 2))
        self.assertFalse(disabled_augmentation.adaptive_parameter_grouping_enabled)
        self.assertEqual(disabled_augmentation.group_count, 2)
        self.assertEqual(
            tuple(disabled_augmentation.state_dict()),
            tuple(grouped.adaptive_behaviour.state_dict()),
        )
        self.assertNotIn(
            "adaptive_parameter_grouping_enabled",
            disabled_augmentation.__dict__,
        )
        disabled_augmentation.grouping_scope = (
            AdaptiveParameterGroupingScopeOptions.ROWS
        )
        self.assertTrue(disabled_augmentation.adaptive_parameter_grouping_enabled)

    def test_partial_override_inherits_enabled_grouping_scope(self):
        grouped = grouped_adaptive_bias_linear(
            AdaptiveParameterGroupingScopeOptions.ROWS,
            2,
        )

        inherited = grouped.adaptive_behaviour.cfg.build(
            AdaptiveParameterAugmentationConfig(
                input_dim=2,
                output_dim=2,
            )
        )

        self.assertEqual(
            inherited.grouping_scope,
            AdaptiveParameterGroupingScopeOptions.ROWS,
        )
        self.assertEqual(inherited.group_count, 2)
        self.assertTrue(inherited.adaptive_parameter_grouping_enabled)

    def test_outer_override_must_supply_a_resolved_nested_scope(self):
        grouped = grouped_adaptive_bias_linear(
            AdaptiveParameterGroupingScopeOptions.ROWS,
            2,
        )
        outer_override = AdaptiveLinearLayerConfig(
            adaptive_augmentation_config=AdaptiveParameterAugmentationConfig(
                input_dim=2,
                output_dim=2,
            )
        )

        with self.assertRaisesRegex(
            ValueError,
            "grouping_scope is required for a resolved",
        ):
            AdaptiveLinearLayer(grouped.cfg, outer_override)

    def test_all_dynamic_weight_variants_generate_context_batched_matrices(self):
        common = dict(
            input_dim=2,
            output_dim=3,
            generator_depth=DynamicDepthOptions.DEPTH_OF_ONE,
            decay_schedule=WeightDecayScheduleOptions.DISABLED,
            decay_rate=0.0,
            decay_warmup_batches=0,
            model_config=linear_stack_config(2, 3),
        )
        square_common = {
            **common,
            "output_dim": 2,
            "model_config": linear_stack_config(2, 2),
        }
        variants = (
            SingleModelDynamicWeightConfig(
                **square_common,
                normalization_option=WeightNormalizationOptions.DISABLED,
                normalization_position_option=(
                    WeightNormalizationPositionOptions.DISABLED
                ),
            ),
            DualModelDynamicWeightConfig(
                **common,
                normalization_option=WeightNormalizationOptions.DISABLED,
                normalization_position_option=(
                    WeightNormalizationPositionOptions.DISABLED
                ),
            ),
            LowRankDynamicWeightConfig(
                **common,
                normalization_option=WeightNormalizationOptions.DISABLED,
            ),
            HypernetworkDynamicWeightConfig(
                **common,
                normalization_option=WeightNormalizationOptions.DISABLED,
            ),
            LayeredWeightedBankDynamicWeightConfig(
                **common,
                bank_expansion_factor=BankExpansionFactorOptions.FACTOR_OF_ONE,
            ),
            SoftWeightedBankDynamicWeightConfig(
                **common,
                bank_expansion_factor=BankExpansionFactorOptions.FACTOR_OF_ONE,
            ),
        )
        layout = RowLayout.rows(4, context_sharing_restricted=False)

        for weight_config in variants:
            with self.subTest(weight_config=type(weight_config).__name__):
                output_dim = weight_config.output_dim
                model = AdaptiveLinearLayer(
                    AdaptiveLinearLayerConfig(
                        input_dim=2,
                        output_dim=output_dim,
                        bias_flag=False,
                        adaptive_augmentation_config=(
                            AdaptiveParameterAugmentationConfig(
                                weight_config=weight_config,
                                grouping_scope=(
                                    AdaptiveParameterGroupingScopeOptions.ROWS
                                ),
                                group_count=2,
                            )
                        ),
                    )
                )
                observed_shapes = []
                hook = model.adaptive_behaviour.weight_model.register_forward_hook(
                    lambda _module, _args, output, observed=observed_shapes: (
                        observed.append(tuple(output.shape))
                    )
                )
                inputs = torch.randn(4, 2, requires_grad=True)
                try:
                    output = model(inputs, row_layout=layout)
                finally:
                    hook.remove()

                self.assertEqual(tuple(output.shape), (4, output_dim))
                self.assertEqual(observed_shapes, [(2, 2, output_dim)])
                self.assertTrue(torch.isfinite(output).all())
                output.square().mean().backward()
                self.assertTrue(torch.isfinite(inputs.grad).all())

    def test_all_dynamic_bias_variants_generate_one_vector_per_context(self):
        common = dict(
            input_dim=2,
            output_dim=3,
            decay_schedule=WeightDecayScheduleOptions.DISABLED,
            decay_rate=0.0,
            decay_warmup_batches=0,
            model_config=linear_stack_config(2, 3),
        )
        variants = (
            AffineTransformDynamicBiasConfig(**common),
            AdditiveDynamicBiasConfig(**common),
            MultiplicativeDynamicBiasConfig(**common),
            SigmoidGatedDynamicBiasConfig(**common),
            TanhGatedDynamicBiasConfig(**common),
            GeneratorDynamicBiasConfig(**common),
            WeightedBankDynamicBiasConfig(
                **common,
                bank_expansion_factor=BankExpansionFactorOptions.FACTOR_OF_ONE,
            ),
        )
        layout = RowLayout.rows(4, context_sharing_restricted=False)

        for bias_config in variants:
            with self.subTest(bias_config=type(bias_config).__name__):
                model = AdaptiveLinearLayer(
                    AdaptiveLinearLayerConfig(
                        input_dim=2,
                        output_dim=3,
                        bias_flag=True,
                        adaptive_augmentation_config=(
                            AdaptiveParameterAugmentationConfig(
                                bias_config=bias_config,
                                grouping_scope=(
                                    AdaptiveParameterGroupingScopeOptions.ROWS
                                ),
                                group_count=2,
                            )
                        ),
                    )
                )
                observed_shapes = []
                hook = model.adaptive_behaviour.bias_model.register_forward_hook(
                    lambda _module, _args, output, observed=observed_shapes: (
                        observed.append(tuple(output.shape))
                    )
                )
                inputs = torch.randn(4, 2, requires_grad=True)
                try:
                    output = model(inputs, row_layout=layout)
                finally:
                    hook.remove()

                self.assertEqual(tuple(output.shape), (4, 3))
                self.assertEqual(observed_shapes, [(2, 3)])
                output.square().mean().backward()
                self.assertTrue(torch.isfinite(inputs.grad).all())

    def test_all_dynamic_diagonal_variants_keep_dense_updates_context_batched(self):
        common = dict(
            input_dim=2,
            output_dim=3,
            model_config=linear_stack_config(2, 3),
        )
        variants = (
            StandardDynamicDiagonalConfig(**common),
            AntiDynamicDiagonalConfig(**common),
            CombinedDynamicDiagonalConfig(**common),
        )
        layout = RowLayout.rows(4, context_sharing_restricted=False)

        for diagonal_config in variants:
            with self.subTest(diagonal_config=type(diagonal_config).__name__):
                model = AdaptiveLinearLayer(
                    AdaptiveLinearLayerConfig(
                        input_dim=2,
                        output_dim=3,
                        bias_flag=False,
                        adaptive_augmentation_config=(
                            AdaptiveParameterAugmentationConfig(
                                diagonal_config=diagonal_config,
                                grouping_scope=(
                                    AdaptiveParameterGroupingScopeOptions.ROWS
                                ),
                                group_count=2,
                            )
                        ),
                    )
                )
                observed_shapes = []
                hook = model.adaptive_behaviour.diagonal_model.register_forward_hook(
                    lambda _module, _args, output, observed=observed_shapes: (
                        observed.append(tuple(output.shape))
                    )
                )
                inputs = torch.randn(4, 2, requires_grad=True)
                try:
                    output = model(inputs, row_layout=layout)
                finally:
                    hook.remove()

                self.assertEqual(tuple(output.shape), (4, 3))
                self.assertEqual(observed_shapes, [(2, 2, 3)])
                output.square().mean().backward()
                self.assertTrue(torch.isfinite(inputs.grad).all())

    def test_all_dynamic_mask_variants_keep_masked_weights_context_batched(self):
        common = dict(
            input_dim=2,
            output_dim=3,
            mask_threshold=0.5,
            mask_surrogate_scale=1.0,
            mask_floor=0.0,
            model_config=linear_stack_config(2, 3),
        )
        variants = (
            WeightInformedScoreAxisMaskConfig(
                **common,
                mask_dimension_option=MaskDimensionOptions.ROW,
            ),
            WeightInformedScoreAxisMaskConfig(
                **common,
                mask_dimension_option=MaskDimensionOptions.COLUMN,
            ),
            PerAxisScoreMaskConfig(
                **common,
                mask_dimension_option=MaskDimensionOptions.ROW,
            ),
            PerAxisScoreMaskConfig(
                **common,
                mask_dimension_option=MaskDimensionOptions.COLUMN,
            ),
            TopSliceAxisMaskConfig(
                **common,
                mask_dimension_option=MaskDimensionOptions.ROW,
                mask_transition_width=1.0,
            ),
            TopSliceAxisMaskConfig(
                **common,
                mask_dimension_option=MaskDimensionOptions.COLUMN,
                mask_transition_width=1.0,
            ),
            OuterProductMaskConfig(**common),
            DiagonalAxisMaskConfig(
                **common,
                mask_transition_width=1.0,
            ),
        )
        layout = RowLayout.rows(4, context_sharing_restricted=False)

        for mask_config in variants:
            with self.subTest(mask_config=type(mask_config).__name__):
                model = AdaptiveLinearLayer(
                    AdaptiveLinearLayerConfig(
                        input_dim=2,
                        output_dim=3,
                        bias_flag=False,
                        adaptive_augmentation_config=(
                            AdaptiveParameterAugmentationConfig(
                                mask_config=mask_config,
                                grouping_scope=(
                                    AdaptiveParameterGroupingScopeOptions.ROWS
                                ),
                                group_count=2,
                            )
                        ),
                    )
                )
                observed_shapes = []
                hook = model.adaptive_behaviour.mask_model.register_forward_hook(
                    lambda _module, _args, output, observed=observed_shapes: (
                        observed.append(tuple(output.shape))
                    )
                )
                inputs = torch.randn(4, 2, requires_grad=True)
                try:
                    output = model(inputs, row_layout=layout)
                finally:
                    hook.remove()

                self.assertEqual(tuple(output.shape), (4, 3))
                self.assertEqual(observed_shapes, [(2, 2, 3)])
                output.square().mean().backward()
                self.assertTrue(torch.isfinite(inputs.grad).all())

    def test_combined_pipeline_never_expands_dynamic_parameters_to_row_count(self):
        generator_config = linear_stack_config(2, 3)
        model = AdaptiveLinearLayer(
            AdaptiveLinearLayerConfig(
                input_dim=2,
                output_dim=3,
                bias_flag=True,
                adaptive_augmentation_config=AdaptiveParameterAugmentationConfig(
                    weight_config=DualModelDynamicWeightConfig(
                        input_dim=2,
                        output_dim=3,
                        generator_depth=DynamicDepthOptions.DEPTH_OF_ONE,
                        decay_schedule=WeightDecayScheduleOptions.DISABLED,
                        decay_rate=0.0,
                        decay_warmup_batches=0,
                        normalization_option=WeightNormalizationOptions.DISABLED,
                        normalization_position_option=(
                            WeightNormalizationPositionOptions.DISABLED
                        ),
                        model_config=generator_config,
                    ),
                    diagonal_config=StandardDynamicDiagonalConfig(
                        input_dim=2,
                        output_dim=3,
                        model_config=generator_config,
                    ),
                    bias_config=AdditiveDynamicBiasConfig(
                        input_dim=2,
                        output_dim=3,
                        decay_schedule=WeightDecayScheduleOptions.DISABLED,
                        decay_rate=0.0,
                        decay_warmup_batches=0,
                        model_config=generator_config,
                    ),
                    mask_config=PerAxisScoreMaskConfig(
                        input_dim=2,
                        output_dim=3,
                        mask_dimension_option=MaskDimensionOptions.COLUMN,
                        mask_threshold=0.5,
                        mask_surrogate_scale=1.0,
                        mask_floor=0.0,
                        model_config=generator_config,
                    ),
                    grouping_scope=AdaptiveParameterGroupingScopeOptions.ROWS,
                    group_count=2,
                ),
            )
        )
        observed_shapes = {}
        hooks = []

        def capture_shape(slot):
            def capture(_module, _args, output):
                observed_shapes[slot] = tuple(output.shape)

            return capture

        for name in ("weight", "diagonal", "bias", "mask"):
            option = getattr(model.adaptive_behaviour, f"{name}_model")
            hooks.append(option.register_forward_hook(capture_shape(name)))
        inputs = torch.randn(4, 2, requires_grad=True)

        try:
            output = model(
                inputs,
                row_layout=RowLayout.rows(
                    4,
                    context_sharing_restricted=False,
                ),
            )
        finally:
            for hook in hooks:
                hook.remove()

        self.assertEqual(tuple(output.shape), (4, 3))
        self.assertEqual(
            observed_shapes,
            {
                "weight": (2, 2, 3),
                "diagonal": (2, 2, 3),
                "bias": (2, 3),
                "mask": (2, 2, 3),
            },
        )
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(inputs.grad).all())

    def test_low_precision_sum_uses_native_input_dtype(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                model = grouped_adaptive_bias_linear(
                    AdaptiveParameterGroupingScopeOptions.ROWS,
                    1,
                ).to(dtype=dtype)
                inputs = torch.full((64, 2), 1000.0, dtype=dtype)
                output = model(
                    inputs,
                    row_layout=RowLayout.rows(
                        64,
                        context_sharing_restricted=False,
                    ),
                )

                self.assertEqual(output.dtype, dtype)
                expected_context = inputs.sum(dim=0)
                torch.testing.assert_close(
                    output,
                    expected_context.expand_as(output),
                )

    def test_padding_has_zero_context_gradient_for_valid_row_objectives(self):
        model = grouped_adaptive_bias_linear(
            AdaptiveParameterGroupingScopeOptions.SEQUENCE,
            2,
        )
        inputs = torch.tensor(
            [[[1.0, 10.0], [2.0, 20.0], [1000.0, 2000.0], [3000.0, 4000.0]]],
            requires_grad=True,
        )
        valid_rows = torch.tensor([True, True, False, False])

        output = model(
            inputs.reshape(-1, 2),
            row_layout=RowLayout.sequence(
                leading_shape=(1, 4),
                batch_axis=0,
                sequence_axis=1,
                valid_rows=valid_rows,
                context_sharing_restricted=False,
            ),
        ).reshape(1, 4, 2)
        output[:, :2].sum().backward()

        torch.testing.assert_close(
            output[:, :2],
            torch.tensor([[[3.0, 30.0], [3.0, 30.0]]]),
        )
        torch.testing.assert_close(
            inputs.grad[:, 2:],
            torch.zeros_like(inputs.grad[:, 2:]),
        )


if __name__ == "__main__":
    unittest.main()

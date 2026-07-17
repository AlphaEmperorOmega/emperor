from __future__ import annotations

import unittest

import torch

from emperor.augmentations.adaptive_parameters import (
    DiagonalAxisMaskConfig,
    MaskDimensionOptions,
    OuterProductMaskConfig,
    PerAxisScoreMaskConfig,
    TopSliceAxisMaskConfig,
)
from emperor.augmentations.adaptive_parameters._masks.variants.diagonal import (
    DiagonalAxisMask,
)
from emperor.augmentations.adaptive_parameters._masks.variants.outer_product import (
    OuterProductMask,
)
from emperor.augmentations.adaptive_parameters._masks.variants.per_axis import (
    PerAxisScoreMask,
)
from emperor.augmentations.adaptive_parameters._masks.variants.top_slice import (
    TopSliceAxisMask,
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


def assign_linear(
    stack: LayerStack,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> None:
    linear = stack[0].model
    if not isinstance(linear, LinearLayer):
        raise TypeError(f"Expected LinearLayer, received {type(linear).__name__}.")
    with torch.no_grad():
        linear.weight_params.copy_(weight)
        linear.bias_params.copy_(bias)


class AdaptiveParameterMaskMutationContractTests(unittest.TestCase):
    def test_mask_generator_overrides_source_dimensions_and_runs_exact_math(
        self,
    ) -> None:
        model = PerAxisScoreMask(
            PerAxisScoreMaskConfig(
                input_dim=2,
                output_dim=3,
                mask_threshold=0.5,
                mask_surrogate_scale=0.0,
                mask_floor=0.25,
                mask_dimension_option=MaskDimensionOptions.ROW,
                model_config=linear_stack_config(7, 11),
            )
        )
        self.assertEqual((model.input_dim, model.output_dim), (2, 3))
        self.assertEqual(
            tuple(model.model[0].model.weight_params.shape),
            (2, 2),
        )
        assign_linear(
            model.model,
            torch.eye(2),
            torch.zeros(2),
        )
        logits = torch.tensor(
            [
                [torch.logit(torch.tensor(0.8)), torch.logit(torch.tensor(0.2))],
                [torch.logit(torch.tensor(0.4)), torch.logit(torch.tensor(0.9))],
            ]
        )
        weights = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]
        )

        actual = model(weights, logits)

        scores = torch.tensor([[0.8, 0.2], [0.4, 0.9]])
        hard = (scores >= 0.5).float()
        adjusted_hard = 0.25 + 0.75 * hard
        expected = weights.unsqueeze(0) * (adjusted_hard * scores).unsqueeze(-1)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=0.0))

    def test_outer_product_uses_distinct_rectangular_generator_outputs(
        self,
    ) -> None:
        model = OuterProductMask(
            OuterProductMaskConfig(
                input_dim=2,
                output_dim=3,
                mask_threshold=0.5,
                mask_surrogate_scale=0.0,
                mask_floor=0.0,
                model_config=linear_stack_config(7, 11),
            )
        )
        self.assertEqual(
            tuple(model.input_model[0].model.weight_params.shape),
            (2, 2),
        )
        self.assertEqual(
            tuple(model.output_model[0].model.weight_params.shape),
            (2, 3),
        )
        input_weight = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )
        output_weight = torch.tensor(
            [
                [1.0, -1.0, 0.5],
                [0.25, 2.0, -2.0],
            ]
        )
        assign_linear(model.input_model, input_weight, torch.zeros(2))
        assign_linear(model.output_model, output_weight, torch.zeros(3))
        logits = torch.tensor([[1.0, 2.0], [-1.0, 0.5]])
        weights = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]
        )

        actual = model(weights, logits)

        input_vectors = logits @ input_weight
        output_vectors = logits @ output_weight
        scores = torch.sigmoid(
            torch.einsum("bi,bj->bij", input_vectors, output_vectors)
        )
        expected = weights.unsqueeze(0) * (scores >= 0.5).float() * scores
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=0.0))

    def test_top_slice_width_two_tracks_each_sample_boundary_exactly(self) -> None:
        model = TopSliceAxisMask(
            TopSliceAxisMaskConfig(
                input_dim=4,
                output_dim=3,
                mask_threshold=0.5,
                mask_surrogate_scale=0.0,
                mask_floor=0.0,
                mask_dimension_option=MaskDimensionOptions.ROW,
                mask_transition_width=2.0,
                model_config=linear_stack_config(9, 7),
            )
        )
        assign_linear(model.model, torch.eye(4), torch.zeros(4))
        scores = torch.tensor(
            [
                [0.9, 0.8, 0.2, 0.1],
                [0.9, 0.2, 0.8, 0.7],
            ]
        )
        logits = torch.logit(scores)
        weights = torch.arange(1, 13, dtype=torch.float32).reshape(4, 3)

        actual = model(weights, logits)

        transition_scores = torch.tensor(
            [
                [1.0, 1.0, 0.5, 0.0],
                [1.0, 0.5, 0.0, 0.0],
            ]
        )
        hard = (transition_scores >= 0.5).float()
        expected = weights.unsqueeze(0) * (hard * scores).unsqueeze(-1)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=0.0))
        self.assertFalse(torch.equal(actual[0], actual[1]))

    def test_top_slice_positions_stay_on_the_input_device(self) -> None:
        model = TopSliceAxisMask(
            TopSliceAxisMaskConfig(
                input_dim=4,
                output_dim=3,
                mask_threshold=0.5,
                mask_surrogate_scale=0.0,
                mask_floor=0.0,
                mask_dimension_option=MaskDimensionOptions.ROW,
                mask_transition_width=2.0,
                model_config=linear_stack_config(4, 4),
            )
        ).double()
        scores = torch.tensor(
            [[0.9, 0.8, 0.2, 0.1], [0.9, 0.2, 0.8, 0.7]],
            dtype=torch.float64,
        )
        expected = torch.tensor(
            [[1.0, 1.0, 0.5, 0.0], [1.0, 0.5, 0.0, 0.0]],
            dtype=torch.float64,
        )

        with torch.device("meta"):
            actual = model._TopSliceAxisMask__compute_transition_scores(scores)

        self.assertEqual(actual.device, scores.device)
        self.assertEqual(actual.dtype, scores.dtype)
        self.assertTrue(torch.equal(actual, expected))

    def test_diagonal_geometry_preserves_tensor_device_dtype_and_batch(self) -> None:
        model = DiagonalAxisMask(
            DiagonalAxisMaskConfig(
                input_dim=3,
                output_dim=4,
                mask_threshold=0.5,
                mask_surrogate_scale=0.0,
                mask_floor=0.0,
                mask_transition_width=None,
                model_config=linear_stack_config(3, 1),
            )
        ).double()
        weights = torch.ones(3, 4, dtype=torch.float64)
        keep_fraction = torch.tensor([[0.25], [0.75]], dtype=torch.float64)
        expected = torch.tensor(
            [
                [
                    [0.875, 0.375, 0.0, 0.0],
                    [0.375, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                [
                    [1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 0.625],
                    [1.0, 1.0, 0.625, 0.125],
                ],
            ],
            dtype=torch.float64,
        )

        with torch.device("meta"):
            actual = model._DiagonalAxisMask__compute_diagonal_scores(
                weights,
                keep_fraction,
            )

        self.assertEqual(actual.device, weights.device)
        self.assertEqual(actual.dtype, keep_fraction.dtype)
        self.assertTrue(torch.equal(actual, expected))
        self.assertFalse(torch.equal(actual[0], actual[1]))


if __name__ == "__main__":
    unittest.main()

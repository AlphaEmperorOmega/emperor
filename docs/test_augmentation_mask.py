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
    AxisMaskOptions,
    MaskDimensionOptions,
)
from emperor.augmentations.adaptive_parameters.core.mask import (
    AxisMaskAbstract,
    AxisMaskConfig,
    DiagonalAxisMask,
    PerAxisScoreMask,
    TopSliceAxisMask,
    WeightInformedScoreAxisMask,
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


class TestAxisMaskHandlers(unittest.TestCase):
    def preset(
        self,
        input_dim: int = 4,
        hidden_dim: int = 8,
        output_dim: int = 3,
        bias_flag: bool = True,
        model_type: AxisMaskOptions = AxisMaskOptions.WEIGHT_INFORMED_SCORE,
        mask_dimension_option: MaskDimensionOptions = MaskDimensionOptions.ROW,
        mask_threshold: float = 0.5,
        mask_surrogate_scale: float = 5.0,
        mask_floor: float = 0.0,
    ) -> AxisMaskConfig:
        return AxisMaskConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            model_type=model_type,
            mask_dimension_option=mask_dimension_option,
            mask_threshold=mask_threshold,
            mask_surrogate_scale=mask_surrogate_scale,
            mask_floor=mask_floor,
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

    def _axis_hybrid_mask(
        self,
        scores: torch.Tensor,
        threshold: float,
        scale: float,
        floor: float = 0.0,
    ) -> torch.Tensor:
        hard_mask = (scores >= threshold).float()
        soft_mask = torch.sigmoid(scale * (scores - threshold))
        adjusted_hard_mask = (floor + (1.0 - floor) * hard_mask).clamp(
            min=floor, max=1.0
        )
        return adjusted_hard_mask * soft_mask

    def _apply_axis_mask(
        self,
        weight_params: torch.Tensor,
        axis_mask: torch.Tensor,
        mask_dimension_option: MaskDimensionOptions,
    ) -> torch.Tensor:
        broadcast_dim = (
            -2
            if mask_dimension_option == MaskDimensionOptions.COLUMN
            else -1
        )
        return weight_params * axis_mask.unsqueeze(broadcast_dim)

    def _expected_global_output(
        self,
        weight_params: torch.Tensor,
        mask_logits: torch.Tensor,
        mask_dimension_option: MaskDimensionOptions,
        threshold: float,
        scale: float,
        floor: float = 0.0,
    ) -> torch.Tensor:
        source_axis_soft_mask = torch.sigmoid(mask_logits)
        source_broadcast_dim = (
            -1
            if mask_dimension_option == MaskDimensionOptions.COLUMN
            else -2
        )
        score_dim = -2 if mask_dimension_option == MaskDimensionOptions.COLUMN else -1
        masked_weights = weight_params * source_axis_soft_mask.unsqueeze(
            source_broadcast_dim
        )
        axis_scores = masked_weights.abs().mean(dim=score_dim)
        axis_mask = self._axis_hybrid_mask(axis_scores, threshold, scale, floor)
        return self._apply_axis_mask(weight_params, axis_mask, mask_dimension_option)

    def _expected_per_axis_output(
        self,
        weight_params: torch.Tensor,
        mask_logits: torch.Tensor,
        mask_dimension_option: MaskDimensionOptions,
        threshold: float,
        scale: float,
        floor: float = 0.0,
    ) -> torch.Tensor:
        axis_scores = torch.sigmoid(mask_logits)
        axis_mask = self._axis_hybrid_mask(axis_scores, threshold, scale, floor)
        return self._apply_axis_mask(weight_params, axis_mask, mask_dimension_option)

    def _expected_top_slice_output(
        self,
        weight_params: torch.Tensor,
        mask_logits: torch.Tensor,
        mask_dimension_option: MaskDimensionOptions,
        threshold: float,
        scale: float,
        floor: float = 0.0,
    ) -> torch.Tensor:
        axis_scores = torch.sigmoid(mask_logits)
        hard_prefix = (axis_scores >= threshold).float().cumprod(dim=-1)
        soft_mask = torch.sigmoid(scale * (axis_scores - threshold))
        adjusted_hard_prefix = floor + (1.0 - floor) * hard_prefix
        axis_mask = adjusted_hard_prefix * soft_mask
        return self._apply_axis_mask(weight_params, axis_mask, mask_dimension_option)

    def _expected_diagonal_output(
        self,
        weight_params: torch.Tensor,
        mask_logits: torch.Tensor,
        threshold: float,
        scale: float,
        floor: float = 0.0,
    ) -> torch.Tensor:
        row_count = weight_params.shape[-2]
        col_count = weight_params.shape[-1]
        keep_fraction = torch.sigmoid(mask_logits)
        min_diagonal_shift = 1 - row_count
        diagonal_shift = (
            keep_fraction.squeeze(-1) * (row_count + col_count) - row_count
        ).clamp(min=float(min_diagonal_shift))
        row_indices = torch.arange(row_count, dtype=keep_fraction.dtype)
        col_indices = torch.arange(col_count, dtype=keep_fraction.dtype)
        boundary = (row_count - 1 - row_indices).unsqueeze(0).unsqueeze(
            -1
        ) + diagonal_shift[:, None, None]
        margins = boundary - col_indices.unsqueeze(0).unsqueeze(0)
        diagonal_scores = ((margins + 1.0) / 2.0).clamp(0.0, 1.0)
        hard_mask = (diagonal_scores >= threshold).float()
        soft_mask = torch.sigmoid(scale * (diagonal_scores - threshold))
        adjusted_hard_mask = (floor + (1.0 - floor) * hard_mask).clamp(
            min=floor, max=1.0
        )
        return weight_params * (adjusted_hard_mask * soft_mask)

    def test_global_score_uses_independent_hybrid_row_and_column_masks(self):
        row_cfg = self.preset(
            input_dim=4,
            output_dim=3,
            model_type=AxisMaskOptions.WEIGHT_INFORMED_SCORE,
            mask_dimension_option=MaskDimensionOptions.ROW,
        )
        row_model = WeightInformedScoreAxisMask(row_cfg)
        row_model.eval()
        row_logits = torch.zeros(2, 4)
        row_weights = torch.tensor(
            [
                [[2.0, 2.0, 2.0], [0.1, 0.1, 0.1], [3.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
                [[0.2, 0.2, 0.2], [2.0, 2.0, 2.0], [0.4, 0.4, 0.4], [2.0, 0.0, 0.0]],
            ]
        )
        row_mask_logits = torch.tensor(
            [[10.0, -10.0, 10.0], [-10.0, 10.0, -10.0]]
        )
        row_model.score_generator = ConstantGenerator(row_mask_logits)

        row_output = row_model(row_weights, row_logits)
        row_expected = self._expected_global_output(
            row_weights,
            row_mask_logits,
            MaskDimensionOptions.ROW,
            row_cfg.mask_threshold,
            row_cfg.mask_surrogate_scale,
        )

        self.assertTrue(torch.allclose(row_output, row_expected, atol=1e-6))
        self.assertTrue(torch.allclose(row_output[0, 1], torch.zeros_like(row_output[0, 1])))
        self.assertTrue(torch.allclose(row_output[1, 0], torch.zeros_like(row_output[1, 0])))

        column_cfg = self.preset(
            input_dim=3,
            output_dim=5,
            model_type=AxisMaskOptions.WEIGHT_INFORMED_SCORE,
            mask_dimension_option=MaskDimensionOptions.COLUMN,
        )
        column_model = WeightInformedScoreAxisMask(column_cfg)
        column_model.eval()
        column_logits = torch.zeros(2, 3)
        column_weights = torch.tensor(
            [
                [
                    [2.0, 0.2, 2.0, 0.2, 2.0],
                    [2.0, 0.2, 2.0, 0.2, 2.0],
                    [2.0, 0.2, 2.0, 0.2, 2.0],
                ],
                [
                    [0.2, 2.0, 0.2, 2.0, 0.2],
                    [0.2, 2.0, 0.2, 2.0, 0.2],
                    [0.2, 2.0, 0.2, 2.0, 0.2],
                ],
            ]
        )
        column_mask_logits = torch.tensor(
            [[10.0, -10.0, 10.0], [-10.0, 10.0, -10.0]]
        )
        column_model.score_generator = ConstantGenerator(column_mask_logits)

        column_output = column_model(column_weights, column_logits)
        column_expected = self._expected_global_output(
            column_weights,
            column_mask_logits,
            MaskDimensionOptions.COLUMN,
            column_cfg.mask_threshold,
            column_cfg.mask_surrogate_scale,
        )

        self.assertTrue(torch.allclose(column_output, column_expected, atol=1e-6))
        self.assertTrue(
            torch.allclose(column_output[0, :, 1], torch.zeros_like(column_output[0, :, 1]))
        )
        self.assertTrue(
            torch.allclose(column_output[1, :, 0], torch.zeros_like(column_output[1, :, 0]))
        )

    def test_per_axis_score_applies_direct_hybrid_axis_masks_in_row_and_column_modes(self):
        row_cfg = self.preset(
            input_dim=4,
            output_dim=3,
            model_type=AxisMaskOptions.PER_AXIS_SCORE,
            mask_dimension_option=MaskDimensionOptions.ROW,
        )
        row_model = PerAxisScoreMask(row_cfg)
        row_model.eval()
        row_logits = torch.zeros(2, 4)
        row_weights = torch.tensor(
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [1.0, 1.0, 1.0]],
                [[-1.0, -2.0, -3.0], [2.0, 3.0, 4.0], [5.0, 6.0, 7.0], [8.0, 9.0, 1.0]],
            ]
        )
        row_mask_logits = torch.tensor(
            [[10.0, -10.0, 10.0, -10.0], [-10.0, 10.0, -10.0, 10.0]]
        )
        row_model.score_generator = ConstantGenerator(row_mask_logits)

        row_output = row_model(row_weights, row_logits)
        row_expected = self._expected_per_axis_output(
            row_weights,
            row_mask_logits,
            MaskDimensionOptions.ROW,
            row_cfg.mask_threshold,
            row_cfg.mask_surrogate_scale,
        )

        self.assertTrue(torch.allclose(row_output, row_expected, atol=1e-6))
        self.assertTrue(torch.allclose(row_output[0, 1], torch.zeros_like(row_output[0, 1])))
        self.assertTrue(torch.allclose(row_output[1, 0], torch.zeros_like(row_output[1, 0])))

        column_cfg = self.preset(
            input_dim=3,
            output_dim=5,
            model_type=AxisMaskOptions.PER_AXIS_SCORE,
            mask_dimension_option=MaskDimensionOptions.COLUMN,
        )
        column_model = PerAxisScoreMask(column_cfg)
        column_model.eval()
        column_logits = torch.zeros(2, 3)
        column_weights = torch.tensor(
            [
                [
                    [1.0, 2.0, 3.0, 4.0, 5.0],
                    [6.0, 7.0, 8.0, 9.0, 1.0],
                    [2.0, 3.0, 4.0, 5.0, 6.0],
                ],
                [
                    [-1.0, -2.0, -3.0, -4.0, -5.0],
                    [5.0, 4.0, 3.0, 2.0, 1.0],
                    [9.0, 8.0, 7.0, 6.0, 5.0],
                ],
            ]
        )
        column_mask_logits = torch.tensor(
            [[10.0, -10.0, 10.0, -10.0, 10.0], [-10.0, 10.0, -10.0, 10.0, -10.0]]
        )
        column_model.score_generator = ConstantGenerator(column_mask_logits)

        column_output = column_model(column_weights, column_logits)
        column_expected = self._expected_per_axis_output(
            column_weights,
            column_mask_logits,
            MaskDimensionOptions.COLUMN,
            column_cfg.mask_threshold,
            column_cfg.mask_surrogate_scale,
        )

        self.assertTrue(torch.allclose(column_output, column_expected, atol=1e-6))
        self.assertTrue(
            torch.allclose(column_output[0, :, 1], torch.zeros_like(column_output[0, :, 1]))
        )
        self.assertTrue(
            torch.allclose(column_output[1, :, 0], torch.zeros_like(column_output[1, :, 0]))
        )

    def test_top_slice_row_mode_zeroes_first_below_threshold_row_and_all_later_rows(self):
        cfg = self.preset(
            input_dim=4,
            output_dim=3,
            model_type=AxisMaskOptions.TOP_SLICE,
            mask_dimension_option=MaskDimensionOptions.ROW,
        )
        model = TopSliceAxisMask(cfg)
        model.eval()
        logits = torch.zeros(2, 4)
        weight_params = torch.tensor(
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [1.0, 1.0, 1.0]],
                [[2.0, 2.0, 2.0], [3.0, 3.0, 3.0], [4.0, 4.0, 4.0], [5.0, 5.0, 5.0]],
            ]
        )
        mask_logits = torch.tensor(
            [[10.0, 10.0, -10.0, 10.0], [10.0, -10.0, 10.0, 10.0]]
        )
        model.score_generator = ConstantGenerator(mask_logits)

        output = model(weight_params, logits)
        expected = self._expected_top_slice_output(
            weight_params,
            mask_logits,
            MaskDimensionOptions.ROW,
            cfg.mask_threshold,
            cfg.mask_surrogate_scale,
        )

        self.assertTrue(torch.allclose(output, expected, atol=1e-6))
        self.assertTrue(torch.allclose(output[0, 2], torch.zeros_like(output[0, 2])))
        self.assertTrue(torch.allclose(output[0, 3], torch.zeros_like(output[0, 3])))
        self.assertTrue(torch.allclose(output[1, 1], torch.zeros_like(output[1, 1])))
        self.assertTrue(torch.allclose(output[1, 2], torch.zeros_like(output[1, 2])))

    def test_top_slice_column_mode_zeroes_first_below_threshold_column_and_all_later_columns(self):
        cfg = self.preset(
            input_dim=3,
            output_dim=5,
            model_type=AxisMaskOptions.TOP_SLICE,
            mask_dimension_option=MaskDimensionOptions.COLUMN,
        )
        model = TopSliceAxisMask(cfg)
        model.eval()
        logits = torch.zeros(2, 3)
        weight_params = torch.tensor(
            [
                [
                    [1.0, 2.0, 3.0, 4.0, 5.0],
                    [6.0, 7.0, 8.0, 9.0, 1.0],
                    [2.0, 3.0, 4.0, 5.0, 6.0],
                ],
                [
                    [-1.0, -2.0, -3.0, -4.0, -5.0],
                    [5.0, 4.0, 3.0, 2.0, 1.0],
                    [9.0, 8.0, 7.0, 6.0, 5.0],
                ],
            ]
        )
        mask_logits = torch.tensor(
            [[10.0, 10.0, -10.0, 10.0, 10.0], [10.0, -10.0, 10.0, 10.0, 10.0]]
        )
        model.score_generator = ConstantGenerator(mask_logits)

        output = model(weight_params, logits)
        expected = self._expected_top_slice_output(
            weight_params,
            mask_logits,
            MaskDimensionOptions.COLUMN,
            cfg.mask_threshold,
            cfg.mask_surrogate_scale,
        )

        self.assertTrue(torch.allclose(output, expected, atol=1e-6))
        self.assertTrue(
            torch.allclose(output[0, :, 2], torch.zeros_like(output[0, :, 2]))
        )
        self.assertTrue(
            torch.allclose(output[0, :, 3], torch.zeros_like(output[0, :, 3]))
        )
        self.assertTrue(
            torch.allclose(output[1, :, 1], torch.zeros_like(output[1, :, 1]))
        )
        self.assertTrue(
            torch.allclose(output[1, :, 4], torch.zeros_like(output[1, :, 4]))
        )

    def test_diagonal_mask_matches_geometric_boundary(self):
        cfg = self.preset(
            input_dim=4,
            output_dim=4,
            model_type=AxisMaskOptions.DIAGONAL,
        )
        model = DiagonalAxisMask(cfg)
        model.eval()
        logits = torch.zeros(2, 4)
        weight_params = torch.ones(2, 4, 4)
        mask_logits = torch.tensor([[2.0], [-2.0]])
        model.score_generator = ConstantGenerator(mask_logits)

        output = model(weight_params, logits)
        expected = self._expected_diagonal_output(
            weight_params,
            mask_logits,
            cfg.mask_threshold,
            cfg.mask_surrogate_scale,
        )

        self.assertTrue(torch.allclose(output, expected, atol=1e-6))
        high_keep_nonzero = (output[0].abs() > 1e-6).sum().item()
        low_keep_nonzero = (output[1].abs() > 1e-6).sum().item()
        self.assertGreater(high_keep_nonzero, low_keep_nonzero)
        for row in range(3):
            kept_this_row = (output[0, row].abs() > 1e-6).sum().item()
            kept_next_row = (output[0, row + 1].abs() > 1e-6).sum().item()
            self.assertGreaterEqual(kept_this_row, kept_next_row)

    def test_diagonal_mask_boundary_slides_with_keep_fraction(self):
        cfg = self.preset(
            input_dim=4,
            output_dim=4,
            model_type=AxisMaskOptions.DIAGONAL,
            mask_threshold=0.5,
        )
        weight_params = torch.ones(3, 4, 4)
        logits = torch.zeros(3, 4)

        high_logits = torch.tensor([[10.0], [-10.0], [0.0]])
        model = DiagonalAxisMask(cfg)
        model.eval()
        model.score_generator = ConstantGenerator(high_logits)

        output = model(weight_params, logits)

        high_kept = (output[0].abs() > 1e-6).sum().item()
        low_kept = (output[1].abs() > 1e-6).sum().item()
        mid_kept = (output[2].abs() > 1e-6).sum().item()

        self.assertGreater(high_kept, mid_kept)
        self.assertGreater(mid_kept, low_kept)

        expected = self._expected_diagonal_output(
            weight_params,
            high_logits,
            cfg.mask_threshold,
            cfg.mask_surrogate_scale,
        )
        self.assertTrue(torch.allclose(output, expected, atol=1e-6))

    def test_diagonal_mask_batch_semantics(self):
        cfg = self.preset(
            input_dim=3,
            output_dim=4,
            model_type=AxisMaskOptions.DIAGONAL,
        )
        model = DiagonalAxisMask(cfg)
        model.eval()
        logits = torch.zeros(2, 3)
        weight_params = torch.ones(2, 3, 4)
        mask_logits = torch.tensor([[3.0], [-3.0]])
        model.score_generator = ConstantGenerator(mask_logits)

        output = model(weight_params, logits)

        self.assertFalse(torch.allclose(output[0], output[1], atol=1e-6))
        kept_0 = (output[0].abs() > 1e-6).sum().item()
        kept_1 = (output[1].abs() > 1e-6).sum().item()
        self.assertGreater(kept_0, kept_1)

    def test_diagonal_mask_floor_attenuates_dropped_region(self):
        cfg = self.preset(
            input_dim=4,
            output_dim=4,
            model_type=AxisMaskOptions.DIAGONAL,
            mask_floor=0.2,
        )
        model = DiagonalAxisMask(cfg)
        model.eval()
        logits = torch.zeros(1, 4)
        weight_params = torch.ones(1, 4, 4)
        mask_logits = torch.tensor([[0.0]])
        model.score_generator = ConstantGenerator(mask_logits)

        output = model(weight_params, logits)
        expected = self._expected_diagonal_output(
            weight_params,
            mask_logits,
            cfg.mask_threshold,
            cfg.mask_surrogate_scale,
            cfg.mask_floor,
        )

        self.assertTrue(torch.allclose(output, expected, atol=1e-6))

        output_no_floor = self._expected_diagonal_output(
            weight_params,
            mask_logits,
            cfg.mask_threshold,
            cfg.mask_surrogate_scale,
            0.0,
        )
        zero_in_no_floor = (output_no_floor.abs() < 1e-6)
        nonzero_in_floor = (output[zero_in_no_floor].abs() > 1e-7)
        if zero_in_no_floor.any():
            self.assertTrue(nonzero_in_floor.any())

    def test_mask_options_produce_distinct_outputs_for_crafted_inputs(self):
        logits = torch.zeros(1, 4)
        weight_params = torch.tensor(
            [[[2.0, 2.0, 2.0], [0.1, 0.1, 0.1], [3.0, 0.0, 0.0], [1.0, 1.0, 1.0]]]
        )

        global_cfg = self.preset(
            input_dim=4,
            output_dim=3,
            model_type=AxisMaskOptions.WEIGHT_INFORMED_SCORE,
            mask_dimension_option=MaskDimensionOptions.ROW,
        )
        global_model = WeightInformedScoreAxisMask(global_cfg)
        global_model.eval()
        global_model.score_generator = ConstantGenerator(torch.tensor([[10.0, -10.0, 10.0]]))
        global_output = global_model(weight_params, logits)

        per_axis_cfg = self.preset(
            input_dim=4,
            output_dim=3,
            model_type=AxisMaskOptions.PER_AXIS_SCORE,
            mask_dimension_option=MaskDimensionOptions.ROW,
        )
        per_axis_model = PerAxisScoreMask(per_axis_cfg)
        per_axis_model.eval()
        per_axis_model.score_generator = ConstantGenerator(
            torch.tensor([[10.0, -10.0, 10.0, -10.0]])
        )
        per_axis_output = per_axis_model(weight_params, logits)

        top_slice_cfg = self.preset(
            input_dim=4,
            output_dim=3,
            model_type=AxisMaskOptions.TOP_SLICE,
            mask_dimension_option=MaskDimensionOptions.ROW,
        )
        top_slice_model = TopSliceAxisMask(top_slice_cfg)
        top_slice_model.eval()
        top_slice_model.score_generator = ConstantGenerator(
            torch.tensor([[10.0, 10.0, -10.0, 10.0]])
        )
        top_slice_output = top_slice_model(weight_params, logits)

        diagonal_cfg = self.preset(
            input_dim=4,
            output_dim=3,
            model_type=AxisMaskOptions.DIAGONAL,
            mask_dimension_option=MaskDimensionOptions.ROW,
        )
        diagonal_model = DiagonalAxisMask(diagonal_cfg)
        diagonal_model.eval()
        diagonal_model.score_generator = ConstantGenerator(
            torch.tensor([[0.0]])
        )
        diagonal_output = diagonal_model(weight_params, logits)

        outputs = [global_output, per_axis_output, top_slice_output, diagonal_output]
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                with self.subTest(i=i, j=j):
                    self.assertFalse(torch.allclose(outputs[i], outputs[j], atol=1e-6))

    def test_batch_semantics_allow_different_sample_masks_for_global_score(self):
        cfg = self.preset(
            input_dim=4,
            output_dim=3,
            model_type=AxisMaskOptions.WEIGHT_INFORMED_SCORE,
            mask_dimension_option=MaskDimensionOptions.ROW,
        )
        model = WeightInformedScoreAxisMask(cfg)
        model.eval()
        logits = torch.zeros(2, 4)
        weight_params = torch.tensor(
            [
                [[3.0, 3.0, 0.1], [0.2, 0.2, 0.2], [2.0, 2.0, 0.1], [0.1, 0.1, 0.1]],
                [[0.1, 3.0, 3.0], [0.2, 0.2, 0.2], [0.1, 2.0, 2.0], [0.1, 0.1, 0.1]],
            ]
        )
        mask_logits = torch.tensor([[10.0, 10.0, -10.0], [-10.0, 10.0, 10.0]])
        model.score_generator = ConstantGenerator(mask_logits)

        output = model(weight_params, logits)

        self.assertFalse(torch.allclose(output[0], output[1], atol=1e-6))
        self.assertTrue(torch.allclose(output[0, 1], torch.zeros_like(output[0, 1])))
        self.assertTrue(torch.allclose(output[1, 1], torch.zeros_like(output[1, 1])))

    def test_batch_semantics_allow_different_sample_masks_for_per_axis_score(self):
        cfg = self.preset(
            input_dim=4,
            output_dim=3,
            model_type=AxisMaskOptions.PER_AXIS_SCORE,
            mask_dimension_option=MaskDimensionOptions.ROW,
        )
        model = PerAxisScoreMask(cfg)
        model.eval()
        logits = torch.zeros(2, 4)
        weight_params = torch.tensor(
            [
                [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0], [4.0, 4.0, 4.0]],
                [[5.0, 5.0, 5.0], [6.0, 6.0, 6.0], [7.0, 7.0, 7.0], [8.0, 8.0, 8.0]],
            ]
        )
        mask_logits = torch.tensor(
            [[10.0, -10.0, 10.0, -10.0], [-10.0, 10.0, -10.0, 10.0]]
        )
        model.score_generator = ConstantGenerator(mask_logits)

        output = model(weight_params, logits)

        self.assertFalse(torch.allclose(output[0], output[1], atol=1e-6))
        self.assertTrue(torch.allclose(output[0, 1], torch.zeros_like(output[0, 1])))
        self.assertTrue(torch.allclose(output[1, 0], torch.zeros_like(output[1, 0])))

    def test_build_creates_model_for_each_option(self):
        for option in AxisMaskOptions:
            with self.subTest(option=option):
                cfg = self.preset(model_type=option)
                if option == AxisMaskOptions.DISABLED:
                    with self.assertRaises(ValueError):
                        cfg.build()
                else:
                    model = cfg.build()
                    self.assertIsInstance(model, AxisMaskAbstract)

    def test_enum_uses_per_axis_score_name(self):
        self.assertTrue(hasattr(AxisMaskOptions, "PER_AXIS_SCORE"))
        legacy_name = "PER_" + "ROW_SCORE"
        self.assertFalse(hasattr(AxisMaskOptions, legacy_name))

    def test_registry_maps_each_option_to_distinct_mask_class(self):
        for option, model_cls in (
            (AxisMaskOptions.WEIGHT_INFORMED_SCORE, WeightInformedScoreAxisMask),
            (AxisMaskOptions.PER_AXIS_SCORE, PerAxisScoreMask),
            (AxisMaskOptions.TOP_SLICE, TopSliceAxisMask),
            (AxisMaskOptions.DIAGONAL, DiagonalAxisMask),
        ):
            with self.subTest(option=option):
                cfg = self.preset(model_type=option)
                self.assertIsInstance(cfg.build(), model_cls)

    def test_invalid_mask_threshold_raises(self):
        cfg = self.preset(mask_threshold=1.5)

        with self.assertRaises(ValueError):
            cfg.build()

    def test_invalid_mask_surrogate_scale_raises(self):
        cfg = self.preset(mask_surrogate_scale=0.0)

        with self.assertRaises(ValueError):
            cfg.build()

    def test_invalid_mask_floor_raises(self):
        for invalid_mask_floor in (-0.1, 1.0):
            with self.subTest(mask_floor=invalid_mask_floor):
                cfg = self.preset(mask_floor=invalid_mask_floor)

                with self.assertRaises(ValueError):
                    cfg.build()

    def test_mask_floor_is_applied_inside_the_hard_mask(self):
        cfg = self.preset(
            input_dim=4,
            output_dim=3,
            model_type=AxisMaskOptions.PER_AXIS_SCORE,
            mask_dimension_option=MaskDimensionOptions.ROW,
            mask_floor=0.2,
        )
        model = PerAxisScoreMask(cfg)
        model.eval()
        logits = torch.zeros(1, 4)
        weight_params = torch.tensor(
            [[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0], [4.0, 4.0, 4.0]]]
        )
        mask_logits = torch.tensor([[10.0, -10.0, 10.0, -10.0]])
        model.score_generator = ConstantGenerator(mask_logits)

        output = model(weight_params, logits)
        expected = self._expected_per_axis_output(
            weight_params,
            mask_logits,
            MaskDimensionOptions.ROW,
            cfg.mask_threshold,
            cfg.mask_surrogate_scale,
            cfg.mask_floor,
        )

        self.assertTrue(torch.allclose(output, expected, atol=1e-6))
        dropped_axis_score = torch.sigmoid(mask_logits[0, 1])
        dropped_row_scale = cfg.mask_floor * torch.sigmoid(
            cfg.mask_surrogate_scale * (dropped_axis_score - cfg.mask_threshold)
        )
        self.assertTrue(
            torch.allclose(
                output[0, 1],
                weight_params[0, 1] * dropped_row_scale,
                atol=1e-6,
            )
        )

    def test_invalid_dimensions_raise(self):
        cfg = AxisMaskConfig(
            input_dim=0,
            output_dim=3,
            model_type=AxisMaskOptions.WEIGHT_INFORMED_SCORE,
            mask_dimension_option=MaskDimensionOptions.ROW,
            model_config=self.preset().model_config,
            mask_threshold=0.5,
            mask_surrogate_scale=5.0,
            mask_floor=0.0,
        )

        with self.assertRaises(ValueError):
            cfg.build()

    def test_validate_generator_model_raises_on_unknown_generator_type(self):
        class InvalidGeneratorConfig:
            def build(self, overrides):
                return nn.Identity()

        cfg = self.preset(model_type=AxisMaskOptions.WEIGHT_INFORMED_SCORE)
        model = WeightInformedScoreAxisMask(cfg)
        model.model_config = InvalidGeneratorConfig()

        with self.assertRaises(TypeError):
            model._init_generator(1)

    def test_gradients_flow_to_generator_parameters_for_all_enabled_options(self):
        batch_size = 2
        input_dim = 4
        output_dim = 3
        torch.manual_seed(0)

        for option in AxisMaskOptions:
            with self.subTest(option=option):
                cfg = self.preset(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    model_type=option,
                    mask_threshold=0.2,
                    mask_surrogate_scale=5.0,
                )
                if option == AxisMaskOptions.DISABLED:
                    with self.assertRaises(ValueError):
                        cfg.build()
                    continue

                model = cfg.build()
                logits = torch.randn(batch_size, input_dim)
                weight_params = torch.randn(
                    batch_size, input_dim, output_dim, requires_grad=True
                )
                loss = model(weight_params, logits).pow(2).sum()
                loss.backward()

                grads = [param.grad for param in model.parameters() if param.requires_grad]
                nonzero_grads = [
                    grad for grad in grads if grad is not None and grad.abs().sum() > 0
                ]
                self.assertTrue(len(nonzero_grads) > 0)
                self.assertIsNotNone(weight_params.grad)

    def test_option_matrix_forward_shapes(self):
        batch_size = 2
        input_dims = [4, 3]
        output_dims = [3, 5]

        for option in AxisMaskOptions:
            if option == AxisMaskOptions.DISABLED:
                continue
            for mask_dimension_option in MaskDimensionOptions:
                for input_dim, output_dim in zip(input_dims, output_dims):
                    with self.subTest(
                        option=option,
                        mask_dimension_option=mask_dimension_option,
                        input_dim=input_dim,
                        output_dim=output_dim,
                    ):
                        cfg = self.preset(
                            input_dim=input_dim,
                            output_dim=output_dim,
                            model_type=option,
                            mask_dimension_option=mask_dimension_option,
                        )
                        model = cfg.build()
                        logits = torch.randn(batch_size, input_dim)
                        weight_params = torch.randn(batch_size, input_dim, output_dim)
                        output = model(weight_params, logits)
                        self.assertEqual(
                            output.shape, (batch_size, input_dim, output_dim)
                        )

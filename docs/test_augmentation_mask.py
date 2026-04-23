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
    MaskDimensionOptions,
    RowMaskOptions,
)
from emperor.augmentations.adaptive_parameters.core.mask import (
    DiagonalRowMask,
    GlobalScoreRowMask,
    PerRowScoreRowMask,
    RowMaskAbstract,
    RowMaskConfig,
    TopSliceRowMask,
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


class TestRowMaskHandlers(unittest.TestCase):
    def preset(
        self,
        input_dim: int = 4,
        hidden_dim: int = 8,
        output_dim: int = 3,
        bias_flag: bool = True,
        model_type: RowMaskOptions = RowMaskOptions.GLOBAL_SCORE,
        mask_dimension_option: MaskDimensionOptions = MaskDimensionOptions.ROW,
    ) -> RowMaskConfig:
        return RowMaskConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            model_type=model_type,
            mask_dimension_option=mask_dimension_option,
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

    def test_global_score_row_mask_forward(self):
        batch_size = 2
        input_dim = 4
        output_dim = 3
        cfg = self.preset(
            input_dim=input_dim,
            output_dim=output_dim,
            model_type=RowMaskOptions.GLOBAL_SCORE,
        )
        model = GlobalScoreRowMask(cfg)
        logits = torch.randn(batch_size, input_dim)
        weight_params = torch.randn(batch_size, input_dim, output_dim)
        output = model(weight_params, logits)

        self.assertEqual(output.shape, (batch_size, input_dim, output_dim))
        self.assertIsInstance(output, torch.Tensor)

    def test_per_row_score_row_mask_forward(self):
        batch_size = 2
        input_dims = [4, 3]
        output_dims = [3, 5]

        for mask_dimension_option in MaskDimensionOptions:
            for input_dim, output_dim in zip(input_dims, output_dims):
                with self.subTest(
                    mask_dimension_option=mask_dimension_option,
                    input_dim=input_dim,
                    output_dim=output_dim,
                ):
                    cfg = self.preset(
                        input_dim=input_dim,
                        output_dim=output_dim,
                        model_type=RowMaskOptions.PER_ROW_SCORE,
                        mask_dimension_option=mask_dimension_option,
                    )
                    model = PerRowScoreRowMask(cfg)
                    logits = torch.randn(batch_size, input_dim)
                    weight_params = torch.randn(batch_size, input_dim, output_dim)
                    output = model(weight_params, logits)

                    self.assertEqual(output.shape, (batch_size, input_dim, output_dim))
                    self.assertIsInstance(output, torch.Tensor)

    def test_top_slice_row_mask_forward(self):
        batch_size = 2
        input_dim = 4
        output_dim = 3
        cfg = self.preset(
            input_dim=input_dim,
            output_dim=output_dim,
            model_type=RowMaskOptions.TOP_SLICE,
        )
        model = TopSliceRowMask(cfg)
        logits = torch.randn(batch_size, input_dim)
        weight_params = torch.randn(batch_size, input_dim, output_dim)
        output = model(weight_params, logits)

        self.assertEqual(output.shape, (batch_size, input_dim, output_dim))
        self.assertIsInstance(output, torch.Tensor)

    def test_diagonal_row_mask_forward(self):
        batch_size = 2
        input_dim = 4
        output_dim = 3
        cfg = self.preset(
            input_dim=input_dim,
            output_dim=output_dim,
            model_type=RowMaskOptions.DIAGONAL,
        )
        model = DiagonalRowMask(cfg)
        logits = torch.randn(batch_size, input_dim)
        weight_params = torch.randn(batch_size, input_dim, output_dim)
        output = model(weight_params, logits)

        self.assertEqual(output.shape, (batch_size, input_dim, output_dim))
        self.assertIsInstance(output, torch.Tensor)

    def test_global_score_row_mask_keeps_highest_magnitude_rows(self):
        cfg = self.preset(
            input_dim=4,
            output_dim=3,
            model_type=RowMaskOptions.GLOBAL_SCORE,
            mask_dimension_option=MaskDimensionOptions.ROW,
        )
        model = GlobalScoreRowMask(cfg)
        model.eval()
        logits = torch.zeros(2, 4)
        weight_params = torch.tensor(
            [
                [[5.0, 0.0, 0.0], [0.0, 4.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.5, 0.0]],
                [[0.0, 3.0, 0.0], [0.0, 0.1, 0.0], [2.0, 0.0, 0.0], [0.0, 5.0, 0.0]],
            ]
        )
        model.score_generator = ConstantGenerator(torch.zeros(2, 1))

        output = model(weight_params, logits)
        expected = torch.tensor(
            [
                [[5.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 3.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 5.0, 0.0]],
            ]
        )

        self.assertTrue(torch.allclose(output, expected, atol=1e-6))

    def test_per_row_score_row_mask_applies_row_mask(self):
        cfg = self.preset(
            input_dim=4,
            output_dim=3,
            model_type=RowMaskOptions.PER_ROW_SCORE,
            mask_dimension_option=MaskDimensionOptions.ROW,
        )
        model = PerRowScoreRowMask(cfg)
        model.eval()
        logits = torch.zeros(2, 4)
        weight_params = torch.tensor(
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [1.0, 1.0, 1.0]],
                [[-1.0, -2.0, -3.0], [2.0, 3.0, 4.0], [5.0, 6.0, 7.0], [8.0, 9.0, 1.0]],
            ]
        )
        keep_fraction_logits = torch.tensor(
            [[10.0, -10.0, 0.0, -1.0], [-10.0, 10.0, 0.0, 10.0]]
        )
        model.score_generator = ConstantGenerator(keep_fraction_logits)

        output = model(weight_params, logits)
        expected = torch.tensor(
            [
                [[1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [7.0, 8.0, 9.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [2.0, 3.0, 4.0], [5.0, 6.0, 7.0], [8.0, 9.0, 1.0]],
            ]
        )

        self.assertTrue(torch.allclose(output, expected, atol=1e-6))

    def test_per_row_score_row_mask_applies_column_mask(self):
        cfg = self.preset(
            input_dim=3,
            output_dim=5,
            model_type=RowMaskOptions.PER_ROW_SCORE,
            mask_dimension_option=MaskDimensionOptions.COLUMN,
        )
        model = PerRowScoreRowMask(cfg)
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
        keep_fraction_logits = torch.tensor(
            [[10.0, -10.0, 0.0, -1.0, 10.0], [-10.0, 10.0, 0.0, 10.0, -10.0]]
        )
        model.score_generator = ConstantGenerator(keep_fraction_logits)

        output = model(weight_params, logits)
        expected = torch.tensor(
            [
                [
                    [1.0, 0.0, 3.0, 0.0, 5.0],
                    [6.0, 0.0, 8.0, 0.0, 1.0],
                    [2.0, 0.0, 4.0, 0.0, 6.0],
                ],
                [
                    [0.0, -2.0, -3.0, -4.0, 0.0],
                    [0.0, 4.0, 3.0, 2.0, 0.0],
                    [0.0, 8.0, 7.0, 6.0, 0.0],
                ],
            ]
        )

        self.assertTrue(torch.allclose(output, expected, atol=1e-6))

    def test_top_slice_row_mask_keeps_leading_rows(self):
        cfg = self.preset(
            input_dim=4,
            output_dim=3,
            model_type=RowMaskOptions.TOP_SLICE,
            mask_dimension_option=MaskDimensionOptions.ROW,
        )
        model = TopSliceRowMask(cfg)
        model.eval()
        logits = torch.zeros(2, 4)
        weight_params = torch.tensor(
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [1.0, 1.0, 1.0]],
                [[-1.0, -2.0, -3.0], [2.0, 3.0, 4.0], [5.0, 6.0, 7.0], [8.0, 9.0, 1.0]],
            ]
        )
        model.score_generator = ConstantGenerator(torch.zeros(2, 1))

        output = model(weight_params, logits)
        expected = torch.tensor(
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[-1.0, -2.0, -3.0], [2.0, 3.0, 4.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ]
        )

        self.assertTrue(torch.allclose(output, expected, atol=1e-6))

    def test_diagonal_row_mask_applies_shifted_diagonal_mask(self):
        cfg = self.preset(
            input_dim=3,
            output_dim=4,
            model_type=RowMaskOptions.DIAGONAL,
        )
        model = DiagonalRowMask(cfg)
        model.eval()
        logits = torch.zeros(2, 3)
        weight_params = torch.tensor(
            [
                [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 1.0, 2.0, 3.0]],
                [[4.0, 3.0, 2.0, 1.0], [8.0, 7.0, 6.0, 5.0], [3.0, 2.0, 1.0, 0.0]],
            ]
        )
        model.score_generator = ConstantGenerator(torch.zeros(2, 1))

        output = model(weight_params, logits)
        expected_mask = torch.tensor(
            [
                [[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
                [[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
            ]
        )
        expected = weight_params * expected_mask

        self.assertTrue(torch.allclose(output, expected, atol=1e-6))

    def test_build_creates_model_for_each_option(self):
        for option in RowMaskOptions:
            with self.subTest(option=option):
                cfg = self.preset(model_type=option)
                if option == RowMaskOptions.DISABLED:
                    with self.assertRaises(ValueError):
                        cfg.build()
                else:
                    model = cfg.build()
                    self.assertIsInstance(model, RowMaskAbstract)

    def test_invalid_dimensions_raise(self):
        cfg = RowMaskConfig(
            input_dim=0,
            output_dim=3,
            model_type=RowMaskOptions.GLOBAL_SCORE,
            mask_dimension_option=MaskDimensionOptions.ROW,
            model_config=self.preset().model_config,
        )

        with self.assertRaises(ValueError):
            cfg.build()

    def test_validate_generator_model_raises_on_unknown_generator_type(self):
        class InvalidGeneratorConfig:
            def build(self, overrides):
                return nn.Identity()

        cfg = self.preset(model_type=RowMaskOptions.GLOBAL_SCORE)
        model = GlobalScoreRowMask(cfg)
        model.model_config = InvalidGeneratorConfig()

        with self.assertRaises(TypeError):
            model._init_generator(1)

    def test_gradients_flow(self):
        batch_size = 2
        input_dim = 4
        output_dim = 3

        for option in RowMaskOptions:
            with self.subTest(option=option):
                cfg = self.preset(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    model_type=option,
                )
                if option == RowMaskOptions.DISABLED:
                    with self.assertRaises(ValueError):
                        cfg.build()
                    continue

                model = cfg.build()
                logits = torch.zeros(batch_size, input_dim, requires_grad=True)
                weight_params = torch.randn(
                    batch_size, input_dim, output_dim, requires_grad=True
                )
                output = model(weight_params, logits)
                output.sum().backward()

                grads = [
                    param.grad for param in model.parameters() if param.requires_grad
                ]
                non_none_grads = [grad for grad in grads if grad is not None]
                self.assertTrue(len(non_none_grads) > 0)
                self.assertIsNotNone(weight_params.grad)

    def test_option_matrix_forward_shapes(self):
        batch_size = 2
        input_dims = [4, 3]
        output_dims = [3, 5]

        for option in RowMaskOptions:
            if option == RowMaskOptions.DISABLED:
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

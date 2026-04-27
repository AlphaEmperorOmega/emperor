import torch
import unittest
import torch.nn as nn

from emperor.base.utils import Module
from emperor.linears.options import LinearOptions
from emperor.base.layer.config import LayerConfig, LayerStackConfig
from emperor.base.enums import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
)
from emperor.linears.core.config import LinearLayerConfig
from emperor.augmentations.adaptive_parameters.config import (
    AdaptiveParameterAugmentationConfig,
)
from emperor.augmentations.adaptive_parameters.model import (
    AdaptiveParameterAugmentation,
)
from emperor.augmentations.adaptive_parameters.options import (
    AxisMaskOptions,
    BankExpansionFactorOptions,
    DynamicBiasOptions,
    DynamicDepthOptions,
    DynamicDiagonalOptions,
    DynamicWeightOptions,
    MaskDimensionOptions,
    WeightDecayScheduleOptions,
    WeightNormalizationOptions,
    WeightNormalizationPositionOptions,
)
from emperor.augmentations.adaptive_parameters.core.weight import (
    DynamicWeightConfig,
    DynamicWeightAbstract,
)
from emperor.augmentations.adaptive_parameters.core.diagonal import (
    DynamicDiagonalConfig,
    DynamicDiagonalAbstract,
)
from emperor.augmentations.adaptive_parameters.core.bias import (
    DynamicBiasConfig,
    DynamicBiasAbstract,
)
from emperor.augmentations.adaptive_parameters.core.mask import (
    AxisMaskConfig,
    AxisMaskAbstract,
)


class TestAdaptiveParameterAugmentation(unittest.TestCase):
    def preset(
        self,
        input_dim: int = 12,
        output_dim: int = 24,
        weight_config: DynamicWeightConfig | None = None,
        diagonal_config: DynamicDiagonalConfig | None = None,
        bias_config: DynamicBiasConfig | None = None,
        mask_config: AxisMaskConfig | None = None,
        model_config: LayerStackConfig | None = None,
    ) -> AdaptiveParameterAugmentationConfig:
        return AdaptiveParameterAugmentationConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            weight_config=weight_config,
            diagonal_config=diagonal_config,
            bias_config=bias_config,
            mask_config=mask_config,
            model_config=model_config,
        )

    def _make_weight_config(
        self,
        input_dim: int = 12,
        output_dim: int = 24,
        model_type: DynamicWeightOptions = DynamicWeightOptions.DUAL_MODEL,
        normalization_option: WeightNormalizationOptions = WeightNormalizationOptions.L2_SCALE,
        normalization_position_option: WeightNormalizationPositionOptions = WeightNormalizationPositionOptions.BEFORE_OUTER_PRODUCT,
        generator_depth: DynamicDepthOptions = DynamicDepthOptions.DEPTH_OF_TWO,
        bank_expansion_factor: BankExpansionFactorOptions | None = None,
        decay_schedule: WeightDecayScheduleOptions = WeightDecayScheduleOptions.DISABLED,
        decay_rate: float = 0.0,
        decay_warmup_batches: int = 0,
    ) -> DynamicWeightConfig:
        return DynamicWeightConfig(
            model_type=model_type,
            normalization_option=normalization_option,
            normalization_position_option=normalization_position_option,
            generator_depth=generator_depth,
            bank_expansion_factor=bank_expansion_factor,
            decay_schedule=decay_schedule,
            decay_rate=decay_rate,
            decay_warmup_batches=decay_warmup_batches,
            model_config=self._make_layer_stack_config(
                input_dim=input_dim, output_dim=output_dim
            ),
        )

    def _make_diagonal_config(
        self,
        input_dim: int = 12,
        output_dim: int = 24,
        model_type: DynamicDiagonalOptions = DynamicDiagonalOptions.DIAGONAL,
    ) -> DynamicDiagonalConfig:
        return DynamicDiagonalConfig(
            model_type=model_type,
            model_config=self._make_layer_stack_config(
                input_dim=input_dim, output_dim=output_dim
            ),
        )

    def _make_bias_config(
        self,
        input_dim: int = 12,
        output_dim: int = 24,
        model_type: DynamicBiasOptions = DynamicBiasOptions.DYNAMIC_PARAMETERS,
        bias_flag: bool = True,
        bank_expansion_factor: int | None = None,
        decay_schedule: WeightDecayScheduleOptions = WeightDecayScheduleOptions.DISABLED,
        decay_rate: float = 0.0,
        decay_warmup_batches: int = 0,
    ) -> DynamicBiasConfig:
        return DynamicBiasConfig(
            model_type=model_type,
            bias_flag=bias_flag,
            bank_expansion_factor=bank_expansion_factor,
            decay_schedule=decay_schedule,
            decay_rate=decay_rate,
            decay_warmup_batches=decay_warmup_batches,
            model_config=self._make_layer_stack_config(
                input_dim=input_dim, output_dim=output_dim
            ),
        )

    def _make_mask_config(
        self,
        input_dim: int = 12,
        output_dim: int = 24,
        model_type: AxisMaskOptions = AxisMaskOptions.WEIGHT_INFORMED_SCORE,
        mask_dimension_option: MaskDimensionOptions = MaskDimensionOptions.COLUMN,
        mask_threshold: float = 0.5,
        mask_surrogate_scale: float = 10.0,
        mask_floor: float = 0.0,
        mask_transition_width: float | None = None,
    ) -> AxisMaskConfig:
        return AxisMaskConfig(
            model_type=model_type,
            mask_dimension_option=mask_dimension_option,
            mask_threshold=mask_threshold,
            mask_surrogate_scale=mask_surrogate_scale,
            mask_floor=mask_floor,
            mask_transition_width=mask_transition_width,
            model_config=self._make_layer_stack_config(
                input_dim=input_dim, output_dim=output_dim
            ),
        )

    def _make_layer_stack_config(
        self,
        input_dim: int = 12,
        hidden_dim: int = 36,
        output_dim: int = 24,
        bias_flag: bool = True,
        num_layers: int = 1,
        activation: ActivationOptions = ActivationOptions.RELU,
        residual_flag: bool = False,
        dropout_probability: float = 0.0,
        last_layer_bias_option: LastLayerBiasOptions = LastLayerBiasOptions.DEFAULT,
        apply_output_pipeline_flag: bool = True,
    ) -> LayerStackConfig:
        return LayerStackConfig(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            last_layer_bias_option=last_layer_bias_option,
            apply_output_pipeline_flag=apply_output_pipeline_flag,
            layer_config=LayerConfig(
                input_dim=input_dim,
                output_dim=output_dim,
                activation=activation,
                layer_norm_position=LayerNormPositionOptions.DISABLED,
                residual_flag=residual_flag,
                dropout_probability=dropout_probability,
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
        )

    def _make_affine_callback(self):
        def callback(weights, bias, X):
            if weights.dim() == 3:
                output = torch.einsum("ij,ijk->ik", X, weights)
            else:
                output = torch.matmul(X, weights)
            if bias is not None:
                output = output + bias
            return output

        return callback

    def _make_weight_and_bias_params(self, input_dim: int = 12, output_dim: int = 24):
        weight_params = Module()._init_parameter_bank((input_dim, output_dim))
        bias_params = Module()._init_parameter_bank((output_dim,), nn.init.zeros_)
        return weight_params, bias_params

    def test_init_all_disabled(self):
        cfg = self.preset()
        model = AdaptiveParameterAugmentation(cfg)
        self.assertEqual(model.input_dim, 12)
        self.assertEqual(model.output_dim, 24)
        self.assertIsNone(model.weight_config)
        self.assertIsNone(model.diagonal_config)
        self.assertIsNone(model.bias_config)
        self.assertIsNone(model.mask_config)
        self.assertIsNone(model.model_config)

    def test_init_with_weight_config(self):
        for option in DynamicWeightOptions:
            if option == DynamicWeightOptions.DISABLED:
                with self.subTest("weight=DISABLED returns None"):
                    cfg = self.preset(
                        weight_config=self._make_weight_config(model_type=option),
                    )
                    model = AdaptiveParameterAugmentation(cfg)
                    self.assertIsNone(model.generator_model)
                continue
            with self.subTest(f"weight={option}"):
                input_dim = 12
                output_dim = 24
                if option == DynamicWeightOptions.SINGLE_MODEL:
                    output_dim = input_dim
                is_bank_type = option in {
                    DynamicWeightOptions.LAYERED_WEIGHTED_BANK,
                    DynamicWeightOptions.SOFT_WEIGHTED_BANK,
                }
                bank_factor = None
                if is_bank_type:
                    bank_factor = BankExpansionFactorOptions.FACTOR_OF_TWO
                cfg = self.preset(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    weight_config=self._make_weight_config(
                        model_type=option,
                        input_dim=input_dim,
                        output_dim=output_dim,
                        bank_expansion_factor=bank_factor,
                    ),
                )
                model = AdaptiveParameterAugmentation(cfg)
                self.assertIsNotNone(model.generator_model)
                self.assertIsInstance(model.generator_model, DynamicWeightAbstract)

    def test_init_with_diagonal_config(self):
        for option in DynamicDiagonalOptions:
            if option == DynamicDiagonalOptions.DISABLED:
                with self.subTest("diagonal=DISABLED returns None"):
                    cfg = self.preset(
                        diagonal_config=self._make_diagonal_config(model_type=option),
                    )
                    model = AdaptiveParameterAugmentation(cfg)
                    self.assertIsNone(model.diagonal_model)
                continue
            with self.subTest(f"diagonal={option}"):
                cfg = self.preset(
                    diagonal_config=self._make_diagonal_config(model_type=option),
                )
                model = AdaptiveParameterAugmentation(cfg)
                self.assertIsNotNone(model.diagonal_model)
                self.assertIsInstance(model.diagonal_model, DynamicDiagonalAbstract)

    def test_init_with_bias_config(self):
        for option in DynamicBiasOptions:
            if option == DynamicBiasOptions.DISABLED:
                with self.subTest("bias=DISABLED returns None"):
                    cfg = self.preset(
                        bias_config=self._make_bias_config(model_type=option),
                    )
                    model = AdaptiveParameterAugmentation(cfg)
                    self.assertIsNone(model.bias_model)
                continue
            with self.subTest(f"bias={option}"):
                bank_factor = None
                if option == DynamicBiasOptions.WEIGHTED_BANK:
                    bank_factor = 4
                cfg = self.preset(
                    bias_config=self._make_bias_config(
                        model_type=option,
                        bank_expansion_factor=bank_factor,
                    ),
                )
                model = AdaptiveParameterAugmentation(cfg)
                self.assertIsNotNone(model.bias_model)
                self.assertIsInstance(model.bias_model, DynamicBiasAbstract)

    def test_init_with_mask_config(self):
        for option in AxisMaskOptions:
            if option == AxisMaskOptions.DISABLED:
                with self.subTest("mask=DISABLED returns None"):
                    cfg = self.preset(
                        mask_config=self._make_mask_config(model_type=option),
                    )
                    model = AdaptiveParameterAugmentation(cfg)
                    self.assertIsNone(model.mask_model)
                continue
            with self.subTest(f"mask={option}"):
                cfg = self.preset(
                    mask_config=self._make_mask_config(model_type=option),
                )
                model = AdaptiveParameterAugmentation(cfg)
                self.assertIsNotNone(model.mask_model)
                self.assertIsInstance(model.mask_model, AxisMaskAbstract)

    def test_forward_all_disabled(self):
        batch_size = 2
        input_dim = 12
        output_dim = 24
        cfg = self.preset(input_dim=input_dim, output_dim=output_dim)
        model = AdaptiveParameterAugmentation(cfg)

        weight_params, bias_params = self._make_weight_and_bias_params(
            input_dim, output_dim
        )
        input_tensor = torch.randn(batch_size, input_dim)
        callback = self._make_affine_callback()

        output = model(callback, weight_params, bias_params, input_tensor)
        self.assertEqual(output.shape, (batch_size, output_dim))

    def test_forward_without_bias(self):
        batch_size = 2
        input_dim = 12
        output_dim = 24
        cfg = self.preset(input_dim=input_dim, output_dim=output_dim)
        model = AdaptiveParameterAugmentation(cfg)

        weight_params = Module()._init_parameter_bank((input_dim, output_dim))
        input_tensor = torch.randn(batch_size, input_dim)
        callback = self._make_affine_callback()

        output = model(callback, weight_params, None, input_tensor)
        self.assertEqual(output.shape, (batch_size, output_dim))

    def test_forward_with_each_diagonal_option(self):
        batch_size = 2
        input_dim = 12
        output_dim = 24
        for option in DynamicDiagonalOptions:
            if option == DynamicDiagonalOptions.DISABLED:
                continue
            with self.subTest(f"diagonal={option}"):
                cfg = self.preset(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    diagonal_config=self._make_diagonal_config(
                        model_type=option, input_dim=input_dim, output_dim=output_dim
                    ),
                )
                model = AdaptiveParameterAugmentation(cfg)
                weight_params, bias_params = self._make_weight_and_bias_params(
                    input_dim, output_dim
                )
                input_tensor = torch.randn(batch_size, input_dim)
                callback = self._make_affine_callback()
                output = model(callback, weight_params, bias_params, input_tensor)
                self.assertEqual(output.shape, (batch_size, output_dim))

    def test_forward_with_each_bias_option(self):
        batch_size = 2
        input_dim = 12
        output_dim = 24
        for option in DynamicBiasOptions:
            if option == DynamicBiasOptions.DISABLED:
                continue
            with self.subTest(f"bias={option}"):
                bank_factor = None
                if option == DynamicBiasOptions.WEIGHTED_BANK:
                    bank_factor = 4
                cfg = self.preset(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    bias_config=self._make_bias_config(
                        model_type=option,
                        input_dim=input_dim,
                        output_dim=output_dim,
                        bank_expansion_factor=bank_factor,
                    ),
                )
                model = AdaptiveParameterAugmentation(cfg)
                weight_params, bias_params = self._make_weight_and_bias_params(
                    input_dim, output_dim
                )
                input_tensor = torch.randn(batch_size, input_dim)
                callback = self._make_affine_callback()
                output = model(callback, weight_params, bias_params, input_tensor)
                self.assertEqual(output.shape, (batch_size, output_dim))

    # def test_all_disabled_is_pure_passthrough(self):
    #     batch_size = 2
    #     input_dim = 12
    #     output_dim = 24
    #     cfg = self.preset(input_dim=input_dim, output_dim=output_dim)
    #     model = AdaptiveParameterAugmentation(cfg)
    #
    #     weight_params, bias_params = self._make_weight_and_bias_params(
    #         input_dim, output_dim
    #     )
    #     input_tensor = torch.randn(batch_size, input_dim)
    #     callback = self._make_affine_callback()
    #
    #     output = model(callback, weight_params, bias_params, input_tensor)
    #     expected = callback(weight_params, bias_params, input_tensor)
    #     self.assertTrue(
    #         torch.allclose(
    #             output.round(decimals=6),
    #             expected.round(decimals=6),
    #             atol=1e-6,
    #             rtol=1e-5,
    #         )
    #     )
    #
    # def test_all_disabled_without_bias_is_pure_passthrough(self):
    #     batch_size = 2
    #     input_dim = 12
    #     output_dim = 24
    #     cfg = self.preset(input_dim=input_dim, output_dim=output_dim)
    #     model = AdaptiveParameterAugmentation(cfg)
    #
    #     weight_params = Module()._init_parameter_bank((input_dim, output_dim))
    #     input_tensor = torch.randn(batch_size, input_dim)
    #     callback = self._make_affine_callback()
    #
    #     output = model(callback, weight_params, None, input_tensor)
    #     expected = callback(weight_params, None, input_tensor)
    #     self.assertTrue(
    #         torch.allclose(
    #             output.round(decimals=6),
    #             expected.round(decimals=6),
    #             atol=1e-6,
    #             rtol=1e-5,
    #         )
    #     )
    #
    # def test_diagonal_modifies_weights(self):
    #     batch_size = 2
    #     input_dim = 12
    #     output_dim = 24
    #     cfg = self.preset(
    #         input_dim=input_dim,
    #         output_dim=output_dim,
    #         diagonal_config=self._make_diagonal_config(
    #             input_dim=input_dim, output_dim=output_dim
    #         ),
    #     )
    #     model = AdaptiveParameterAugmentation(cfg)
    #
    #     weight_params, bias_params = self._make_weight_and_bias_params(
    #         input_dim, output_dim
    #     )
    #     input_tensor = torch.randn(batch_size, input_dim)
    #     callback = self._make_affine_callback()
    #
    #     output = model(callback, weight_params, bias_params, input_tensor)
    #     baseline = callback(weight_params, bias_params, input_tensor)
    #     self.assertFalse(torch.allclose(output, baseline, atol=1e-6))
    #
    # def test_bias_modifies_output(self):
    #     batch_size = 2
    #     input_dim = 12
    #     output_dim = 24
    #     cfg = self.preset(
    #         input_dim=input_dim,
    #         output_dim=output_dim,
    #         bias_config=self._make_bias_config(
    #             input_dim=input_dim, output_dim=output_dim
    #         ),
    #     )
    #     model = AdaptiveParameterAugmentation(cfg)
    #
    #     weight_params, bias_params = self._make_weight_and_bias_params(
    #         input_dim, output_dim
    #     )
    #     input_tensor = torch.randn(batch_size, input_dim)
    #     callback = self._make_affine_callback()
    #
    #     output = model(callback, weight_params, bias_params, input_tensor)
    #     baseline = callback(weight_params, bias_params, input_tensor)
    #     self.assertFalse(torch.allclose(output, baseline, atol=1e-6))
    #
    # def test_init_raises_on_missing_input_dim(self):
    #     cfg = AdaptiveParameterAugmentationConfig(output_dim=24)
    #     with self.assertRaises(ValueError):
    #         AdaptiveParameterAugmentation(cfg)
    #
    # def test_init_raises_on_missing_output_dim(self):
    #     cfg = AdaptiveParameterAugmentationConfig(input_dim=12)
    #     with self.assertRaises(ValueError):
    #         AdaptiveParameterAugmentation(cfg)
    #
    # def test_init_raises_on_zero_input_dim(self):
    #     cfg = AdaptiveParameterAugmentationConfig(input_dim=0, output_dim=24)
    #     with self.assertRaises(ValueError):
    #         AdaptiveParameterAugmentation(cfg)
    #
    # def test_init_raises_on_negative_output_dim(self):
    #     cfg = AdaptiveParameterAugmentationConfig(input_dim=12, output_dim=-1)
    #     with self.assertRaises(ValueError):
    #         AdaptiveParameterAugmentation(cfg)
    #
    # def test_init_raises_on_missing_model_config(self):
    #     cfg = AdaptiveParameterAugmentationConfig(
    #         input_dim=12,
    #         output_dim=24,
    #         diagonal_config=DynamicDiagonalConfig(
    #             model_type=DynamicDiagonalOptions.DIAGONAL,
    #         ),
    #     )
    #     with self.assertRaises(ValueError):
    #         AdaptiveParameterAugmentation(cfg)
    #
    # def test_sub_config_inherits_parent_model_config(self):
    #     parent_model_config = self._make_layer_stack_config(input_dim=12, output_dim=24)
    #     cfg = AdaptiveParameterAugmentationConfig(
    #         input_dim=12,
    #         output_dim=24,
    #         model_config=parent_model_config,
    #         diagonal_config=DynamicDiagonalConfig(
    #             model_type=DynamicDiagonalOptions.DIAGONAL,
    #         ),
    #     )
    #     model = AdaptiveParameterAugmentation(cfg)
    #     self.assertIsNotNone(model.diagonal_model)
    #     self.assertIsInstance(model.diagonal_model, DynamicDiagonalAbstract)
    #
    # def test_gradients_flow_with_diagonal(self):
    #     batch_size = 2
    #     input_dim = 12
    #     output_dim = 24
    #     for option in DynamicDiagonalOptions:
    #         if option == DynamicDiagonalOptions.DISABLED:
    #             continue
    #         with self.subTest(f"diagonal={option}"):
    #             cfg = self.preset(
    #                 input_dim=input_dim,
    #                 output_dim=output_dim,
    #                 diagonal_config=self._make_diagonal_config(
    #                     model_type=option,
    #                     input_dim=input_dim,
    #                     output_dim=output_dim,
    #                 ),
    #             )
    #             model = AdaptiveParameterAugmentation(cfg)
    #             weight_params, bias_params = self._make_weight_and_bias_params(
    #                 input_dim, output_dim
    #             )
    #             input_tensor = torch.randn(batch_size, input_dim, requires_grad=True)
    #             callback = self._make_affine_callback()
    #             output = model(callback, weight_params, bias_params, input_tensor)
    #             output.sum().backward()
    #             grads = [p.grad for p in model.parameters() if p.requires_grad]
    #             non_none_grads = [
    #                 g for g in grads if g is not None and g.abs().sum() > 0
    #             ]
    #             self.assertTrue(len(non_none_grads) > 0)
    #             self.assertIsNotNone(input_tensor.grad)
    #
    # def test_gradients_flow_with_bias(self):
    #     batch_size = 2
    #     input_dim = 12
    #     output_dim = 24
    #     for option in DynamicBiasOptions:
    #         if option == DynamicBiasOptions.DISABLED:
    #             continue
    #         with self.subTest(f"bias={option}"):
    #             cfg = self.preset(
    #                 input_dim=input_dim,
    #                 output_dim=output_dim,
    #                 bias_config=self._make_bias_config(
    #                     model_type=option,
    #                     input_dim=input_dim,
    #                     output_dim=output_dim,
    #                 ),
    #             )
    #             model = AdaptiveParameterAugmentation(cfg)
    #             weight_params, bias_params = self._make_weight_and_bias_params(
    #                 input_dim, output_dim
    #             )
    #             input_tensor = torch.randn(batch_size, input_dim, requires_grad=True)
    #             callback = self._make_affine_callback()
    #             output = model(callback, weight_params, bias_params, input_tensor)
    #             output.sum().backward()
    #             grads = [p.grad for p in model.parameters() if p.requires_grad]
    #             non_none_grads = [
    #                 g for g in grads if g is not None and g.abs().sum() > 0
    #             ]
    #             self.assertTrue(len(non_none_grads) > 0)
    #             self.assertIsNotNone(input_tensor.grad)
    #
    # def test_forward_full_pipeline(self):
    #     batch_size = 2
    #     input_dim = 12
    #     output_dim = 24
    #     for diagonal_option in DynamicDiagonalOptions:
    #         if diagonal_option == DynamicDiagonalOptions.DISABLED:
    #             continue
    #         for bias_option in DynamicBiasOptions:
    #             if bias_option == DynamicBiasOptions.DISABLED:
    #                 continue
    #             msg = f"diagonal={diagonal_option}, bias={bias_option}"
    #             with self.subTest(msg):
    #                 cfg = self.preset(
    #                     input_dim=input_dim,
    #                     output_dim=output_dim,
    #                     diagonal_config=self._make_diagonal_config(
    #                         model_type=diagonal_option,
    #                         input_dim=input_dim,
    #                         output_dim=output_dim,
    #                     ),
    #                     bias_config=self._make_bias_config(
    #                         model_type=bias_option,
    #                         input_dim=input_dim,
    #                         output_dim=output_dim,
    #                     ),
    #                 )
    #                 model = AdaptiveParameterAugmentation(cfg)
    #                 weight_params, bias_params = self._make_weight_and_bias_params(
    #                     input_dim, output_dim
    #                 )
    #                 input_tensor = torch.randn(batch_size, input_dim)
    #                 callback = self._make_affine_callback()
    #                 output = model(callback, weight_params, bias_params, input_tensor)
    #                 self.assertEqual(output.shape, (batch_size, output_dim))
    #                 self.assertIsInstance(output, torch.Tensor)

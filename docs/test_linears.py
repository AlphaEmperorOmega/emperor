import torch
import unittest
from torch.nn import Sequential

from emperor.base.layer import Layer
from emperor.base.layer import LayerStack
from emperor.base.layer.config import LayerConfig
from emperor.halting.config import StickBreakingConfig
from emperor.halting.options import HaltingHiddenStateModeOptions
from emperor.linears.core.config import (
    AdaptiveLinearLayerConfig,
    LinearLayerConfig,
)
from emperor.linears.core.layers import (
    AdaptiveLinearLayer,
    LinearLayer,
)

from emperor.base.layer import LayerStackConfig
from emperor.base.enums import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
)
from emperor.augmentations.adaptive_parameters import (
    AdaptiveParameterAugmentationConfig,
    DualModelDynamicWeightConfig,
    GeneratorDynamicBiasConfig,
    StandardDynamicDiagonalConfig,
    WeightInformedScoreAxisMaskConfig,
)
from emperor.augmentations.adaptive_parameters.options import (
    DynamicDepthOptions,
    MaskDimensionOptions,
    WeightDecayScheduleOptions,
    WeightNormalizationOptions,
    WeightNormalizationPositionOptions,
)


def make_layer_stack_config(
    hidden_dim: int = 16,
    num_layers: int = 2,
) -> LayerStackConfig:
    return LayerStackConfig(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
        apply_output_pipeline_flag=False,
        layer_config=LayerConfig(
            activation=ActivationOptions.RELU,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            residual_flag=False,
            dropout_probability=0.0,
            gate_config=None,
            halting_config=None,
            shared_halting_flag=False,
            layer_model_config=LinearLayerConfig(
                bias_flag=True,
            ),
        ),
    )


def make_weight_config(input_dim: int, output_dim: int) -> DualModelDynamicWeightConfig:
    return DualModelDynamicWeightConfig(
        generator_depth=DynamicDepthOptions.DEPTH_OF_TWO,
        normalization_option=WeightNormalizationOptions.L2_SCALE,
        normalization_position_option=WeightNormalizationPositionOptions.BEFORE_OUTER_PRODUCT,
        decay_schedule=WeightDecayScheduleOptions.DISABLED,
        decay_rate=0.0,
        decay_warmup_batches=0,
        model_config=make_layer_stack_config(input_dim, output_dim),
    )


def make_bias_config(
    input_dim: int, output_dim: int, bias_flag: bool = True
) -> GeneratorDynamicBiasConfig:
    return GeneratorDynamicBiasConfig(
        bias_flag=bias_flag,
        decay_schedule=WeightDecayScheduleOptions.DISABLED,
        decay_rate=0.0,
        decay_warmup_batches=0,
        model_config=make_layer_stack_config(input_dim, output_dim),
    )


def make_diagonal_config(
    input_dim: int, output_dim: int
) -> StandardDynamicDiagonalConfig:
    return StandardDynamicDiagonalConfig(
        model_config=make_layer_stack_config(input_dim, output_dim),
    )


def make_mask_config(
    input_dim: int, output_dim: int
) -> WeightInformedScoreAxisMaskConfig:
    return WeightInformedScoreAxisMaskConfig(
        mask_threshold=0.5,
        mask_surrogate_scale=10.0,
        mask_floor=0.0,
        mask_dimension_option=MaskDimensionOptions.COLUMN,
        model_config=make_layer_stack_config(input_dim, output_dim),
    )


class TestLinearLayer(unittest.TestCase):
    def preset(
        self,
        input_dim: int = 12,
        output_dim: int = 6,
        bias_flag: bool = True,
    ) -> LinearLayerConfig:

        return LinearLayerConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            bias_flag=bias_flag,
        )

    def test_init_with_different_configation_options(self):
        bias_options = [True, False]
        for bias_flag in bias_options:
            message = f"Test failed for the inputs: {bias_flag}"
            with self.subTest(i=message):
                c = self.preset(bias_flag=bias_flag)
                m = LinearLayer(c)

                expected_weight_shape = (c.input_dim, c.output_dim)
                expected_bias_shape = (c.output_dim,)
                self.assertEqual(m.input_dim, c.input_dim)
                self.assertEqual(m.output_dim, c.output_dim)
                self.assertEqual(m.bias_flag, bias_flag)
                self.assertIsInstance(m.weight_params, torch.Tensor)
                self.assertEqual(m.weight_params.shape, expected_weight_shape)
                if bias_flag:
                    self.assertIsInstance(m.bias_params, torch.Tensor)
                    self.assertEqual(m.bias_params.shape, expected_bias_shape)
                else:
                    self.assertIsNone(m.bias_params)

    def test_forward(self):
        batch_size = 5
        bias_options = [True, False]
        input_params = output_params = [4, 8, 16]

        for input_dim in input_params:
            for output_dim in output_params:
                for bias_flag in bias_options:
                    message = f"Test failed for the options: {input_dim}, {output_dim}, {bias_flag}"
                    with self.subTest(i=message):
                        c = self.preset()
                        overrides = LinearLayerConfig(
                            input_dim=input_dim,
                            output_dim=output_dim,
                            bias_flag=bias_flag,
                        )
                        m = LinearLayer(c, overrides)

                        input_batch = torch.randn(batch_size, overrides.input_dim)
                        output = m.forward(input_batch)
                        expected_output_shape = (batch_size, overrides.output_dim)
                        self.assertEqual(output.shape, expected_output_shape)

    def test_gradients_flow_through_linear_layer(self):
        batch_size = 5
        input_params = [4, 8]
        output_params = [3, 6]
        for input_dim in input_params:
            for output_dim in output_params:
                for bias_flag in [True, False]:
                    with self.subTest(
                        input_dim=input_dim, output_dim=output_dim, bias_flag=bias_flag
                    ):
                        c = self.preset()
                        overrides = LinearLayerConfig(
                            input_dim=input_dim,
                            output_dim=output_dim,
                            bias_flag=bias_flag,
                        )
                        m = LinearLayer(c, overrides)

                        input_batch = torch.randn(
                            batch_size, overrides.input_dim, requires_grad=True
                        )
                        output = m.forward(input_batch)
                        output.sum().backward()

                        self.assertIsNotNone(m.weight_params.grad)
                        self.assertEqual(
                            m.weight_params.grad.shape, m.weight_params.shape
                        )

                        if bias_flag:
                            self.assertIsNotNone(m.bias_params.grad)
                            self.assertEqual(
                                m.bias_params.grad.shape, m.bias_params.shape
                            )
                        else:
                            self.assertIsNone(m.bias_params)

    def test_output_matches_torch_linear(self):
        for bias_flag in [True, False]:
            with self.subTest(bias_flag=bias_flag):
                c = self.preset(input_dim=4, output_dim=3, bias_flag=bias_flag)
                m = LinearLayer(c)

                ref = torch.nn.Linear(4, 3, bias=bias_flag)
                with torch.no_grad():
                    ref.weight.copy_(m.weight_params.T)
                    if bias_flag:
                        ref.bias.copy_(m.bias_params)

                input_batch = torch.randn(2, 4)
                torch.testing.assert_close(m.forward(input_batch), ref(input_batch))

    def test_deterministic_output(self):
        c = self.preset(input_dim=4, output_dim=3, bias_flag=True)
        m = LinearLayer(c)

        input_batch = torch.randn(2, 4)
        output_1 = m.forward(input_batch)
        output_2 = m.forward(input_batch)
        torch.testing.assert_close(output_1, output_2)

    def test_forward_raises_on_1d_input(self):
        c = self.preset()
        m = LinearLayer(c)
        with self.assertRaises(ValueError):
            m.forward(torch.randn(c.input_dim))

    def test_forward_raises_on_3d_input(self):
        c = self.preset()
        m = LinearLayer(c)
        with self.assertRaises(ValueError):
            m.forward(torch.randn(2, 3, c.input_dim))

    def test_init_raises_on_missing_input_dim(self):
        with self.assertRaises(ValueError):
            LinearLayer(LinearLayerConfig(output_dim=4, bias_flag=True))

    def test_init_raises_on_missing_output_dim(self):
        with self.assertRaises(ValueError):
            LinearLayer(LinearLayerConfig(input_dim=4, bias_flag=True))

    def test_init_raises_on_zero_or_negative_dims(self):
        invalid_cases = [
            ("zero_input_dim", {"input_dim": 0, "output_dim": 4}),
            ("zero_output_dim", {"input_dim": 4, "output_dim": 0}),
            ("negative_input_dim", {"input_dim": -1, "output_dim": 4}),
            ("negative_output_dim", {"input_dim": 4, "output_dim": -1}),
        ]
        for case, kwargs in invalid_cases:
            with self.subTest(case=case):
                with self.assertRaises(ValueError):
                    LinearLayer(LinearLayerConfig(bias_flag=True, **kwargs))

    def test_overrides_take_precedence_over_base_cfg(self):
        base = LinearLayerConfig(input_dim=4, output_dim=4, bias_flag=True)
        overrides = LinearLayerConfig(input_dim=8, output_dim=2, bias_flag=False)
        m = LinearLayer(base, overrides)

        self.assertEqual(m.input_dim, 8)
        self.assertEqual(m.output_dim, 2)
        self.assertEqual(m.bias_flag, False)
        self.assertIsNone(m.bias_params)

    def test_partial_overrides_keep_unset_fields_from_base_cfg(self):
        base = LinearLayerConfig(input_dim=4, output_dim=6, bias_flag=True)
        overrides = LinearLayerConfig(input_dim=8)
        m = LinearLayer(base, overrides)

        self.assertEqual(m.input_dim, 8)
        self.assertEqual(m.output_dim, 6)
        self.assertEqual(m.bias_flag, True)

    def test_model_config_dispatch_extracts_linear_layer_config(self):
        from emperor.config import ModelConfig

        wrapper = ModelConfig()
        wrapper.linear_layer_config = LinearLayerConfig(
            input_dim=5, output_dim=3, bias_flag=True
        )
        m = LinearLayer(wrapper)

        self.assertEqual(m.input_dim, 5)
        self.assertEqual(m.output_dim, 3)
        self.assertEqual(m.bias_flag, True)
        self.assertEqual(m.weight_params.shape, (5, 3))


class TestLinearLayerStack(unittest.TestCase):
    def preset(
        self,
        input_dim: int = 12,
        hidden_dim: int = 24,
        output_dim: int = 6,
        bias_flag: bool = True,
        layer_norm_position: LayerNormPositionOptions = LayerNormPositionOptions.DISABLED,
        stack_num_layers: int = 2,
        stack_activation: ActivationOptions = ActivationOptions.RELU,
        stack_residual_flag: bool = True,
        stack_dropout_probability: float = 0.2,
        shared_halting_flag: bool = False,
        last_layer_bias_option: LastLayerBiasOptions = LastLayerBiasOptions.DEFAULT,
        apply_output_pipeline_flag: bool = True,
        gate_config: "LayerStackConfig | None" = None,
    ) -> LayerStackConfig:

        if gate_config is None:
            gate_config = LayerStackConfig(
                hidden_dim=hidden_dim,
                num_layers=stack_num_layers,
                last_layer_bias_option=last_layer_bias_option,
                apply_output_pipeline_flag=apply_output_pipeline_flag,
                layer_config=LayerConfig(
                    activation=stack_activation,
                    layer_norm_position=layer_norm_position,
                    residual_flag=stack_residual_flag,
                    dropout_probability=stack_dropout_probability,
                    halting_config=None,
                    shared_halting_flag=False,
                    gate_config=None,
                    layer_model_config=LinearLayerConfig(
                        input_dim=input_dim,
                        output_dim=output_dim,
                        bias_flag=bias_flag,
                    ),
                ),
            )

        halting_config = None
        if stack_num_layers > 1 and input_dim == hidden_dim == output_dim:
            halting_config = StickBreakingConfig(
                threshold=0.99,
                halting_dropout=0.0,
                hidden_state_mode=HaltingHiddenStateModeOptions.RAW,
                halting_gate_config=LayerStackConfig(
                    hidden_dim=output_dim,
                    output_dim=2,
                    num_layers=stack_num_layers,
                    last_layer_bias_option=LastLayerBiasOptions.DISABLED,
                    apply_output_pipeline_flag=False,
                    layer_config=LayerConfig(
                        activation=ActivationOptions.DISABLED,
                        layer_norm_position=LayerNormPositionOptions.DISABLED,
                        residual_flag=stack_residual_flag,
                        dropout_probability=stack_dropout_probability,
                        halting_config=None,
                        shared_halting_flag=False,
                        gate_config=None,
                        layer_model_config=LinearLayerConfig(
                            bias_flag=True,
                        ),
                    ),
                ),
            )

        return LayerStackConfig(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=stack_num_layers,
            last_layer_bias_option=last_layer_bias_option,
            apply_output_pipeline_flag=apply_output_pipeline_flag,
            layer_config=LayerConfig(
                activation=stack_activation,
                layer_norm_position=layer_norm_position,
                residual_flag=stack_residual_flag,
                dropout_probability=stack_dropout_probability,
                gate_config=gate_config,
                halting_config=halting_config,
                shared_halting_flag=shared_halting_flag,
                layer_model_config=LinearLayerConfig(
                    bias_flag=bias_flag,
                ),
            ),
        )

    def test_stack_layers_contain_linear_layer(self):
        num_layer_options = [1, 2, 3]

        for num_layers in num_layer_options:
            with self.subTest(num_layers=num_layers):
                cfg = self.preset(stack_num_layers=num_layers)
                m = LayerStack(cfg).build()

                layers = [m] if isinstance(m, Layer) else list(m)
                for i, layer in enumerate(layers):
                    with self.subTest(layer_index=i):
                        self.assertIsInstance(layer.model, LinearLayer)

    def test_gradients_flow_through_linear_layer_stack(self):
        num_layer_options = [1, 2, 3]
        for num_layers in num_layer_options:
            with self.subTest(num_layers=num_layers):
                batch_size = 2
                input_dim = 8
                output_dim = 4
                cfg = self.preset(
                    stack_num_layers=num_layers,
                    input_dim=input_dim,
                    output_dim=output_dim,
                )
                m = LayerStack(cfg).build()

                input_batch = torch.randn(batch_size, input_dim, requires_grad=True)
                output = Layer.forward_with_state(m, input_batch)
                output.sum().backward()

                grads = [p.grad for p in m.parameters() if p.requires_grad]
                non_none_grads = [g for g in grads if g is not None]
                self.assertTrue(len(non_none_grads) > 0)


class TestAdaptiveLinearLayer(unittest.TestCase):
    def preset(
        self,
        input_dim: int = 12,
        output_dim: int = 6,
        bias_flag: bool = True,
        weight_config=None,
        bias_config=None,
        diagonal_config=None,
        mask_config=None,
    ) -> AdaptiveLinearLayerConfig:

        return AdaptiveLinearLayerConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            bias_flag=bias_flag,
            adaptive_augmentation_config=AdaptiveParameterAugmentationConfig(
                weight_config=weight_config,
                bias_config=bias_config,
                diagonal_config=diagonal_config,
                mask_config=mask_config,
                model_config=make_layer_stack_config(input_dim, output_dim),
            ),
        )

    def test_init_with_all_sub_configs_disabled(self):
        bias_options = [True, False]
        for bias_flag in bias_options:
            with self.subTest(bias_flag=bias_flag):
                cfg = self.preset(bias_flag=bias_flag)
                m = AdaptiveLinearLayer(cfg)

                self.assertEqual(m.input_dim, cfg.input_dim)
                self.assertEqual(m.output_dim, cfg.output_dim)
                self.assertEqual(m.bias_flag, bias_flag)
                self.assertEqual(m.weight_params.shape, (cfg.input_dim, cfg.output_dim))
                if bias_flag:
                    self.assertEqual(m.bias_params.shape, (cfg.output_dim,))
                else:
                    self.assertIsNone(m.bias_params)

    def test_init_raises_on_missing_adaptive_augmentation_config(self):
        cfg = AdaptiveLinearLayerConfig(
            input_dim=4,
            output_dim=3,
            bias_flag=True,
            adaptive_augmentation_config=None,
        )
        with self.assertRaises(ValueError):
            AdaptiveLinearLayer(cfg)

    def test_output_matches_linear_layer_when_all_sub_configs_disabled(self):
        input_dim = 4
        output_dim = 3
        batch_size = 2
        adaptive_cfg = self.preset(
            input_dim=input_dim, output_dim=output_dim, bias_flag=True
        )
        adaptive_layer = AdaptiveLinearLayer(adaptive_cfg)

        linear_cfg = LinearLayerConfig(
            input_dim=input_dim, output_dim=output_dim, bias_flag=True
        )
        linear_layer = LinearLayer(linear_cfg)

        with torch.no_grad():
            linear_layer.weight_params.copy_(adaptive_layer.weight_params)
            linear_layer.bias_params.copy_(adaptive_layer.bias_params)

        input_batch = torch.randn(batch_size, input_dim)
        torch.testing.assert_close(
            adaptive_layer.forward(input_batch),
            linear_layer.forward(input_batch),
        )

    def test_init_with_different_sub_config_combinations(self):
        input_dim = 12
        output_dim = 6
        bias_options = [True, False]
        weight_options = [None, make_weight_config]
        bias_config_options = [None, make_bias_config]
        diagonal_options = [None, make_diagonal_config]
        mask_options = [None, make_mask_config]

        for bias_flag in bias_options:
            for weight_builder in weight_options:
                for bias_builder in bias_config_options:
                    for diagonal_builder in diagonal_options:
                        for mask_builder in mask_options:
                            message = (
                                f"bias_flag={bias_flag}, "
                                f"weight={'set' if weight_builder else 'none'}, "
                                f"bias_config={'set' if bias_builder else 'none'}, "
                                f"diagonal={'set' if diagonal_builder else 'none'}, "
                                f"mask={'set' if mask_builder else 'none'}"
                            )
                            with self.subTest(message=message):
                                weight_config = (
                                    weight_builder(input_dim, output_dim)
                                    if weight_builder
                                    else None
                                )
                                bias_config = (
                                    bias_builder(input_dim, output_dim)
                                    if bias_builder
                                    else None
                                )
                                diagonal_config = (
                                    diagonal_builder(input_dim, output_dim)
                                    if diagonal_builder
                                    else None
                                )
                                mask_config = (
                                    mask_builder(input_dim, output_dim)
                                    if mask_builder
                                    else None
                                )
                                cfg = self.preset(
                                    input_dim=input_dim,
                                    output_dim=output_dim,
                                    bias_flag=bias_flag,
                                    weight_config=weight_config,
                                    bias_config=bias_config,
                                    diagonal_config=diagonal_config,
                                    mask_config=mask_config,
                                )
                                m = AdaptiveLinearLayer(cfg)

                                self.assertEqual(m.input_dim, cfg.input_dim)
                                self.assertEqual(m.output_dim, cfg.output_dim)
                                self.assertIsInstance(m.weight_params, torch.Tensor)
                                if bias_flag:
                                    self.assertIsInstance(m.bias_params, torch.Tensor)
                                else:
                                    self.assertIsNone(m.bias_params)

    def test_forward(self):
        batch_size = 2
        bias_options = [True, False]
        input_params = output_params = [8, 16]
        weight_options = [None, make_weight_config]
        bias_config_options = [None, make_bias_config]
        diagonal_options = [None, make_diagonal_config]
        mask_options = [None, make_mask_config]

        for bias_flag in bias_options:
            for input_dim in input_params:
                for output_dim in output_params:
                    for weight_builder in weight_options:
                        for bias_builder in bias_config_options:
                            for diagonal_builder in diagonal_options:
                                for mask_builder in mask_options:
                                    message = (
                                        f"bias_flag={bias_flag}, "
                                        f"input_dim={input_dim}, output_dim={output_dim}, "
                                        f"weight={'set' if weight_builder else 'none'}, "
                                        f"bias_config={'set' if bias_builder else 'none'}, "
                                        f"diagonal={'set' if diagonal_builder else 'none'}, "
                                        f"mask={'set' if mask_builder else 'none'}"
                                    )
                                    with self.subTest(message=message):
                                        weight_config = (
                                            weight_builder(input_dim, output_dim)
                                            if weight_builder
                                            else None
                                        )
                                        bias_config = (
                                            bias_builder(
                                                input_dim, output_dim, bias_flag
                                            )
                                            if bias_builder
                                            else None
                                        )
                                        diagonal_config = (
                                            diagonal_builder(input_dim, output_dim)
                                            if diagonal_builder
                                            else None
                                        )
                                        mask_config = (
                                            mask_builder(input_dim, output_dim)
                                            if mask_builder
                                            else None
                                        )
                                        cfg = self.preset(
                                            input_dim=input_dim,
                                            output_dim=output_dim,
                                            bias_flag=bias_flag,
                                            weight_config=weight_config,
                                            bias_config=bias_config,
                                            diagonal_config=diagonal_config,
                                            mask_config=mask_config,
                                        )

                                        m = AdaptiveLinearLayer(cfg)
                                        input_batch = torch.randn(batch_size, input_dim)
                                        output = m.forward(input_batch)
                                        self.assertEqual(
                                            output.shape, (batch_size, output_dim)
                                        )


class TestLinearLayerAdaptiveStack(unittest.TestCase):
    def preset(
        self,
        input_dim: int = 8,
        hidden_dim: int = 16,
        output_dim: int = 4,
        bias_flag: bool = True,
        stack_num_layers: int = 2,
        stack_activation: ActivationOptions = ActivationOptions.RELU,
        stack_residual_flag: bool = False,
        stack_dropout_probability: float = 0.0,
        layer_norm_position: LayerNormPositionOptions = LayerNormPositionOptions.DISABLED,
        last_layer_bias_option: LastLayerBiasOptions = LastLayerBiasOptions.DEFAULT,
        weight_config=None,
        bias_config=None,
        diagonal_config=None,
        mask_config=None,
    ) -> LayerStackConfig:
        return LayerStackConfig(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=stack_num_layers,
            last_layer_bias_option=last_layer_bias_option,
            apply_output_pipeline_flag=True,
            layer_config=LayerConfig(
                activation=stack_activation,
                layer_norm_position=layer_norm_position,
                residual_flag=stack_residual_flag,
                dropout_probability=stack_dropout_probability,
                gate_config=None,
                halting_config=None,
                shared_halting_flag=False,
                layer_model_config=AdaptiveLinearLayerConfig(
                    bias_flag=bias_flag,
                    adaptive_augmentation_config=AdaptiveParameterAugmentationConfig(
                        weight_config=weight_config,
                        bias_config=bias_config,
                        diagonal_config=diagonal_config,
                        mask_config=mask_config,
                        model_config=make_layer_stack_config(input_dim, output_dim),
                    ),
                ),
            ),
        )

    def test_stack_layers_contain_adaptive_linear_layer(self):
        num_layer_options = [1, 2, 3]

        for num_layers in num_layer_options:
            with self.subTest(num_layers=num_layers):
                cfg = self.preset(stack_num_layers=num_layers)
                m = LayerStack(cfg).build()

                if num_layers == 1:
                    self.assertIsInstance(m, Layer)
                else:
                    self.assertIsInstance(m, Sequential)

                layers = [m] if isinstance(m, Layer) else list(m)
                for i, layer in enumerate(layers):
                    with self.subTest(layer_index=i):
                        self.assertIsInstance(layer.model, AdaptiveLinearLayer)

    def test_forward_with_different_sub_config_combinations(self):
        batch_size = 2
        input_dim = 8
        output_dim = 4
        stack_num_layers = 2
        bias_options = [True, False]
        weight_options = [None, make_weight_config]
        bias_config_options = [None, make_bias_config]
        diagonal_options = [None, make_diagonal_config]
        mask_options = [None, make_mask_config]

        for bias_flag in bias_options:
            for weight_builder in weight_options:
                for bias_builder in bias_config_options:
                    for diagonal_builder in diagonal_options:
                        for mask_builder in mask_options:
                            message = (
                                f"bias_flag={bias_flag}, "
                                f"weight={'set' if weight_builder else 'none'}, "
                                f"bias_config={'set' if bias_builder else 'none'}, "
                                f"diagonal={'set' if diagonal_builder else 'none'}, "
                                f"mask={'set' if mask_builder else 'none'}"
                            )
                            with self.subTest(message=message):
                                weight_config = (
                                    weight_builder(input_dim, output_dim)
                                    if weight_builder
                                    else None
                                )
                                bias_config = (
                                    bias_builder(input_dim, output_dim, bias_flag)
                                    if bias_builder
                                    else None
                                )
                                diagonal_config = (
                                    diagonal_builder(input_dim, output_dim)
                                    if diagonal_builder
                                    else None
                                )
                                mask_config = (
                                    mask_builder(input_dim, output_dim)
                                    if mask_builder
                                    else None
                                )
                                cfg = self.preset(
                                    input_dim=input_dim,
                                    output_dim=output_dim,
                                    bias_flag=bias_flag,
                                    stack_num_layers=stack_num_layers,
                                    weight_config=weight_config,
                                    bias_config=bias_config,
                                    diagonal_config=diagonal_config,
                                    mask_config=mask_config,
                                )
                                m = LayerStack(cfg).build()

                                input_batch = torch.randn(batch_size, input_dim)
                                output = Layer.forward_with_state(m, input_batch)
                                self.assertEqual(output.shape, (batch_size, output_dim))

    def test_gradients_flow_through_adaptive_linear_layer_stack(self):
        batch_size = 2
        input_dim = 8
        output_dim = 4
        num_layer_options = [1, 2, 3]
        bias_options = [True, False]
        weight_options = [None, make_weight_config]
        bias_config_options = [None, make_bias_config]
        diagonal_options = [None, make_diagonal_config]
        mask_options = [None, make_mask_config]

        for num_layers in num_layer_options:
            for bias_flag in bias_options:
                for weight_builder in weight_options:
                    for bias_builder in bias_config_options:
                        for diagonal_builder in diagonal_options:
                            for mask_builder in mask_options:
                                message = (
                                    f"num_layers={num_layers}, "
                                    f"bias_flag={bias_flag}, "
                                    f"weight={'set' if weight_builder else 'none'}, "
                                    f"bias_config={'set' if bias_builder else 'none'}, "
                                    f"diagonal={'set' if diagonal_builder else 'none'}, "
                                    f"mask={'set' if mask_builder else 'none'}"
                                )
                                with self.subTest(message=message):
                                    weight_config = (
                                        weight_builder(input_dim, output_dim)
                                        if weight_builder
                                        else None
                                    )
                                    bias_config = (
                                        bias_builder(input_dim, output_dim, bias_flag)
                                        if bias_builder
                                        else None
                                    )
                                    diagonal_config = (
                                        diagonal_builder(input_dim, output_dim)
                                        if diagonal_builder
                                        else None
                                    )
                                    mask_config = (
                                        mask_builder(input_dim, output_dim)
                                        if mask_builder
                                        else None
                                    )
                                    cfg = self.preset(
                                        stack_num_layers=num_layers,
                                        input_dim=input_dim,
                                        output_dim=output_dim,
                                        bias_flag=bias_flag,
                                        weight_config=weight_config,
                                        bias_config=bias_config,
                                        diagonal_config=diagonal_config,
                                        mask_config=mask_config,
                                    )
                                    m = LayerStack(cfg).build()

                                    input_batch = torch.randn(
                                        batch_size, input_dim, requires_grad=True
                                    )
                                    output = Layer.forward_with_state(m, input_batch)
                                    output.sum().backward()

                                    grads = [
                                        p.grad
                                        for p in m.parameters()
                                        if p.requires_grad
                                    ]
                                    non_none_grads = [g for g in grads if g is not None]
                                    self.assertTrue(len(non_none_grads) > 0)

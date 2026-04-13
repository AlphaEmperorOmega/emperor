import torch
import unittest

from emperor.halting.config import StickBreakingConfig
from emperor.halting.options import HaltingHiddenStateModeOptions
from emperor.linears.core.config import LinearLayerConfig
from emperor.base.layer.config import LayerConfig, LayerStackConfig
from emperor.augmentations.adaptive_parameters.options import DynamicDepthOptions
from emperor.base.enums import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
)
from emperor.augmentations.adaptive_parameters.core.handlers.depth_mapper import (
    DepthMappingHandlerConfig,
    DepthMappingLayer,
    DepthMappingLayerConfig,
    DepthMappingLayerStack,
)


class TestDepthMappingAugmentation(unittest.TestCase):
    def preset(
        self,
        input_dim: int = 12,
        output_dim: int = 6,
        bias_flag: bool = True,
        generator_depth: DynamicDepthOptions = DynamicDepthOptions.DISABLED,
    ) -> DepthMappingLayerConfig:

        return DepthMappingLayerConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            bias_flag=bias_flag,
            generator_depth=generator_depth,
        )

    def test_forward_across_depths(self):
        batch_size = 2
        input_dim = 12
        output_dim = 6
        bias_options = [True, False]
        valid_depths = [
            DynamicDepthOptions.DEPTH_OF_ONE,
            DynamicDepthOptions.DEPTH_OF_TWO,
            DynamicDepthOptions.DEPTH_OF_THREE,
        ]
        for depth in valid_depths:
            for bias_flag in bias_options:
                with self.subTest(depth=depth, bias_flag=bias_flag):
                    cfg = self.preset(
                        input_dim=input_dim,
                        output_dim=output_dim,
                        bias_flag=bias_flag,
                        generator_depth=depth,
                    )
                    model = DepthMappingLayer(cfg)

                    input_tensor = torch.randn(batch_size, depth.value, input_dim)
                    output = model(input_tensor)
                    expected_shape = (batch_size, depth.value, output_dim)
                    self.assertEqual(output.shape, expected_shape)

                    if bias_flag:
                        self.assertIsNotNone(model.bias_params)
                    else:
                        self.assertIsNone(model.bias_params)

                    for i in range(batch_size):
                        for j in range(depth.value):
                            weight_slice = model.weight_params[j]
                            expected_output = torch.matmul(
                                input_tensor[i, j], weight_slice
                            )
                            if bias_flag:
                                expected_output = expected_output + model.bias_params[j]
                            torch.testing.assert_close(output[i, j], expected_output)

    def test_build_creates_depth_mapping_layer(self):
        cfg = self.preset(generator_depth=DynamicDepthOptions.DEPTH_OF_ONE)
        model = cfg.build()
        self.assertIsInstance(model, DepthMappingLayer)

    def test_build_with_overrides(self):
        cfg = self.preset(generator_depth=DynamicDepthOptions.DEPTH_OF_ONE)
        overrides = self.preset(
            input_dim=8,
            output_dim=4,
            bias_flag=False,
            generator_depth=DynamicDepthOptions.DEPTH_OF_TWO,
        )
        model = cfg.build(overrides)
        self.assertIsInstance(model, DepthMappingLayer)
        self.assertEqual(model.input_dim, 8)
        self.assertEqual(model.output_dim, 4)
        self.assertFalse(model.bias_flag)
        self.assertEqual(model.depth_value, DynamicDepthOptions.DEPTH_OF_TWO.value)

    def test_disabled_depth_raises_error(self):
        cfg = self.preset(generator_depth=DynamicDepthOptions.DISABLED)
        with self.assertRaises(ValueError):
            DepthMappingLayer(cfg)


class TestDepthMappingLayerStack(unittest.TestCase):
    def preset(
        self,
        input_dim: int = 12,
        hidden_dim: int = 24,
        output_dim: int = 6,
        bias_flag: bool = True,
        layer_norm_position: LayerNormPositionOptions = LayerNormPositionOptions.DISABLED,
        stack_num_layers: int = 2,
        stack_activation: ActivationOptions = ActivationOptions.RELU,
        stack_residual_flag: bool = False,
        stack_dropout_probability: float = 0.2,
        shared_halting_flag: bool = False,
        last_layer_bias_option: LastLayerBiasOptions = LastLayerBiasOptions.DEFAULT,
        apply_output_pipeline_flag: bool = True,
        gate_config: "LayerStackConfig | None" = None,
        halting_config: "StickBreakingConfig | None" = None,
        generator_depth: DynamicDepthOptions = DynamicDepthOptions.DEPTH_OF_TWO,
    ) -> DepthMappingHandlerConfig:
        return DepthMappingHandlerConfig(
            generator_depth=generator_depth,
            model_config=LayerStackConfig(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                num_layers=stack_num_layers,
                last_layer_bias_option=last_layer_bias_option,
                apply_output_pipeline_flag=apply_output_pipeline_flag,
                layer_config=LayerConfig(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    activation=stack_activation,
                    layer_norm_position=layer_norm_position,
                    residual_flag=stack_residual_flag,
                    dropout_probability=stack_dropout_probability,
                    gate_config=gate_config,
                    halting_config=halting_config,
                    shared_halting_flag=shared_halting_flag,
                    layer_model_config=LinearLayerConfig(
                        input_dim=input_dim,
                        output_dim=output_dim,
                        bias_flag=bias_flag,
                    ),
                ),
            ),
        )

    def test_forward_shape(self):
        batch_size = 2
        input_dim = 12
        output_dim = 6
        valid_depths = [
            DynamicDepthOptions.DEPTH_OF_ONE,
            DynamicDepthOptions.DEPTH_OF_TWO,
            DynamicDepthOptions.DEPTH_OF_THREE,
        ]
        valid_num_layers = [1, 2, 3]
        pipeline_flags = [True, False]
        for depth in valid_depths:
            for num_layers in valid_num_layers:
                for pipeline_flag in pipeline_flags:
                    with self.subTest(
                        depth=depth, num_layers=num_layers, pipeline_flag=pipeline_flag
                    ):
                        cfg = self.preset(
                            input_dim=input_dim,
                            output_dim=output_dim,
                            generator_depth=depth,
                            stack_num_layers=num_layers,
                            apply_output_pipeline_flag=pipeline_flag,
                        )
                        model = DepthMappingLayerStack(cfg)
                        input_tensor = torch.randn(batch_size, input_dim)
                        output = model(input_tensor)
                        self.assertEqual(
                            output.shape, (batch_size, depth.value, output_dim)
                        )

    def test_forward_shape_with_residual_connections(self):
        batch_size = 2
        dim = 12
        valid_depths = [
            DynamicDepthOptions.DEPTH_OF_ONE,
            DynamicDepthOptions.DEPTH_OF_TWO,
            DynamicDepthOptions.DEPTH_OF_THREE,
        ]
        valid_num_layers = [1, 2, 3]
        pipeline_flags = [True, False]
        for depth in valid_depths:
            for num_layers in valid_num_layers:
                for pipeline_flag in pipeline_flags:
                    with self.subTest(
                        depth=depth, num_layers=num_layers, pipeline_flag=pipeline_flag
                    ):
                        cfg = self.preset(
                            input_dim=dim,
                            hidden_dim=dim,
                            output_dim=dim,
                            generator_depth=depth,
                            stack_num_layers=num_layers,
                            stack_residual_flag=True,
                            apply_output_pipeline_flag=pipeline_flag,
                        )
                        model = DepthMappingLayerStack(cfg)
                        input_tensor = torch.randn(batch_size, dim)
                        output = model(input_tensor)
                        self.assertEqual(output.shape, (batch_size, depth.value, dim))

    def test_gate_config_raises_error(self):
        gate_config = LayerStackConfig(
            input_dim=12,
            hidden_dim=24,
            output_dim=12,
            num_layers=2,
            last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
            apply_output_pipeline_flag=False,
            layer_config=LayerConfig(
                input_dim=12,
                output_dim=12,
                activation=ActivationOptions.RELU,
                layer_norm_position=LayerNormPositionOptions.DISABLED,
                residual_flag=False,
                dropout_probability=0.0,
                gate_config=None,
                halting_config=None,
                shared_halting_flag=False,
                layer_model_config=LinearLayerConfig(
                    input_dim=12,
                    output_dim=12,
                    bias_flag=True,
                ),
            ),
        )
        cfg = self.preset(gate_config=gate_config)
        with self.assertRaises(ValueError):
            DepthMappingLayerStack(cfg)

    def test_halting_config_raises_error(self):
        halting_config = StickBreakingConfig(
            input_dim=12,
            threshold=0.99,
            halting_dropout=0.0,
            hidden_state_mode=HaltingHiddenStateModeOptions.RAW,
            halting_gate_config=LayerStackConfig(
                input_dim=12,
                hidden_dim=12,
                output_dim=2,
                num_layers=2,
                last_layer_bias_option=LastLayerBiasOptions.DISABLED,
                apply_output_pipeline_flag=False,
                layer_config=LayerConfig(
                    input_dim=12,
                    output_dim=12,
                    activation=ActivationOptions.DISABLED,
                    layer_norm_position=LayerNormPositionOptions.DISABLED,
                    residual_flag=False,
                    dropout_probability=0.0,
                    gate_config=None,
                    halting_config=None,
                    shared_halting_flag=False,
                    layer_model_config=LinearLayerConfig(
                        input_dim=12,
                        output_dim=12,
                        bias_flag=True,
                    ),
                ),
            ),
        )
        cfg = self.preset(halting_config=halting_config)
        with self.assertRaises(ValueError):
            DepthMappingLayerStack(cfg)

    def test_shared_halting_flag_raises_error(self):
        cfg = self.preset(shared_halting_flag=True)
        with self.assertRaises(ValueError):
            DepthMappingLayerStack(cfg)

    def test_layer_config_replaced_with_depth_mapping(self):
        depth = DynamicDepthOptions.DEPTH_OF_TWO
        cfg = self.preset(generator_depth=depth)
        model = DepthMappingLayerStack(cfg)
        layer_model_config = model.model_config.layer_config.layer_model_config
        self.assertIsInstance(layer_model_config, DepthMappingLayerConfig)
        self.assertEqual(layer_model_config.generator_depth, depth)

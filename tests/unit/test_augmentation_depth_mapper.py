import copy
import unittest
from dataclasses import fields

import torch

from emperor.augmentations.adaptive_parameters import DynamicDepthOptions
from emperor.augmentations.adaptive_parameters._weights.depth_mapping import (
    DepthMappingHandlerConfig,
    DepthMappingLayer,
    DepthMappingLayerConfig,
    DepthMappingLayerStack,
)
from emperor.augmentations.adaptive_parameters._weights.validation import (
    DepthMappingValidator,
)
from emperor.halting import HaltingHiddenStateModeOptions, StickBreakingConfig
from emperor.layers import (
    ActivationOptions,
    GateConfig,
    LastLayerBiasOptions,
    LayerConfig,
    LayerGateOptions,
    LayerNormPositionOptions,
    LayerStackConfig,
    ResidualConfig,
    ResidualConnectionOptions,
)
from emperor.linears import LinearLayerConfig
from emperor.memory import GatedResidualDynamicMemoryConfig, MemoryPositionOptions


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

    def test_gradients_flow(self):
        batch_size = 2
        input_dim = 12
        output_dim = 6
        depth = DynamicDepthOptions.DEPTH_OF_TWO
        cfg = self.preset(
            input_dim=input_dim, output_dim=output_dim, generator_depth=depth
        )
        model = DepthMappingLayer(cfg)

        input_tensor = torch.randn(
            batch_size, depth.value, input_dim, requires_grad=True
        )
        output = model(input_tensor)
        output.sum().backward()

        for parameter in (input_tensor, model.weight_params, model.bias_params):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertTrue(torch.any(parameter.grad != 0))
        self.assertEqual(model.weight_params.grad.shape, model.weight_params.shape)

    def test_float64_state_round_trip_preserves_bias_contract_and_output(self):
        input_dim = 3
        output_dim = 2
        depth = DynamicDepthOptions.DEPTH_OF_TWO

        for bias_flag in (True, False):
            with self.subTest(bias_flag=bias_flag):
                torch.manual_seed(13)
                config = self.preset(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    bias_flag=bias_flag,
                    generator_depth=depth,
                )
                source = DepthMappingLayer(config).double()
                for state_value in source.state_dict().values():
                    self.assertEqual(state_value.dtype, torch.float64)
                    self.assertEqual(state_value.device.type, "cpu")
                input_tensor = torch.tensor(
                    [
                        [[1.0, 2.0, -1.0], [0.5, -2.0, 3.0]],
                        [[-1.0, 0.25, 2.0], [4.0, -0.5, 1.5]],
                    ],
                    dtype=torch.float64,
                )
                expected_keys = (
                    ("weight_params", "bias_params")
                    if bias_flag
                    else ("weight_params",)
                )

                source_output = source(input_tensor)
                model_state = copy.deepcopy(source.state_dict())
                torch.manual_seed(97)
                restored = DepthMappingLayer(config).double()
                incompatible = restored.load_state_dict(model_state, strict=True)
                restored_output = restored(input_tensor)

                self.assertEqual(incompatible.missing_keys, [])
                self.assertEqual(incompatible.unexpected_keys, [])
                self.assertTupleEqual(tuple(source.state_dict()), expected_keys)
                self.assertTupleEqual(
                    tuple(dict(source.named_parameters())),
                    expected_keys,
                )
                self.assertEqual(source_output.dtype, torch.float64)
                self.assertEqual(source_output.device.type, "cpu")
                torch.testing.assert_close(restored_output, source_output)


class TestDepthMappingLayerStack(unittest.TestCase):
    def preset(
        self,
        input_dim: int = 12,
        hidden_dim: int = 24,
        output_dim: int = 6,
        bias_flag: bool = True,
        layer_norm_position: LayerNormPositionOptions = (
            LayerNormPositionOptions.DISABLED
        ),
        stack_num_layers: int = 2,
        stack_activation: ActivationOptions = ActivationOptions.RELU,
        stack_residual_connection_option: ResidualConnectionOptions | None = None,
        stack_dropout_probability: float = 0.2,
        shared_halting_config: "StickBreakingConfig | None" = None,
        last_layer_bias_option: LastLayerBiasOptions = LastLayerBiasOptions.DEFAULT,
        apply_output_pipeline_flag: bool = True,
        shared_gate_config: "GateConfig | None" = None,
        gate_config: "GateConfig | None" = None,
        halting_config: "StickBreakingConfig | None" = None,
        memory_config: "GatedResidualDynamicMemoryConfig | None" = None,
        shared_memory_config: "GatedResidualDynamicMemoryConfig | None" = None,
        generator_depth: DynamicDepthOptions = DynamicDepthOptions.DEPTH_OF_TWO,
    ) -> DepthMappingHandlerConfig:
        return DepthMappingHandlerConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            generator_depth=generator_depth,
            model_config=LayerStackConfig(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                num_layers=stack_num_layers,
                last_layer_bias_option=last_layer_bias_option,
                apply_output_pipeline_flag=apply_output_pipeline_flag,
                shared_gate_config=shared_gate_config,
                shared_halting_config=shared_halting_config,
                shared_memory_config=shared_memory_config,
                layer_config=LayerConfig(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    activation=stack_activation,
                    layer_norm_position=layer_norm_position,
                    residual_config=None
                    if stack_residual_connection_option is None
                    else ResidualConfig(option=stack_residual_connection_option),
                    dropout_probability=stack_dropout_probability,
                    gate_config=gate_config,
                    halting_config=halting_config,
                    memory_config=memory_config,
                    layer_model_config=LinearLayerConfig(
                        input_dim=input_dim,
                        output_dim=output_dim,
                        bias_flag=bias_flag,
                    ),
                ),
            ),
        )

    def gate_config(self, dim: int = 12) -> GateConfig:
        return GateConfig(
            gate_dim=dim,
            model_config=LayerStackConfig(
                input_dim=dim,
                hidden_dim=dim,
                output_dim=dim,
                num_layers=1,
                last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
                apply_output_pipeline_flag=False,
                layer_config=LayerConfig(
                    input_dim=dim,
                    output_dim=dim,
                    activation=ActivationOptions.DISABLED,
                    layer_norm_position=LayerNormPositionOptions.DISABLED,
                    residual_config=None,
                    dropout_probability=0.0,
                    gate_config=None,
                    halting_config=None,
                    layer_model_config=LinearLayerConfig(
                        input_dim=dim,
                        output_dim=dim,
                        bias_flag=True,
                    ),
                ),
            ),
            option=LayerGateOptions.ADDITION,
            activation=ActivationOptions.DISABLED,
        )

    def memory_config(self, dim: int = 12) -> GatedResidualDynamicMemoryConfig:
        return GatedResidualDynamicMemoryConfig(
            input_dim=dim,
            output_dim=dim,
            memory_position_option=MemoryPositionOptions.AFTER_AFFINE,
            test_time_training_learning_rate=None,
            test_time_training_num_inner_steps=None,
            model_config=LayerStackConfig(
                input_dim=dim,
                hidden_dim=dim,
                output_dim=dim,
                num_layers=1,
                last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
                apply_output_pipeline_flag=False,
                layer_config=LayerConfig(
                    input_dim=dim,
                    output_dim=dim,
                    activation=ActivationOptions.DISABLED,
                    layer_norm_position=LayerNormPositionOptions.DISABLED,
                    residual_config=None,
                    dropout_probability=0.0,
                    gate_config=None,
                    halting_config=None,
                    layer_model_config=LinearLayerConfig(
                        input_dim=dim,
                        output_dim=dim,
                        bias_flag=True,
                    ),
                ),
            ),
        )

    def test_depth_mapping_validator_optional_fields_match_configs(self):
        config_fields = {
            field.name
            for config_cls in (DepthMappingLayerConfig, DepthMappingHandlerConfig)
            for field in fields(config_cls)
        }
        self.assertLessEqual(DepthMappingValidator.OPTIONAL_FIELDS, config_fields)

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
                            stack_residual_connection_option=ResidualConnectionOptions.RESIDUAL,
                            apply_output_pipeline_flag=pipeline_flag,
                        )
                        model = DepthMappingLayerStack(cfg)
                        input_tensor = torch.randn(batch_size, dim)
                        output = model(input_tensor)
                        self.assertEqual(output.shape, (batch_size, depth.value, dim))

    def test_gate_config_raises_error(self):
        cfg = self.preset(gate_config=self.gate_config())
        with self.assertRaises(ValueError):
            DepthMappingLayerStack(cfg)

    def test_shared_gate_config_raises_error(self):
        cfg = self.preset(shared_gate_config=self.gate_config())
        with self.assertRaises(ValueError):
            DepthMappingLayerStack(cfg)

    def test_halting_config_raises_error(self):
        halting_config = StickBreakingConfig(
            input_dim=12,
            threshold=0.99,
            dropout_probability=0.0,
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
                    residual_config=None,
                    dropout_probability=0.0,
                    gate_config=None,
                    halting_config=None,
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

    def test_shared_halting_config_raises_error(self):
        dim = 12
        shared_halting_config = StickBreakingConfig(
            input_dim=dim,
            threshold=0.99,
            dropout_probability=0.0,
            hidden_state_mode=HaltingHiddenStateModeOptions.RAW,
            halting_gate_config=LayerStackConfig(
                input_dim=dim,
                hidden_dim=dim,
                output_dim=2,
                num_layers=2,
                last_layer_bias_option=LastLayerBiasOptions.DISABLED,
                apply_output_pipeline_flag=False,
                layer_config=LayerConfig(
                    input_dim=dim,
                    output_dim=dim,
                    activation=ActivationOptions.DISABLED,
                    layer_norm_position=LayerNormPositionOptions.DISABLED,
                    residual_config=None,
                    dropout_probability=0.0,
                    gate_config=None,
                    halting_config=None,
                    layer_model_config=LinearLayerConfig(
                        input_dim=dim,
                        output_dim=dim,
                        bias_flag=True,
                    ),
                ),
            ),
        )
        cfg = self.preset(shared_halting_config=shared_halting_config)
        with self.assertRaises(ValueError):
            DepthMappingLayerStack(cfg)

    def test_memory_config_raises_error(self):
        cfg = self.preset(memory_config=self.memory_config())
        with self.assertRaises(ValueError):
            DepthMappingLayerStack(cfg)

    def test_shared_memory_config_raises_error(self):
        cfg = self.preset(shared_memory_config=self.memory_config())
        with self.assertRaises(ValueError):
            DepthMappingLayerStack(cfg)

    def test_layer_config_replaced_with_depth_mapping(self):
        depth = DynamicDepthOptions.DEPTH_OF_TWO
        cfg = self.preset(generator_depth=depth)
        model = DepthMappingLayerStack(cfg)
        layer_model_config = model.model_config.layer_config.layer_model_config
        self.assertIsInstance(layer_model_config, DepthMappingLayerConfig)
        self.assertEqual(layer_model_config.generator_depth, depth)

    def test_preconfigured_depth_mapping_config_is_synchronized_without_mutation(self):
        handler_depth = DynamicDepthOptions.DEPTH_OF_THREE
        configured_depth = DynamicDepthOptions.DEPTH_OF_ONE
        input_dim = 12
        output_dim = 6
        configured_input_dim = 8
        configured_output_dim = 4
        depth_mapping_config = DepthMappingLayerConfig(
            input_dim=configured_input_dim,
            output_dim=configured_output_dim,
            bias_flag=True,
            generator_depth=configured_depth,
        )
        cfg = self.preset(
            input_dim=input_dim,
            output_dim=output_dim,
            generator_depth=handler_depth,
        )
        source_stack_config = cfg.model_config
        source_stack_config.input_dim = configured_input_dim
        source_stack_config.output_dim = configured_output_dim
        cfg.model_config.layer_config.layer_model_config = depth_mapping_config

        model = DepthMappingLayerStack(cfg)

        result_depth_mapping_config = model.model_config.layer_config.layer_model_config
        self.assertIsNot(model.model_config, source_stack_config)
        self.assertIsNot(result_depth_mapping_config, depth_mapping_config)
        self.assertEqual(model.model_config.input_dim, input_dim)
        self.assertEqual(model.model_config.output_dim, output_dim)
        self.assertEqual(result_depth_mapping_config.generator_depth, handler_depth)
        self.assertEqual(source_stack_config.input_dim, configured_input_dim)
        self.assertEqual(source_stack_config.output_dim, configured_output_dim)
        self.assertEqual(depth_mapping_config.generator_depth, configured_depth)

        output = model(torch.randn(2, input_dim))
        self.assertEqual(output.shape, (2, handler_depth.value, output_dim))

    def test_preconfigured_depth_mapping_config_does_not_skip_validation(self):
        cfg = self.preset(
            gate_config=self.gate_config(),
            generator_depth=DynamicDepthOptions.DEPTH_OF_TWO,
        )
        cfg.model_config.layer_config.layer_model_config = DepthMappingLayerConfig(
            input_dim=cfg.input_dim,
            output_dim=cfg.output_dim,
            bias_flag=True,
            generator_depth=cfg.generator_depth,
        )

        with self.assertRaises(ValueError):
            DepthMappingLayerStack(cfg)

    def test_handler_config_build_creates_depth_mapping_layer_stack(self):
        cfg = self.preset(input_dim=8, output_dim=4)
        model = cfg.build()

        self.assertIsInstance(model, DepthMappingLayerStack)
        self.assertEqual(model.input_dim, 8)
        self.assertEqual(model.output_dim, 4)

    def test_gradients_flow(self):
        batch_size = 2
        input_dim = 12
        depth = DynamicDepthOptions.DEPTH_OF_TWO
        cfg = self.preset(
            input_dim=input_dim,
            generator_depth=depth,
            stack_activation=ActivationOptions.DISABLED,
            stack_dropout_probability=0.0,
            apply_output_pipeline_flag=False,
        )
        model = DepthMappingLayerStack(cfg)

        input_tensor = torch.randn(batch_size, input_dim, requires_grad=True)
        output = model(input_tensor)
        output.square().sum().backward()

        self.assertIsNotNone(input_tensor.grad)
        self.assertTrue(torch.isfinite(input_tensor.grad).all())
        self.assertTrue(torch.any(input_tensor.grad != 0))
        gradients = [parameter.grad for parameter in model.parameters()]
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
        self.assertTrue(all(torch.any(gradient != 0) for gradient in gradients))

    def test_float64_noncontiguous_input_and_strict_stack_state_round_trip(self):
        input_dim = 3
        output_dim = 2
        depth = DynamicDepthOptions.DEPTH_OF_TWO
        config = self.preset(
            input_dim=input_dim,
            hidden_dim=4,
            output_dim=output_dim,
            generator_depth=depth,
            stack_num_layers=2,
            stack_activation=ActivationOptions.DISABLED,
            stack_dropout_probability=0.0,
            apply_output_pipeline_flag=False,
        )
        torch.manual_seed(29)
        source = DepthMappingLayerStack(config).double().eval()
        for state_value in source.state_dict().values():
            if torch.is_floating_point(state_value):
                self.assertEqual(state_value.dtype, torch.float64)
                self.assertEqual(state_value.device.type, "cpu")
        input_tensor = torch.tensor(
            [
                [1.0, -2.0],
                [0.5, 3.0],
                [-1.0, 4.0],
            ],
            dtype=torch.float64,
        ).transpose(0, 1)
        self.assertFalse(input_tensor.is_contiguous())

        source_output = source(input_tensor)
        model_state = copy.deepcopy(source.state_dict())
        torch.manual_seed(71)
        restored = DepthMappingLayerStack(config).double().eval()
        incompatible = restored.load_state_dict(model_state, strict=True)
        restored_output = restored(input_tensor)

        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])
        self.assertTupleEqual(
            tuple(source.state_dict()),
            tuple(restored.state_dict()),
        )
        self.assertTupleEqual(
            tuple(dict(source.named_parameters())),
            tuple(dict(restored.named_parameters())),
        )
        self.assertEqual(source_output.shape, (2, depth.value, output_dim))
        self.assertEqual(source_output.dtype, torch.float64)
        self.assertEqual(source_output.device.type, "cpu")
        self.assertTrue(torch.isfinite(source_output).all())
        torch.testing.assert_close(restored_output, source_output)

    def test_non_2d_input_raises_error(self):
        cfg = self.preset(generator_depth=DynamicDepthOptions.DEPTH_OF_TWO)
        model = DepthMappingLayerStack(cfg)
        input_tensor = torch.randn(2, 4, 12)
        with self.assertRaises(ValueError):
            model(input_tensor)

    def test_non_tensor_input_raises_error(self):
        cfg = self.preset(generator_depth=DynamicDepthOptions.DEPTH_OF_TWO)
        model = DepthMappingLayerStack(cfg)
        with self.assertRaises(TypeError):
            model([[0.0] * cfg.input_dim])

    def test_wrong_input_feature_dimension_raises_error(self):
        cfg = self.preset(generator_depth=DynamicDepthOptions.DEPTH_OF_TWO)
        model = DepthMappingLayerStack(cfg)
        input_tensor = torch.randn(2, cfg.input_dim - 1)
        with self.assertRaises(ValueError):
            model(input_tensor)

    def test_disabled_depth_raises_error(self):
        cfg = self.preset(generator_depth=DynamicDepthOptions.DISABLED)
        with self.assertRaises(ValueError):
            DepthMappingLayerStack(cfg)

    def test_invalid_layer_model_config_type_raises_error(self):
        from emperor.config import ConfigBase

        cfg = self.preset(generator_depth=DynamicDepthOptions.DEPTH_OF_TWO)
        cfg.model_config.layer_config.layer_model_config = ConfigBase()
        with self.assertRaises(TypeError):
            DepthMappingLayerStack(cfg)

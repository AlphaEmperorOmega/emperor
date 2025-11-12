import torch
import unittest
import torch.nn as nn

from dataclasses import asdict
from Emperor.base.enums import ActivationOptions
from docs.config import default_unittest_config
from Emperor.generators.utils.base import (
    Layer,
    LayerStack,
    LayerStackConfig,
    LayerStackAdjustments,
)
from Emperor.generators.utils.enums import (
    LinearLayerTypes,
    ParameterGeneratorTypes,
)


class TestLayerStack(unittest.TestCase):
    def setUp(self):
        self.rebuild_presets()

    def tearDown(self):
        self.cfg = None
        self.config = None
        self.model = None
        self.batch_size = None
        self.embedding_dim = None
        self.target_sequence_length = None
        self.num_heads = None
        self.head_dim = None
        self.query_model = None
        self.key_model = None
        self.value_model = None
        self.qkv_model = None

    def rebuild_presets(self, config: LayerStackConfig | None = None):
        self.cfg = default_unittest_config()
        self.config = self.cfg.layer_block_stack_config
        if config is not None:
            for k in asdict(config):
                if hasattr(self.config, k) and getattr(config, k) is not None:
                    setattr(self.config, k, getattr(config, k))

        self.model = LayerStack(self.cfg)

        self.batch_size = self.cfg.batch_size

        self.input_dim = self.config.input_dim
        self.hidden_dim = self.config.hidden_dim
        self.output_dim = self.config.output_dim
        self.num_layers = self.config.num_layers

        self.activation = self.config.activation
        self.layer_norm_dim = self.config.layer_norm_dim
        self.residual_flag = self.config.residual_flag
        self.adaptive_computation_flag = self.config.adaptive_computation_flag
        self.dropout_probability = self.config.dropout_probability
        self.layer_norm_position = self.config.layer_norm_position


class Test__init(TestLayerStack):
    def test_init_input_layer_with_default_config(self):
        self.assertIsInstance(self.model, LayerStack)

        self.assertEqual(self.model.model_type, self.config.model_type)
        self.assertEqual(self.model.activation, self.config.activation)
        self.assertEqual(self.model.residual_flag, self.config.residual_flag)
        self.assertEqual(
            self.model.adaptive_computation_flag, self.config.adaptive_computation_flag
        )
        self.assertEqual(
            self.model.dropout_probability, self.config.dropout_probability
        )
        self.assertEqual(
            self.model.layer_norm_position, self.config.layer_norm_position
        )

        self.assertEqual(self.model.input_dim, self.config.input_dim)
        self.assertEqual(self.model.hidden_dim, self.config.hidden_dim)
        self.assertEqual(self.model.output_dim, self.config.output_dim)
        self.assertEqual(self.model.num_layers, self.config.num_layers)


class Test___resolve_model_type_overrides(TestLayerStack):
    def test_if_config_is_updated_for_linear_model(self):
        config = LayerStackConfig(
            model_type=LinearLayerTypes.DYNAMIC,
        )
        self.rebuild_presets(config)

        input_dim = 8
        output_dim = 16

        updated_config = self.model._LayerStack__resolve_model_type_overrides(
            input_dim, output_dim
        )
        self.assertEqual(updated_config.linear_layer_model_config.input_dim, input_dim)
        self.assertEqual(
            updated_config.linear_layer_model_config.output_dim, output_dim
        )

    def test_if_config_is_updated_for_parameter_generator_model(self):
        config = LayerStackConfig(
            model_type=ParameterGeneratorTypes.VECTOR,
        )
        self.rebuild_presets(config)

        input_dim = 8
        output_dim = 16

        updated_config = self.model._LayerStack__resolve_model_type_overrides(
            input_dim, output_dim
        )

        self.assertEqual(updated_config.router_model_config.input_dim, input_dim)
        self.assertEqual(updated_config.mixture_model_config.input_dim, input_dim)
        self.assertEqual(updated_config.mixture_model_config.output_dim, output_dim)


class Test___create_layer(TestLayerStack):
    def test_if_all_model_types_are_computed(self):
        types = (LinearLayerTypes, ParameterGeneratorTypes)
        for type in types:
            for layer_type in type:
                config = LayerStackConfig(
                    model_type=layer_type,
                )
                self.rebuild_presets(config)

                model = self.model._LayerStack__create_layer(
                    self.input_dim, self.output_dim, self.residual_flag
                )

                input = torch.randn(self.batch_size, self.input_dim)
                output = model(input)
                if isinstance(output, tuple):
                    output, _ = model(input)

                expected_output = (self.batch_size, self.output_dim)

                self.assertIsInstance(model, Layer)
                self.assertIsInstance(model.model, layer_type.value)
                self.assertEqual(
                    model.activation_function, ActivationOptions.GELU.value
                )
                self.assertEqual(model.layer_norm_dim, None)
                self.assertEqual(model.residual_connection_flag, self.residual_flag)
                self.assertEqual(
                    model.is_adaptive_computation, self.adaptive_computation_flag
                )
                self.assertEqual(model.dropout_probability, self.dropout_probability)
                self.assertEqual(output.shape, expected_output)

    def test_if_all_model_types_are_computed_with_true_residual_flag(self):
        types = (LinearLayerTypes, ParameterGeneratorTypes)
        for type in types:
            for layer_type in type:
                config = LayerStackConfig(
                    input_dim=8,
                    output_dim=8,
                    residual_flag=True,
                    model_type=layer_type,
                )
                self.rebuild_presets(config)

                model = self.model._LayerStack__create_layer(
                    self.input_dim, self.output_dim, residual_flag=True
                )

                input = torch.randn(self.batch_size, self.input_dim)
                output = model(input)
                if isinstance(output, tuple):
                    output, _ = model(input)

                expected_output = (self.batch_size, self.output_dim)

                self.assertIsInstance(model, Layer)
                self.assertIsInstance(model.model, layer_type.value)
                self.assertEqual(
                    model.activation_function, ActivationOptions.GELU.value
                )
                self.assertEqual(model.layer_norm_dim, None)
                self.assertEqual(model.residual_connection_flag, self.residual_flag)
                self.assertEqual(
                    model.is_adaptive_computation, self.adaptive_computation_flag
                )
                self.assertEqual(model.dropout_probability, self.dropout_probability)
                self.assertEqual(output.shape, expected_output)


class Test___add_initial_layer(TestLayerStack):
    def test_no_layer_is_added_when_num_layers_is_one(self):
        types = (LinearLayerTypes, ParameterGeneratorTypes)
        for type in types:
            for layer_type in type:
                config = LayerStackConfig(
                    input_dim=8,
                    output_dim=16,
                    num_layers=1,
                    model_type=layer_type,
                )
                self.rebuild_presets(config)

                model_list = []
                adjustment = self.model._LayerStack__add_initial_layer(model_list)

                self.assertEqual(len(model_list), 0)
                self.assertEqual(
                    adjustment, LayerStackAdjustments.SHARED_INPUT_OUTPUT_DIM
                )

    def test_no_layer_is_added_when_num_layers_is_three_with_same_input_output_dim(
        self,
    ):
        types = (LinearLayerTypes, ParameterGeneratorTypes)
        for type in types:
            for layer_type in type:
                config = LayerStackConfig(
                    input_dim=16,
                    output_dim=16,
                    num_layers=1,
                    model_type=layer_type,
                )
                self.rebuild_presets(config)

                model_list = []
                adjustment = self.model._LayerStack__add_initial_layer(model_list)

                self.assertEqual(len(model_list), 0)
                self.assertEqual(
                    adjustment, LayerStackAdjustments.SHARED_INPUT_OUTPUT_DIM
                )

    def test_initial_layer_is_added_with_multiple_layers_and_different_input_output_dim(
        self,
    ):
        num_layers_array = [2, 3, 4]
        types = (LinearLayerTypes, ParameterGeneratorTypes)
        for type in types:
            for layer_type in type:
                for num_layer in num_layers_array:
                    config = LayerStackConfig(
                        input_dim=8,
                        output_dim=16,
                        num_layers=num_layer,
                        model_type=layer_type,
                    )
                    self.rebuild_presets(config)

                    model_list = []
                    adjustment = self.model._LayerStack__add_initial_layer(
                        model_list
                    )
                    model = nn.Sequential(*model_list)

                    input = torch.randn(self.batch_size, self.input_dim)
                    output = model(input)
                    if isinstance(output, tuple):
                        output, _ = model(input)

                    expected_output = (self.batch_size, self.hidden_dim)

                    self.assertEqual(
                        adjustment, LayerStackAdjustments.SEPARATE_INPUT_OUTPUT_DIM
                    )
                    self.assertEqual(len(model_list), 1)
                    self.assertEqual(output.shape, expected_output)


class Test___add_hidden_layers(TestLayerStack):
    def test_no_layer_added_when_num_layers_is_one(self):
        types = (LinearLayerTypes, ParameterGeneratorTypes)
        for type in types:
            for layer_type in type:
                config = LayerStackConfig(
                    input_dim=8,
                    output_dim=16,
                    num_layers=1,
                    model_type=layer_type,
                )
                self.rebuild_presets(config)

                model_list = []
                adjustment = LayerStackAdjustments.SHARED_INPUT_OUTPUT_DIM
                self.model._LayerStack__add_hidden_layers(model_list, adjustment)

                self.assertEqual(len(model_list), 0)

    def test_no_layer_added_when_num_layers_is_two_with_initial_layer_added(self):
        types = (LinearLayerTypes, ParameterGeneratorTypes)
        for type in types:
            for layer_type in type:
                config = LayerStackConfig(
                    input_dim=8,
                    output_dim=16,
                    num_layers=2,
                    model_type=layer_type,
                )
                self.rebuild_presets(config)

                model_list = []
                adjustment = LayerStackAdjustments.SEPARATE_INPUT_OUTPUT_DIM
                self.model._LayerStack__add_hidden_layers(model_list, adjustment)

                self.assertEqual(len(model_list), 0)

    def test_hidden_layers_are_added_multiple_hidden_layers(self):
        num_layers_array = [2, 3, 4]
        types = (LinearLayerTypes, ParameterGeneratorTypes)
        for type in types:
            for layer_type in type:
                for num_layer in num_layers_array:
                    config = LayerStackConfig(
                        input_dim=8,
                        output_dim=16,
                        num_layers=num_layer,
                        model_type=layer_type,
                    )
                    self.rebuild_presets(config)

                    model_list = []

                    adjustment = LayerStackAdjustments.SHARED_INPUT_OUTPUT_DIM
                    self.model._LayerStack__add_hidden_layers(
                        model_list, adjustment
                    )
                    model = nn.Sequential(*model_list)

                    input = torch.randn(self.batch_size, self.hidden_dim)
                    output = model(input)
                    if isinstance(output, tuple):
                        output, _ = model(input)

                    expected_output = (self.batch_size, self.hidden_dim)

                    self.assertEqual(len(model_list), num_layer - adjustment.value)
                    self.assertEqual(output.shape, expected_output)


class Test___add_output_layer(TestLayerStack):
    def test_ensure_input_layer_is_returned_when_num_layers_is_one(self):
        types = (LinearLayerTypes, ParameterGeneratorTypes)
        for type in types:
            for layer_type in type:
                config = LayerStackConfig(
                    input_dim=8,
                    output_dim=16,
                    num_layers=1,
                    model_type=layer_type,
                )
                self.rebuild_presets(config)

                model_list = []
                self.model._LayerStack__add_output_layer(model_list)
                model = model_list[0]

                input = torch.randn(self.batch_size, self.input_dim)
                output = model(input)
                if isinstance(output, tuple):
                    output, _ = model(input)

                expected_output = (self.batch_size, self.output_dim)

                self.assertEqual(len(model_list), 1)
                self.assertEqual(output.shape, expected_output)

    def test_ensure_output_layer_is_added_when_multiple_layers_are_added(self):
        types = (LinearLayerTypes, ParameterGeneratorTypes)
        for type in types:
            for layer_type in type:
                config = LayerStackConfig(
                    input_dim=8,
                    output_dim=16,
                    num_layers=2,
                    model_type=layer_type,
                )
                self.rebuild_presets(config)

                model_list = []
                self.model._LayerStack__add_output_layer(model_list)
                model = model_list[0]

                input = torch.randn(self.batch_size, self.hidden_dim)
                output = model(input)
                if isinstance(output, tuple):
                    output, _ = model(input)

                expected_output = (self.batch_size, self.output_dim)

                self.assertEqual(len(model_list), 1)
                self.assertEqual(output.shape, expected_output)


class Test_build_model(TestLayerStack):
    def test_model_layer_block_returned_when_num_layers_is_one(self):
        types = (LinearLayerTypes, ParameterGeneratorTypes)
        for type in types:
            for layer_type in type:
                config = LayerStackConfig(
                    input_dim=8,
                    output_dim=16,
                    num_layers=1,
                    model_type=layer_type,
                )
                self.rebuild_presets(config)

                model = self.model.build_model()

                input = torch.randn(self.batch_size, self.input_dim)
                output = model(input)
                if isinstance(output, tuple):
                    output, _ = model(input)

                expected_output = (self.batch_size, self.output_dim)

                self.assertIsInstance(model, Layer)
                self.assertEqual(output.shape, expected_output)

    def test_sequential_is_returned_when_num_layers_is_greater_than_one(self):
        num_layers_array = [2, 3, 4]

        types = (LinearLayerTypes, ParameterGeneratorTypes)
        for type in types:
            for layer_type in type:
                for num_layers in num_layers_array:
                    config = LayerStackConfig(
                        input_dim=8,
                        output_dim=16,
                        num_layers=num_layers,
                        model_type=layer_type,
                    )
                    self.rebuild_presets(config)

                    model = self.model.build_model()

                    input = torch.randn(self.batch_size, self.input_dim)
                    output = model(input)
                    if isinstance(output, tuple):
                        output, _ = model(input)

                    expected_output = (self.batch_size, self.output_dim)

                    self.assertIsInstance(model, nn.Sequential)
                    self.assertEqual(output.shape, expected_output)

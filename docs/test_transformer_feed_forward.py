import torch
import unittest

from dataclasses import asdict
from Emperor.feedForward.feed_forward import FeedForward, FeedForwardConfig
from Emperor.generators.utils.enums import LinearLayerTypes, ParameterGeneratorTypes
from docs.config import default_unittest_config


class TestFeedForward(unittest.TestCase):
    def setUp(self):
        self.rebuild_presets()

    def tearDown(self):
        self.cfg = None
        self.config = None
        self.model = None
        self.batch_size = None
        self.input_dim = None
        self.output_dim = None

    def rebuild_presets(self, config: FeedForwardConfig | None = None):
        self.cfg = default_unittest_config()
        self.config = self.cfg.transformer_feed_forward_config
        if config is not None:
            for k in asdict(config):
                if hasattr(self.config, k) and getattr(config, k) is not None:
                    setattr(self.config, k, getattr(config, k))

        self.model = FeedForward(self.cfg)

        self.batch_size = self.cfg.batch_size
        first_layer_of_sequential = self.model.model[0].model
        self.input_dim = first_layer_of_sequential.input_dim
        layer_layer_of_sequential = self.model.model[-1].model
        self.output_dim = layer_layer_of_sequential.output_dim


class Test___init(TestFeedForward):
    def test_init_input_layer_with_default_config(self):
        config = FeedForwardConfig(
            num_layers=2,
        )
        self.rebuild_presets(config)
        self.assertIsInstance(self.model, FeedForward)
        self.assertEqual(self.model.num_layers, self.config.num_layers)
        self.assertEqual(len(self.model.model), self.config.num_layers)

    def test_ensure_error_is_triggered_if_num_layers_is_one(self):
        config = FeedForwardConfig(
            num_layers=1,
        )
        with self.assertRaises(RuntimeError):
            self.rebuild_presets(config)

    def test_differet_layer_types(self):
        num_layers_list = [2, 4, 6]
        types = (LinearLayerTypes, ParameterGeneratorTypes)
        for type in types:
            for layer_type in type:
                for num_layers in num_layers_list:
                    config = FeedForwardConfig(
                        model_type=layer_type,
                        num_layers=num_layers,
                    )
                    self.rebuild_presets(config)
                    self.assertIsInstance(self.model.model[0].model, layer_type.value)


class Test_forward(TestFeedForward):
    def test_ensure_the_feed_forward_model_processes_2D_input_batch(self):
        num_layers_list = [2, 4, 6]

        types = (LinearLayerTypes, ParameterGeneratorTypes)
        for type in types:
            for layer_type in type:
                for num_layers in num_layers_list:
                    config = FeedForwardConfig(
                        model_type=layer_type,
                        num_layers=num_layers,
                    )
                    self.rebuild_presets(config)
                    input = torch.randn(self.batch_size, self.input_dim)
                    output = self.model(input)

                    expected_output = (self.batch_size, self.output_dim)

                    if isinstance(output, tuple):
                        output, _ = output

                    self.assertIsInstance(self.model.model[0].model, layer_type.value)
                    self.assertEqual(output.shape, expected_output)

    def test_ensure_the_feed_forward_model_processes_3D_input_batch(self):
        num_layers_list = [2, 4, 6]
        types = (LinearLayerTypes, ParameterGeneratorTypes)
        for type in types:
            for layer_type in type:
                for num_layers in num_layers_list:
                    config = FeedForwardConfig(
                        model_type=layer_type,
                        num_layers=num_layers,
                    )
                    self.rebuild_presets(config)
                    sequence_length = 4
                    input = torch.randn(
                        self.batch_size, sequence_length, self.input_dim
                    )
                    output = self.model(input)
                    output, _ = output
                    expected_output = (
                        self.batch_size,
                        sequence_length,
                        self.output_dim,
                    )

                    self.assertIsInstance(self.model.model[0].model, layer_type.value)
                    self.assertEqual(output.shape, expected_output)

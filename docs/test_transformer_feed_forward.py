import torch
import unittest

from Emperor.transformer.presets import TransformerPresets
from Emperor.linears.options import LinearLayerStackOptions
from Emperor.transformer.utils.feed_forward import FeedForward
from Emperor.adaptive.options import AdaptiveLayerStackOptions


class TestFeedForward(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        self.cfg = None
        self.config = None
        self.model = None
        self.batch_size = None
        self.input_dim = None
        self.output_dim = None

    def test_init_with_valid_inputs(self):
        num_layers_list = [2, 4, 6]
        stack_options = [LinearLayerStackOptions, AdaptiveLayerStackOptions]
        for stack_type in stack_options:
            for model_type in stack_type:
                for num_layers in num_layers_list:
                    message = f"Testing configuration with model_type={model_type}, num_layers={num_layers}"
                    with self.subTest(msg=message):
                        c = TransformerPresets.transformer_feed_forward_preset(
                            layer_stack_option=model_type,
                            num_layers=num_layers,
                        )
                        m = FeedForward(c)
                        self.assertIsInstance(m, FeedForward)
                        self.assertEqual(m.num_layers, m.num_layers)
                        self.assertEqual(len(m.model), num_layers)

    def test_init_with_invalid_inputs(self):
        num_layers_list = [1, 3, 5]
        for num_layers in num_layers_list:
            message = f"Testing configuration with num_layers={num_layers}"
            with self.subTest(msg=message):
                c = TransformerPresets.transformer_feed_forward_preset(
                    num_layers=num_layers,
                )
                with self.assertRaises(RuntimeError):
                    m = FeedForward(c)

    def test_forward(self):
        num_layers_list = [2, 4, 6]
        flag_options = [True, False]
        stack_options = [LinearLayerStackOptions, AdaptiveLayerStackOptions]
        for stack_type in stack_options:
            for model_type in stack_type:
                for num_layers in num_layers_list:
                    for matrix_input_flag in flag_options:
                        message = f"Testing FeedForward configuration with model_type={model_type}, num_layers={num_layers}, matrix_input_flag={matrix_input_flag}"
                        with self.subTest(msg=message):
                            c = TransformerPresets.transformer_feed_forward_preset(
                                layer_stack_option=model_type,
                                num_layers=num_layers,
                            )
                            m = FeedForward(c)
                            batch_size = 8
                            sequence_length = 4
                            if matrix_input_flag:
                                input = torch.randn(batch_size, c.input_dim)
                            else:
                                input = torch.randn(
                                    batch_size, sequence_length, c.input_dim
                                )
                            output, loss = m(input)

                            if matrix_input_flag:
                                expected_output = (batch_size, c.output_dim)
                            else:
                                expected_output = (
                                    batch_size,
                                    sequence_length,
                                    c.output_dim,
                                )

                            if isinstance(output, tuple):
                                output, _ = output

                            self.assertEqual(output.shape, expected_output)

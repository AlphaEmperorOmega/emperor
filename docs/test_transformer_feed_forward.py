import torch
import unittest

from emperor.experts.utils.enums import InitSamplerOptions
from emperor.linears.options import LinearLayerStackOptions
from emperor.transformer.utils.feed_forward import FeedForward
from emperor.adaptive.options import AdaptiveLayerStackOptions
from emperor.experts.options import MixtureOfExpertsStackOptions
from emperor.transformer.utils.presets import TransformerPresets


class TestFeedForward(unittest.TestCase):
    def test_init_with_valid_inputs(self):
        num_layers_list = [2, 4, 6]
        stack_options = [
            LinearLayerStackOptions,
            AdaptiveLayerStackOptions,
            MixtureOfExpertsStackOptions,
        ]
        expert_sampler_options = [
            InitSamplerOptions.LAYER,
            InitSamplerOptions.SHARED,
        ]

        for expert_sampler_option in expert_sampler_options:
            for stack_type in stack_options:
                for model_type in stack_type:
                    for num_layers in num_layers_list:
                        message = f"Testing configuration with model_type={model_type}, num_layers={num_layers}"
                        options = {}
                        if model_type == MixtureOfExpertsStackOptions.BASE:
                            options = {
                                "experts_init_sampler_option": expert_sampler_option,
                            }
                        with self.subTest(msg=message):
                            c = TransformerPresets.transformer_feed_forward_preset(
                                layer_stack_option=model_type,
                                num_layers=num_layers,
                                **options,
                            )
                            m = FeedForward(c)
                            self.assertIsInstance(m, FeedForward)
                            self.assertEqual(m.num_layers, m.num_layers)

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
        stack_options = [
            LinearLayerStackOptions,
            AdaptiveLayerStackOptions,
            MixtureOfExpertsStackOptions,
        ]
        expert_sampler_options = [
            InitSamplerOptions.LAYER,
            InitSamplerOptions.SHARED,
        ]
        for expert_sampler_option in expert_sampler_options:
            for stack_type in stack_options:
                for model_type in stack_type:
                    for num_layers in num_layers_list:
                        for matrix_input_flag in flag_options:
                            message = f"Testing FeedForward configuration with model_type={model_type}, num_layers={num_layers}, matrix_input_flag={matrix_input_flag}"
                            with self.subTest(msg=message):
                                options = {}
                                if model_type == MixtureOfExpertsStackOptions.BASE:
                                    options = {
                                        "experts_init_sampler_option": expert_sampler_option,
                                    }
                                c = TransformerPresets.transformer_feed_forward_preset(
                                    layer_stack_option=model_type,
                                    num_layers=num_layers,
                                    **options,
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

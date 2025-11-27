import torch
import unittest

from dataclasses import asdict
from Emperor.config import ModelConfig
from Emperor.linears.utils.config import LinearsConfigs
from Emperor.linears.utils.enums import (
    DynamicBiasOptions,
    DynamicDepthOptions,
    DynamicDiagonalOptions,
    LinearMemoryOptions,
    LinearMemoryPositionOptions,
    LinearMemorySizeOptions,
)
from Emperor.linears.utils.layers import (
    DynamicLinearLayer,
    LinearLayer,
    LinearLayerConfig,
)


class TestLinears(unittest.TestCase):
    def setUp(self):
        self.rebuild_presets()

    def tearDown(self):
        self.cfg = None
        self.config = None
        self.batch_size = None
        self.input_dim = None
        self.output_dim = None

    def rebuild_presets(self, config: ModelConfig | None = None):
        self.cfg = LinearsConfigs.base_preset() if config is None else config
        self.config = self.cfg.linear_layer_config
        if config is not None:
            for k in asdict(config):
                if hasattr(self.config, k) and getattr(config, k) is not None:
                    setattr(self.config, k, getattr(config, k))

        self.batch_size = self.cfg.batch_size
        self.input_dim = self.cfg.input_dim
        self.output_dim = self.cfg.output_dim


class TestLinearLayer(TestLinears):
    def test_init_with_different_configation_options(self):
        bias_options = [True, False]
        for bias_flag in bias_options:
            message = f"Test failed for the inputs: {bias_flag}"
            with self.subTest(i=message):
                c = LinearsConfigs.base_preset(bias_flag=bias_flag)
                overrides = LinearLayerConfig(bias_flag=bias_flag)
                m = LinearLayer(self.cfg, overrides)

                self.assertEqual(m.input_dim, c.input_dim)
                self.assertEqual(m.output_dim, c.output_dim)
                self.assertIsInstance(m.weight_params, torch.Tensor)
                if bias_flag:
                    self.assertIsInstance(m.bias_params, torch.Tensor)
                else:
                    self.assertIsNone(m.bias_params)

    def test_forward(self):
        bias_options = [True, False]
        input_params = output_params = [4, 8, 16]

        for input_dim in input_params:
            for output_dim in output_params:
                for bias_flag in bias_options:
                    message = f"Test failed for the options: {input_dim}, {output_dim}, {bias_flag}"
                    with self.subTest(i=message):
                        c = LinearsConfigs.base_preset()
                        overrides = LinearLayerConfig(
                            input_dim=input_dim,
                            output_dim=output_dim,
                            bias_flag=bias_flag,
                        )
                        m = LinearLayer(c, overrides)

                        input_batch = torch.randn(c.batch_size, overrides.input_dim)
                        output = m.forward(input_batch)
                        expected_output_shape = (c.batch_size, overrides.output_dim)
                        self.assertEqual(output.shape, expected_output_shape)


class TestDynamicLinearLayer(TestLinears):
    def test_init_with_different_configation_options(self):
        bias_options = [True, False]

        for bias_flag in bias_options:
            for generators_depth in DynamicDepthOptions:
                for diagonal_option in DynamicDiagonalOptions:
                    for bias_option in DynamicBiasOptions:
                        message = f"Test failed for the options: {bias_flag}, {generators_depth}, {diagonal_option}, {bias_option}"
                        with self.subTest(message=message):
                            cfg = LinearsConfigs.dynamic_preset(
                                bias_flag=bias_flag,
                                generator_depth=generators_depth,
                                diagonal_option=diagonal_option,
                                bias_option=bias_option,
                            )
                            cfg = cfg.linear_layer_config
                            m = DynamicLinearLayer(cfg)

                            self.assertEqual(m.input_dim, cfg.input_dim)
                            self.assertEqual(m.output_dim, cfg.output_dim)
                            self.assertIsInstance(m.weight_params, torch.Tensor)
                            if bias_flag:
                                self.assertIsInstance(m.bias_params, torch.Tensor)
                            else:
                                self.assertIsNone(m.bias_params)

    def test_forward(self):
        bias_options = [True, False]
        input_params = output_params = [4, 8, 16]

        for bias_flag in bias_options:
            for input_dim in input_params:
                for output_dim in output_params:
                    for generators_depth in DynamicDepthOptions:
                        for diagonal_option in DynamicDiagonalOptions:
                            for bias_option in DynamicBiasOptions:
                                for memory_option in LinearMemoryOptions:
                                    for position_option in LinearMemoryPositionOptions:
                                        for size_option in LinearMemorySizeOptions:
                                            message = f"Test failed for options - Bias flag: {bias_flag}, Generator depth: {generators_depth}, Diagonal option: {diagonal_option}, Bias option: {bias_option}, Memory option: {memory_option}, Position option: {position_option}, Size option: {size_option}, Input dimension: {input_dim}, Output dimension: {output_dim}."
                                            with self.subTest(message=message):
                                                batch_size = 2
                                                cfg = LinearsConfigs.dynamic_preset(
                                                    batch_size=batch_size,
                                                    input_dim=input_dim,
                                                    output_dim=output_dim,
                                                    bias_flag=bias_flag,
                                                    generator_depth=generators_depth,
                                                    diagonal_option=diagonal_option,
                                                    bias_option=bias_option,
                                                    memory_option=memory_option,
                                                    memory_position_option=position_option,
                                                    memory_size_option=size_option,
                                                )
                                                cfg = cfg.linear_layer_config

                                                if (
                                                    memory_option
                                                    != LinearMemoryOptions.DISABLED
                                                    and size_option
                                                    == LinearMemorySizeOptions.DISABLED
                                                ):
                                                    with self.assertRaises(ValueError):
                                                        m = DynamicLinearLayer(cfg)
                                                else:
                                                    m = DynamicLinearLayer(cfg)
                                                    input_batch = torch.randn(
                                                        batch_size, input_dim
                                                    )
                                                    output = m.forward(input_batch)
                                                    expected_output_shape = (
                                                        batch_size,
                                                        output_dim,
                                                    )
                                                    self.assertEqual(
                                                        output.shape,
                                                        expected_output_shape,
                                                    )

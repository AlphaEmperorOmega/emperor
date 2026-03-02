import torch
import unittest

from emperor.config import ModelConfig
from emperor.linears.options import LinearLayerStackOptions
from emperor.sampler.utils.presets import SamplerPresets
from emperor.sampler.utils.routers import RouterModel
from emperor.behaviours.utils.enums import (
    DynamicBiasOptions,
    DynamicDepthOptions,
    DynamicDiagonalOptions,
    LinearMemoryOptions,
    LinearMemorySizeOptions,
)


class TestRouterModel(unittest.TestCase):
    def setUp(self):
        self.rebuild_presets()

    def tearDown(self):
        self.cfg = None
        self.config = None
        self.batch_size = None
        self.input_dim = None
        self.output_dim = None

    def rebuild_presets(self, config: ModelConfig | None = None):
        self.cfg = (
            SamplerPresets.router_preset(
                return_model_config_flag=True,
            )
            if config is None
            else config
        )

        self.batch_size = self.cfg.batch_size
        self.input_dim = self.cfg.input_dim
        self.output_dim = self.cfg.output_dim

    def test_ensure_invalid_inputs_throw_errors(self):
        num_experts = [0, -1]
        for n in num_experts:
            message = f"AssertionError should be raised for the inputs: {n}"
            with self.subTest(msg=message):
                with self.assertRaises(AssertionError):
                    config = SamplerPresets.router_preset(num_experts=n)
                    RouterModel(config)

    def test_init_with_different_configs(self):
        num_experts_options = [1, 4, 8]
        noisy_flag_options = [True, False]

        for num_experts in num_experts_options:
            for noisy_flag_opition in noisy_flag_options:
                message = f"Testing configuration with num_experts={num_experts} and noisy_flag_option={noisy_flag_opition}"
                with self.subTest(msg=message):
                    config = SamplerPresets.router_preset(
                        return_model_config_flag=True,
                        num_experts=num_experts,
                        noisy_topk_flag=noisy_flag_opition,
                    )
                    model = RouterModel(config)
                    self.assertEqual(model.noisy_topk_flag, noisy_flag_opition)
                    if noisy_flag_opition:
                        self.assertEqual(model.num_experts, num_experts * 2)
                    else:
                        self.assertEqual(model.num_experts, num_experts)

    def test_forward(self):
        num_layer_options = [1, 2, 3]
        num_experts_options = [4, 8]
        noisy_flag_options = [True, False]

        for num_layers in num_layer_options:
            for layer_stack_option in LinearLayerStackOptions:
                for num_experts in num_experts_options:
                    for noisy_flag_option in noisy_flag_options:
                        message = f"Testing the configuration with num_layers={num_layers}, layer_stack_option={layer_stack_option}, num_experts={num_experts}, and noisy_flag_option={noisy_flag_option}"
                        with self.subTest(msg=message):
                            cfg = SamplerPresets.router_preset(
                                return_model_config_flag=True,
                                layer_stack_option=layer_stack_option,
                                num_experts=num_experts,
                                noisy_topk_flag=noisy_flag_option,
                                stack_num_layers=num_layers,
                                bias_option=DynamicBiasOptions.DYNAMIC_PARAMETERS,
                                memory_option=LinearMemoryOptions.WEIGHTED,
                                generator_depth=DynamicDepthOptions.DEPTH_OF_THREE,
                                diagonal_option=DynamicDiagonalOptions.DIAGONAL_AND_ANTI_DIAGONAL,
                                memory_size_option=LinearMemorySizeOptions.LARGE,
                            )
                            model = RouterModel(cfg)

                            input_batch = torch.randn(cfg.batch_size, cfg.input_dim)
                            output = model.compute_logit_scores(input_batch)
                            if noisy_flag_option:
                                self.assertEqual(
                                    output.shape, (cfg.batch_size, num_experts * 2)
                                )
                                self.assertEqual(model.num_experts, num_experts * 2)
                            else:
                                self.assertEqual(model.num_experts, num_experts)
                                self.assertEqual(
                                    output.shape, (cfg.batch_size, num_experts)
                                )

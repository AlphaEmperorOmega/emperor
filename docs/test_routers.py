import copy
import torch
import unittest

from Emperor.base.layer import Layer
from Emperor.config import ModelConfig
from Emperor.generators.utils.routers import VectorRouterModel
from Emperor.linears.options import LinearLayerOptions
from Emperor.linears.utils.enums import (
    DynamicBiasOptions,
    DynamicDepthOptions,
    DynamicDiagonalOptions,
    LinearMemoryOptions,
    LinearMemorySizeOptions,
)
from Emperor.sampler.utils.config import SamplerConfigs
from Emperor.sampler.utils.routers import RouterConfig, RouterModel
from Emperor.linears.utils.layers import DynamicLinearLayer, LinearLayer


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
        self.cfg = SamplerConfigs.router_preset() if config is None else config

        self.batch_size = self.cfg.batch_size
        self.input_dim = self.cfg.input_dim
        self.output_dim = self.cfg.output_dim

    def test_ensure_invalid_inputs_throw_errors(self):
        num_experts = [0, -1]
        for n in num_experts:
            message = f"AssertionError should be raised for the inputs: {n}"
            with self.subTest(msg=message):
                with self.assertRaises(AssertionError):
                    config = SamplerConfigs.router_preset(num_experts=n)
                    RouterModel(config)

    def test_init_with_different_configs(self):
        num_experts_options = [1, 4, 8]
        noisy_flag_options = [True, False]

        for num_experts in num_experts_options:
            for noisy_flag_opition in noisy_flag_options:
                message = f"Testing configuration with num_experts={num_experts} and noisy_flag_option={noisy_flag_opition}"
                with self.subTest(msg=message):
                    config = SamplerConfigs.router_preset(
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
        num_experts_options = [1, 4, 8]
        noisy_flag_options = [True, False]

        for num_experts in num_experts_options:
            for noisy_flag_option in noisy_flag_options:
                message = f"Testing the configuration with num_experts={num_experts} and noisy_flag_option={noisy_flag_option}"
                with self.subTest(msg=message):
                    config = SamplerConfigs.router_preset(
                        num_experts=num_experts,
                        noisy_topk_flag=noisy_flag_option,
                        model_type=LinearLayerOptions.DYNAMIC,
                        bias_option=DynamicBiasOptions.DYNAMIC_PARAMETERS,
                        memory_option=LinearMemoryOptions.FUSION,
                        generator_depth=DynamicDepthOptions.DEPTH_OF_TWO,
                        diagonal_option=DynamicDiagonalOptions.DIAGONAL_AND_ANTI_DIAGONAL,
                        memory_size_option=LinearMemorySizeOptions.MEDIUM,
                    )
                    model = RouterModel(config)

                    input_batch = torch.randn(config.batch_size, config.input_dim)
                    output = model.compute_logit_scores(input_batch)

                    self.assertEqual(output.shape, (config.batch_size, num_experts))
                    if noisy_flag_option:
                        self.assertEqual(model.num_experts, num_experts * 2)
                    else:
                        self.assertEqual(model.num_experts, num_experts)


# def test__compute_logit_scores__noisy_topk__False(self):
#     c = copy.deepcopy(self.cfg)
#     overrides = RouterConfig(
#         noisy_topk_flag=False,
#         diagonal_model_type_flag=True,
#     )
#     m = RouterModel(c, overrides)
#
#     batch_size = 2
#     input_batch = randn(batch_size, m.input_dim)
#     output = m.compute_logit_scores(input_batch)
#
#     self.assertEqual(list(output.shape), [batch_size, m.num_experts])


#     def test__compute_logit_scores__noisy_topk__True(self):
#         c = copy.deepcopy(self.cfg)
#         overrides = RouterConfig(
#             noisy_topk_flag=True,
#             diagonal_model_type_flag=True,
#         )
#         m = RouterModel(c, overrides)
#
#         batch_size = 2
#         input_batch = randn(batch_size, m.input_dim)
#         output = m.compute_logit_scores(input_batch)
#
#         self.assertEqual(list(output.shape), [batch_size, c.num_experts * 2])
#
#     def test__create_router_layer_model__diagonal_model_type_flag__False(self):
#         c = copy.deepcopy(self.cfg)
#         overrides = RouterConfig(noisy_topk_flag=True, diagonal_model_type_flag=False)
#         m = RouterModel(c, overrides)
#
#         model = m._RouterModel__create_router_layer_model(c.input_dim, c.num_experts)
#
#         self.assertIsInstance(model, LinearLayer)
#
#     def test__create_router_layer_model__diagonal_model_type_flag__True(self):
#         c = copy.deepcopy(self.cfg)
#         overrides = RouterConfig(
#             noisy_topk_flag=True,
#             diagonal_model_type_flag=True,
#         )
#         m = RouterModel(c, overrides)
#
#         model = m._RouterModel__create_router_layer_model(c.input_dim, c.num_experts)
#
#         self.assertIsInstance(model, DynamicLinearLayer)
#
#
# class TestVectorRouterModel(unittest.TestCase):
#     def setUp(self):
#         self.cfg = RouterConfig(
#             input_dim=5,
#             hidden_dim=6,
#             num_experts=7,
#             noisy_topk_flag=False,
#             residual_flag=False,
#             activation=nn.Sigmoid(),
#             num_layers=3,
#             diagonal_model_type_flag=False,
#         )
#
#     def test__generate_parameter_bank__bias_parameters_flag__False(self):
#         c = copy.deepcopy(self.cfg)
#         m = VectorRouterModel(c)
#
#         parameters = m._VectorRouterModel__generate_parameter_bank()
#
#         expected_shape = [c.input_dim, c.input_dim, c.num_experts]
#         self.assertEqual(list(parameters.shape), expected_shape)
#
#     def test__generate_parameter_bank__bias_parameters_flag__True(self):
#         c = copy.deepcopy(self.cfg)
#         m = VectorRouterModel(c)
#
#         parameters = m._VectorRouterModel__generate_parameter_bank()
#
#         expected_shape = [c.input_dim, c.input_dim, c.num_experts]
#         self.assertEqual(list(parameters.shape), expected_shape)
#
#     def test__compute_logit_scores__noisy_topk__False(self):
#         c = copy.deepcopy(self.cfg)
#         overrides = RouterConfig(
#             noisy_topk_flag=False,
#             diagonal_model_type_flag=True,
#         )
#         m = VectorRouterModel(c)
#
#         batch_size = 2
#         input_batch = randn(batch_size, m.input_dim)
#         output = m.compute_logit_scores(input_batch)
#
#         self.assertEqual(list(output.shape), [m.input_dim, batch_size, m.num_experts])
#
#     def test__compute_logit_scores__noisy_topk__True(self):
#         c = copy.deepcopy(self.cfg)
#         overrides = RouterConfig(
#             noisy_topk_flag=True,
#             diagonal_model_type_flag=True,
#         )
#         m = VectorRouterModel(c, overrides)
#
#         batch_size = 2
#         input_batch = randn(batch_size, m.input_dim)
#         output = m.compute_logit_scores(input_batch)
#
#         self.assertEqual(list(output.shape), [m.input_dim, batch_size, m.num_experts])

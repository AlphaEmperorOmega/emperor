import unittest
import copy
import torch
import torch.nn as nn
from math import prod

from dataclasses import asdict
from Emperor.config import ModelConfig
from Emperor.linears.utils.config import LinearsConfigs
from Emperor.generators.utils.behaviours import (
    DynamicDiagonalParametersBehaviour,
)
from Emperor.linears.utils.enums import (
    DynamicBiasOptions,
    DynamicDepthOptions,
    DynamicDiagonalOptions,
)
from Emperor.linears.utils.layers import (
    DynamicLinearLayer,
    LinearLayer,
    LinearLayerConfig,
)
# from Emperor.generators.utils.linears import (
#     DynamicLinearLayer,
#     LinearLayer,
#     LinearLayerConfig,
# )

# from Emperor.generators.utils.mixture import MixtureConfig
# from Emperor.generators.utils.routers import RouterConfig
# from Emperor.generators.utils.samplers import SamplerConfig


class TestLinears(unittest.TestCase):
    def setUp(self):
        self.rebuild_presets()

    def tearDown(self):
        self.cfg = None
        self.config = None
        self.model = None
        self.batch_size = None
        self.num_heads = None
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
                m = LinearLayer(c)

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
                        c = LinearsConfigs.base_preset(
                            input_dim, output_dim, bias_flag=bias_flag
                        )
                        m = LinearLayer(c)

                        input_batch = torch.randn(c.batch_size, c.input_dim)
                        output = m.forward(input_batch)
                        expected_output_shape = (c.batch_size, c.output_dim)
                        self.assertEqual(output.shape, expected_output_shape)


class TestDynamicLinearLayer(TestLinears):
    def test_init_with_different_configation_options(self):
        bias_options = [True, False]
        input_params = output_params = [4, 8, 16]

        for input_dim in input_params:
            for output_dim in output_params:
                for bias_flag in bias_options:
                    for generators_depth in DynamicDepthOptions:
                        for diagonal_option in DynamicDiagonalOptions:
                            for bias_option in DynamicBiasOptions:
                                message = f"Test failed for the options: {input_dim}, {output_dim}, {bias_flag}, {generators_depth}, {diagonal_option}, {bias_option}"
                                with self.subTest(message=message):
                                    c = LinearsConfigs.dynamic_preset(
                                        batch_size=2,
                                        input_dim=input_dim,
                                        output_dim=output_dim,
                                        bias_flag=bias_flag,
                                        generators_depth=generators_depth,
                                        diagonal_option=diagonal_option,
                                        bias_option=bias_option,
                                    )
                                    m = DynamicLinearLayer(c)

                                    self.assertEqual(m.input_dim, c.input_dim)
                                    self.assertEqual(m.output_dim, c.output_dim)
                                    self.assertIsInstance(m.weight_params, torch.Tensor)
                                    if bias_flag:
                                        self.assertIsInstance(
                                            m.bias_params, torch.Tensor
                                        )
                                    else:
                                        self.assertIsNone(m.bias_params)


# class TestLinearLayers(unittest.TestCase):
#     def setUp(self):
#         # MODEL WISE CONFI
#         BATCH_SIZE = 2
#         INPUT_DIM = 4
#         HIDDEN_DIM = 5
#         OUTPUT_DIM = 6
#         GATHER_FREQUENCY_FLAG = False
#
#         # AUXILIARY LOSSES OPITONS
#         COEFFICIENT_OF_VARIATION_LOSS_WEIGHT: float = 0.0
#         SWITCH_LOSS_WEIGHT: float = 0.0
#         ZERO_CENTERED_LOSS_WEIGHT: float = 0.0
#         MUTUAL_INFORMATION_LOSS_WEIGHT: float = 0.0
#
#         # PARAMETER GENRETOR ROUTER OPITONS
#         ROUTER_INPUT_DIM = HIDDEN_DIM
#         ROUTER_HIDDEN_DIM = 8
#         ROUTER_OUTPUT_DIM = 9
#         ROUTER_NOISY_TOPK_FLAG = False
#         ROUTER_ACTIVATION_FUNCTION = nn.ReLU()
#         ROUTER_NUM_LAYERNUM_LAYERSS = 5
#         ROUTER_DYNAMIC_LINEAR_MODEL_FLAG = False
#
#         # PARAMETER GENRETOR SAMPLER OPITONS
#         SAMPLER_TOP_K = 3
#         SAMPLER_THRESHOLD = 0.1
#         SAMPLER_DYNAMIC_TOPK_THRESHOLD = (1 / ROUTER_OUTPUT_DIM) * 1e-6
#         SAMPLER_NUM_TOPK_SAMPLES = 0
#         SAMPLER_NORMALIZE_PROBABILITIES_FLAG = False
#         SAMPLER_NOISY_TOPK_FLAG = ROUTER_NOISY_TOPK_FLAG
#         SAMPLER_ROUTER_OUTPUT_DIM = ROUTER_OUTPUT_DIM
#         SAMPLER_COEFFICIENT_OF_VARIATION_WEIGHT = 0.0
#         SAMPLER_SWITCH_WEIGHT = 0.0
#         SAMPLER_ZERO_CENTRED_WEIGHT = 0.0
#         SAMPLER_MUTUAL_INFORMATION_WEIGHT = 0.0
#
#         # PARAMETER GENRETOR MIXTURE OPITONS
#         MIXTURE_INPUT_DIM = ROUTER_INPUT_DIM
#         MIXTURE_OUTPUT_DIM = ROUTER_OUTPUT_DIM
#         MIXTURE_DEPTH_DIM = ROUTER_OUTPUT_DIM
#         MIXTURE_TOP_K = SAMPLER_TOP_K
#         MIXTURE_WEIGHTED_PARAMETERS_FLAG = False
#         MIXTURE_BIAS_PARAMETERS_FLAG = False
#         MIXTURE_ROUTER_OUTPUT_DIM = ROUTER_OUTPUT_DIM
#         MIXTURE_CROSS_DIAGONAL_FLAG = False
#
#         # PARAMETER GENERATOR OPTIONS
#         PARAMETER_GENERATOR_BIAS_PARAMETER_FLAG = MIXTURE_WEIGHTED_PARAMETERS_FLAG
#
#         self.cfg = ModelConfig(
#             batch_size=BATCH_SIZE,
#             input_dim=INPUT_DIM,
#             hidden_dim=HIDDEN_DIM,
#             output_dim=OUTPUT_DIM,
#             gather_frequency_flag=GATHER_FREQUENCY_FLAG,
#             router_model_config=RouterConfig(
#                 input_dim=ROUTER_INPUT_DIM,
#                 hidden_dim=ROUTER_HIDDEN_DIM,
#                 num_experts=ROUTER_OUTPUT_DIM,
#                 noisy_topk_flag=ROUTER_NOISY_TOPK_FLAG,
#                 activation=ROUTER_ACTIVATION_FUNCTION,
#                 num_layers=ROUTER_NUM_LAYERNUM_LAYERSS,
#                 diagonal_model_type_flag=ROUTER_DYNAMIC_LINEAR_MODEL_FLAG,
#             ),
#             sampler_model_config=SamplerConfig(
#                 top_k=SAMPLER_TOP_K,
#                 threshold=SAMPLER_THRESHOLD,
#                 num_topk_samples=SAMPLER_NUM_TOPK_SAMPLES,
#                 normalize_probabilities_flag=SAMPLER_NORMALIZE_PROBABILITIES_FLAG,
#                 noisy_topk_flag=SAMPLER_NOISY_TOPK_FLAG,
#                 num_experts=SAMPLER_ROUTER_OUTPUT_DIM,
#                 coefficient_of_variation_loss_weight=SAMPLER_COEFFICIENT_OF_VARIATION_WEIGHT,
#                 switch_loss_weight=SAMPLER_SWITCH_WEIGHT,
#                 zero_centred_loss_weight=SAMPLER_ZERO_CENTRED_WEIGHT,
#                 mutual_information_loss_weight=SAMPLER_MUTUAL_INFORMATION_WEIGHT,
#             ),
#             mixture_model_config=MixtureConfig(
#                 input_dim=MIXTURE_INPUT_DIM,
#                 output_dim=MIXTURE_OUTPUT_DIM,
#                 depth_dim=MIXTURE_DEPTH_DIM,
#                 top_k=MIXTURE_TOP_K,
#                 bias_parameters_flag=MIXTURE_BIAS_PARAMETERS_FLAG,
#                 weighted_parameters_flag=MIXTURE_WEIGHTED_PARAMETERS_FLAG,
#                 num_experts=MIXTURE_ROUTER_OUTPUT_DIM,
#                 dynamic_diagonal_params_flag=MIXTURE_CROSS_DIAGONAL_FLAG,
#             ),
#             parameter_generator_model_config=ParameterLayerConfig(
#                 bias_parameters_flag=PARAMETER_GENERATOR_BIAS_PARAMETER_FLAG,
#             ),
#             linear_layer_config=LinearLayerConfig(
#                 input_dim=INPUT_DIM,
#                 output_dim=OUTPUT_DIM,
#                 bias_flag=True,
#                 anti_diagonal_flag=True,
#             ),
#         )
#
#         self.parameter_generator_cfg = self.cfg.mixture_model_config
#
#     def test__default_linear_layer__init_with_main_config__bias_parameters_flag__False(
#         self,
#     ):
#         c = copy.deepcopy(self.cfg)
#         overrides = LinearLayerConfig(bias_flag=False)
#         m = LinearLayer(c, overrides)
#
#         self.assertEqual(m.input_dim, c.input_dim)
#         self.assertEqual(m.output_dim, c.output_dim)
#         self.assertIsInstance(m.weight_params, torch.Tensor)
#         self.assertIsNone(m.bias_params, None)
#
#     def test__default_linear_layer__init_with_main_config__bias_parameters_flag__True(
#         self,
#     ):
#         c = copy.deepcopy(self.cfg)
#         overrides = LinearLayerConfig(bias_flag=True)
#         m = LinearLayer(c, overrides)
#
#         self.assertEqual(m.input_dim, c.input_dim)
#         self.assertEqual(m.output_dim, c.output_dim)
#         self.assertIsInstance(m.weight_params, torch.Tensor)
#         self.assertIsInstance(m.bias_params, torch.Tensor)
#
#     def test__default_linear_layer__forward(self):
#         m = LinearLayer(self.cfg)
#
#         batch_size = 2
#         input_batch_shape = (batch_size, self.cfg.input_dim)
#         input_batch = (
#             torch.arange(prod(input_batch_shape)).reshape(input_batch_shape).float()
#         )
#         output = m.forward(input_batch)
#         self.assertEqual(list(output.shape), [batch_size, m.output_dim])
#
#
# class TestDynamicLinearLayer(unittest.TestCase):
#     def setUp(self):
#         # MODEL WISE CONFI
#         BATCH_SIZE = 2
#         INPUT_DIM = 4
#         HIDDEN_DIM = 5
#         OUTPUT_DIM = 6
#         GATHER_FREQUENCY_FLAG = False
#
#         # AUXILIARY LOSSES OPITONS
#         COEFFICIENT_OF_VARIATION_LOSS_WEIGHT: float = 0.0
#         SWITCH_LOSS_WEIGHT: float = 0.0
#         ZERO_CENTERED_LOSS_WEIGHT: float = 0.0
#         MUTUAL_INFORMATION_LOSS_WEIGHT: float = 0.0
#
#         # PARAMETER GENRETOR ROUTER OPITONS
#         ROUTER_INPUT_DIM = HIDDEN_DIM
#         ROUTER_HIDDEN_DIM = 8
#         ROUTER_OUTPUT_DIM = 9
#         ROUTER_NOISY_TOPK_FLAG = False
#         ROUTER_ACTIVATION_FUNCTION = nn.ReLU()
#         ROUTER_NUM_LAYERNUM_LAYERSS = 5
#         ROUTER_DYNAMIC_LINEAR_MODEL_FLAG = False
#
#         # PARAMETER GENRETOR SAMPLER OPITONS
#         SAMPLER_TOP_K = 3
#         SAMPLER_THRESHOLD = 0.1
#         SAMPLER_DYNAMIC_TOPK_THRESHOLD = (1 / ROUTER_OUTPUT_DIM) * 1e-6
#         SAMPLER_NUM_TOPK_SAMPLES = 0
#         SAMPLER_NORMALIZE_PROBABILITIES_FLAG = False
#         SAMPLER_NOISY_TOPK_FLAG = ROUTER_NOISY_TOPK_FLAG
#         SAMPLER_ROUTER_OUTPUT_DIM = ROUTER_OUTPUT_DIM
#         SAMPLER_COEFFICIENT_OF_VARIATION_WEIGHT = 0.0
#         SAMPLER_SWITCH_WEIGHT = 0.0
#         SAMPLER_ZERO_CENTRED_WEIGHT = 0.0
#         SAMPLER_MUTUAL_INFORMATION_WEIGHT = 0.0
#
#         # PARAMETER GENRETOR MIXTURE OPITONS
#         MIXTURE_INPUT_DIM = ROUTER_INPUT_DIM
#         MIXTURE_OUTPUT_DIM = ROUTER_OUTPUT_DIM
#         MIXTURE_DEPTH_DIM = ROUTER_OUTPUT_DIM
#         MIXTURE_TOP_K = SAMPLER_TOP_K
#         MIXTURE_WEIGHTED_PARAMETERS_FLAG = False
#         MIXTURE_BIAS_PARAMETERS_FLAG = False
#         MIXTURE_ROUTER_OUTPUT_DIM = ROUTER_OUTPUT_DIM
#         MIXTURE_CROSS_DIAGONAL_FLAG = False
#
#         # PARAMETER GENERATOR OPTIONS
#         PARAMETER_GENERATOR_BIAS_PARAMETER_FLAG = MIXTURE_WEIGHTED_PARAMETERS_FLAG
#
#         self.cfg = ModelConfig(
#             batch_size=BATCH_SIZE,
#             input_dim=INPUT_DIM,
#             hidden_dim=HIDDEN_DIM,
#             output_dim=OUTPUT_DIM,
#             gather_frequency_flag=GATHER_FREQUENCY_FLAG,
#             router_model_config=RouterConfig(
#                 input_dim=ROUTER_INPUT_DIM,
#                 hidden_dim=ROUTER_HIDDEN_DIM,
#                 num_experts=ROUTER_OUTPUT_DIM,
#                 noisy_topk_flag=ROUTER_NOISY_TOPK_FLAG,
#                 activation=ROUTER_ACTIVATION_FUNCTION,
#                 num_layers=ROUTER_NUM_LAYERNUM_LAYERSS,
#                 diagonal_model_type_flag=ROUTER_DYNAMIC_LINEAR_MODEL_FLAG,
#             ),
#             sampler_model_config=SamplerConfig(
#                 top_k=SAMPLER_TOP_K,
#                 threshold=SAMPLER_THRESHOLD,
#                 num_topk_samples=SAMPLER_NUM_TOPK_SAMPLES,
#                 normalize_probabilities_flag=SAMPLER_NORMALIZE_PROBABILITIES_FLAG,
#                 noisy_topk_flag=SAMPLER_NOISY_TOPK_FLAG,
#                 num_experts=SAMPLER_ROUTER_OUTPUT_DIM,
#                 coefficient_of_variation_loss_weight=SAMPLER_COEFFICIENT_OF_VARIATION_WEIGHT,
#                 switch_loss_weight=SAMPLER_SWITCH_WEIGHT,
#                 zero_centred_loss_weight=SAMPLER_ZERO_CENTRED_WEIGHT,
#                 mutual_information_loss_weight=SAMPLER_MUTUAL_INFORMATION_WEIGHT,
#             ),
#             mixture_model_config=MixtureConfig(
#                 input_dim=MIXTURE_INPUT_DIM,
#                 output_dim=MIXTURE_OUTPUT_DIM,
#                 depth_dim=MIXTURE_DEPTH_DIM,
#                 top_k=MIXTURE_TOP_K,
#                 bias_parameters_flag=MIXTURE_BIAS_PARAMETERS_FLAG,
#                 weighted_parameters_flag=MIXTURE_WEIGHTED_PARAMETERS_FLAG,
#                 num_experts=MIXTURE_ROUTER_OUTPUT_DIM,
#                 dynamic_diagonal_params_flag=MIXTURE_CROSS_DIAGONAL_FLAG,
#             ),
#             parameter_generator_model_config=ParameterLayerConfig(
#                 bias_parameters_flag=PARAMETER_GENERATOR_BIAS_PARAMETER_FLAG,
#             ),
#             linear_layer_config=LinearLayerConfig(
#                 input_dim=INPUT_DIM,
#                 output_dim=OUTPUT_DIM,
#                 bias_flag=True,
#                 anti_diagonal_flag=True,
#             ),
#         )
#
#         self.parameter_generator_cfg = self.cfg.mixture_model_config
#
#     def test__init_with_main_config(self):
#         m = DynamicLinearLayer(self.cfg)
#         self.assertIsInstance(
#             m.dynamic_diagonal_params_model, DynamicDiagonalParametersBehaviour
#         )
#
#     def test__forward__anti_diagonal_flag__False(self):
#         c = copy.deepcopy(self.cfg)
#         overrides = LinearLayerConfig(anti_diagonal_flag=False)
#         m = DynamicLinearLayer(c, overrides)
#         batch_size = 2
#         input_batch_shape = (batch_size, self.cfg.input_dim)
#         input_batch = (
#             torch.arange(prod(input_batch_shape)).reshape(input_batch_shape).float()
#         )
#         output = m.forward(input_batch)
#         self.assertEqual(list(output.shape), [batch_size, m.output_dim])
#
#     def test__forward__anti_diagonal_flag__True(self):
#         c = copy.deepcopy(self.cfg)
#         overrides = LinearLayerConfig(
#             anti_diagonal_flag=True,
#         )
#         m = DynamicLinearLayer(c, overrides)
#         batch_size = 2
#         input_batch_shape = (batch_size, self.cfg.input_dim)
#         input_batch = (
#             torch.arange(prod(input_batch_shape)).reshape(input_batch_shape).float()
#         )
#
#         output = m.forward(input_batch)
#         self.assertEqual(list(output.shape), [batch_size, m.output_dim])
#
#     def test__forward__dynamic_bias_flag__True(self):
#         c = copy.deepcopy(self.cfg)
#         overrides = LinearLayerConfig(
#             dynamic_bias_flag=True,
#         )
#         m = DynamicLinearLayer(c, overrides)
#         batch_size = 2
#         input_batch_shape = (batch_size, self.cfg.input_dim)
#         input_batch = (
#             torch.arange(prod(input_batch_shape)).reshape(input_batch_shape).float()
#         )
#
#         output = m.forward(input_batch)
#         self.assertEqual(list(output.shape), [batch_size, m.output_dim])
#
#     def test__forward__all_flags_true(self):
#         c = copy.deepcopy(self.cfg)
#         overrides = LinearLayerConfig(
#             dynamic_bias_flag=True,
#             bias_flag=True,
#             anti_diagonal_flag=True,
#         )
#         m = DynamicLinearLayer(c, overrides)
#         batch_size = 2
#         input_batch_shape = (batch_size, self.cfg.input_dim)
#         input_batch = (
#             torch.arange(prod(input_batch_shape)).reshape(input_batch_shape).float()
#         )
#
#         output = m.forward(input_batch)
#         self.assertEqual(list(output.shape), [batch_size, m.output_dim])
#
#     def test__compute_linear_transformation(self):
#         c = copy.deepcopy(self.cfg)
#         overrides = LinearLayerConfig(anti_diagonal_flag=False)
#         m = DynamicLinearLayer(c, overrides)
#         batch_size = 2
#         shape = (batch_size, self.cfg.input_dim)
#         logits = torch.randn(shape)
#         shape = (batch_size, self.cfg.input_dim, self.cfg.output_dim)
#         dynamic_weight_params = torch.randn(shape)
#         output = m._DynamicLinearLayer__compute_linear_transformation(
#             logits, dynamic_weight_params
#         )
#         for i in range(batch_size):
#             expected_vector = torch.matmul(logits[i], dynamic_weight_params[i])
#             self.assertTrue(torch.allclose(output[i], expected_vector))
#
#         self.assertEqual(list(output.shape), [batch_size, m.output_dim])
#
#     def test__add_bias_parameters__bias_flag__True(self):
#         c = copy.deepcopy(self.cfg)
#         overrides = LinearLayerConfig(
#             anti_diagonal_flag=False,
#             bias_flag=True,
#         )
#         m = DynamicLinearLayer(c, overrides)
#         batch_size = 2
#         shape = (batch_size, self.cfg.output_dim)
#         linear_transform = torch.randn(shape)
#         shape = (1, self.cfg.output_dim)
#         bias_params = torch.randn(shape)
#         output = m._DynamicLinearLayer__add_bias_parameters(
#             linear_transform, bias_params
#         )
#
#         expected_output = linear_transform + bias_params
#         self.assertEqual(list(output.shape), [batch_size, m.output_dim])
#         self.assertTrue(torch.allclose(output, expected_output))

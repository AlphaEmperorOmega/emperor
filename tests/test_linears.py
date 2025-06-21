import unittest
import copy
import torch
import torch.nn as nn
from math import prod

from Emperor.components.parameter_generators.layers import ParameterLayerConfig
from Emperor.components.parameter_generators.utils.behaviours import (
    DynamicDiagonalParametersBehaviour,
)
from Emperor.components.parameter_generators.utils.linears import (
    DynamicDiagonalLinearLayer,
    LinearLayer,
    LinearLayerConfig,
)
from Emperor.components.parameter_generators.utils.mixture import MixtureConfig
from Emperor.components.parameter_generators.utils.routers import RouterConfig
from Emperor.components.parameter_generators.utils.samplers import SamplerConfig
from Emperor.config import ModelConfig


class TestLinearLayers(unittest.TestCase):
    def setUp(self):
        # MODEL WISE CONFI
        BATCH_SIZE = 2
        INPUT_DIM = 4
        HIDDEN_DIM = 5
        OUTPUT_DIM = 6
        GATHER_FREQUENCY_FLAG = False

        # AUXILIARY LOSSES OPITONS
        COEFFICIENT_OF_VARIATION_LOSS_WEIGHT: float = 0.0
        SWITCH_LOSS_WEIGHT: float = 0.0
        ZERO_CENTERED_LOSS_WEIGHT: float = 0.0
        MUTUAL_INFORMATION_LOSS_WEIGHT: float = 0.0

        # PARAMETER GENRETOR ROUTER OPITONS
        ROUTER_INPUT_DIM = HIDDEN_DIM
        ROUTER_HIDDEN_DIM = 8
        ROUTER_OUTPUT_DIM = 9
        ROUTER_NOISY_TOPK_FLAG = False
        ROUTER_ACTIVATION_FUNCTION = nn.ReLU()
        ROUTER_NUM_LAYERNUM_LAYERSS = 5
        ROUTER_DYNAMIC_LINEAR_MODEL_FLAG = False

        # PARAMETER GENRETOR SAMPLER OPITONS
        SAMPLER_TOP_K = 3
        SAMPLER_THRESHOLD = 0.1
        SAMPLER_DYNAMIC_TOPK_THRESHOLD = (1 / ROUTER_OUTPUT_DIM) * 1e-6
        SAMPLER_NUM_TOPK_SAMPLES = 0
        SAMPLER_NORMALIZE_PROBABILITIES_FLAG = False
        SAMPLER_NOISY_TOPK_FLAG = ROUTER_NOISY_TOPK_FLAG
        SAMPLER_ROUTER_OUTPUT_DIM = ROUTER_OUTPUT_DIM
        SAMPLER_COEFFICIENT_OF_VARIATION_WEIGHT = 0.0
        SAMPLER_SWITCH_WEIGHT = 0.0
        SAMPLER_ZERO_CENTRED_WEIGHT = 0.0
        SAMPLER_MUTUAL_INFORMATION_WEIGHT = 0.0

        # PARAMETER GENRETOR MIXTURE OPITONS
        MIXTURE_INPUT_DIM = ROUTER_INPUT_DIM
        MIXTURE_OUTPUT_DIM = ROUTER_OUTPUT_DIM
        MIXTURE_DEPTH_DIM = ROUTER_OUTPUT_DIM
        MIXTURE_TOP_K = SAMPLER_TOP_K
        MIXTURE_WEIGHTED_PARAMETERS_FLAG = False
        MIXTURE_BIAS_PARAMETERS_FLAG = False
        MIXTURE_ROUTER_OUTPUT_DIM = ROUTER_OUTPUT_DIM
        MIXTURE_CROSS_DIAGONAL_FLAG = False

        # PARAMETER GENERATOR OPTIONS
        PARAMETER_GENERATOR_BIAS_PARAMETER_FLAG = MIXTURE_WEIGHTED_PARAMETERS_FLAG

        self.cfg = ModelConfig(
            batch_size=BATCH_SIZE,
            input_dim=INPUT_DIM,
            hidden_dim=HIDDEN_DIM,
            output_dim=OUTPUT_DIM,
            gather_frequency_flag=GATHER_FREQUENCY_FLAG,
            router_model_config=RouterConfig(
                input_dim=ROUTER_INPUT_DIM,
                hidden_dim=ROUTER_HIDDEN_DIM,
                output_dim=ROUTER_OUTPUT_DIM,
                noisy_topk_flag=ROUTER_NOISY_TOPK_FLAG,
                activation=ROUTER_ACTIVATION_FUNCTION,
                num_layers=ROUTER_NUM_LAYERNUM_LAYERSS,
                diagonal_linear_model_flag=ROUTER_DYNAMIC_LINEAR_MODEL_FLAG,
            ),
            sampler_model_config=SamplerConfig(
                top_k=SAMPLER_TOP_K,
                threshold=SAMPLER_THRESHOLD,
                num_topk_samples=SAMPLER_NUM_TOPK_SAMPLES,
                normalize_probabilities_flag=SAMPLER_NORMALIZE_PROBABILITIES_FLAG,
                noisy_topk_flag=SAMPLER_NOISY_TOPK_FLAG,
                router_output_dim=SAMPLER_ROUTER_OUTPUT_DIM,
                coefficient_of_variation_weight=SAMPLER_COEFFICIENT_OF_VARIATION_WEIGHT,
                switch_weight=SAMPLER_SWITCH_WEIGHT,
                zero_centred_weight=SAMPLER_ZERO_CENTRED_WEIGHT,
                mutual_information_weight=SAMPLER_MUTUAL_INFORMATION_WEIGHT,
            ),
            mixture_model_config=MixtureConfig(
                input_dim=MIXTURE_INPUT_DIM,
                output_dim=MIXTURE_OUTPUT_DIM,
                depth_dim=MIXTURE_DEPTH_DIM,
                top_k=MIXTURE_TOP_K,
                bias_parameters_flag=MIXTURE_BIAS_PARAMETERS_FLAG,
                weighted_parameters_flag=MIXTURE_WEIGHTED_PARAMETERS_FLAG,
                router_output_dim=MIXTURE_ROUTER_OUTPUT_DIM,
                cross_diagonal_flag=MIXTURE_CROSS_DIAGONAL_FLAG,
            ),
            parameter_generator_model_config=ParameterLayerConfig(
                bias_parameters_flag=PARAMETER_GENERATOR_BIAS_PARAMETER_FLAG,
            ),
            linear_layer_model_config=LinearLayerConfig(
                input_dim=INPUT_DIM,
                output_dim=OUTPUT_DIM,
                bias_flag=True,
                anti_diagonal_flag=True,
            ),
        )

        self.parameter_generator_cfg = self.cfg.mixture_model_config

    def test__default_linear_layer__init_with_main_config__bias_parameters_flag__False(
        self,
    ):
        c = copy.deepcopy(self.cfg)
        overrides = LinearLayerConfig(bias_flag=False)
        m = LinearLayer(c, overrides)

        self.assertEqual(m.input_dim, c.input_dim)
        self.assertEqual(m.output_dim, c.output_dim)
        self.assertIsInstance(m.weight_params, torch.Tensor)
        self.assertIsNone(m.bias_params, None)

    def test__default_linear_layer__init_with_main_config__bias_parameters_flag__True(
        self,
    ):
        c = copy.deepcopy(self.cfg)
        overrides = LinearLayerConfig(bias_flag=True)
        m = LinearLayer(c, overrides)

        self.assertEqual(m.input_dim, c.input_dim)
        self.assertEqual(m.output_dim, c.output_dim)
        self.assertIsInstance(m.weight_params, torch.Tensor)
        self.assertIsInstance(m.bias_params, torch.Tensor)

    def test__default_linear_layer__forward(self):
        m = LinearLayer(self.cfg)

        batch_size = 2
        input_batch_shape = (batch_size, self.cfg.input_dim)
        input_batch = (
            torch.arange(prod(input_batch_shape)).reshape(input_batch_shape).float()
        )
        output = m.forward(input_batch)
        self.assertEqual(list(output.shape), [batch_size, m.output_dim])

    def test__cross_diagonal_linear_layer__init_with_main_config(self):
        m = DynamicDiagonalLinearLayer(self.cfg)
        self.assertIsInstance(m.diagonal_behaviour, DynamicDiagonalParametersBehaviour)

    def test__cross_diagonal_linear_layer__forward__anti_diagonal_flag__False(self):
        c = copy.deepcopy(self.cfg)
        overrides = LinearLayerConfig(anti_diagonal_flag=False)
        m = DynamicDiagonalLinearLayer(c, overrides)
        batch_size = 2
        input_batch_shape = (batch_size, self.cfg.input_dim)
        input_batch = (
            torch.arange(prod(input_batch_shape)).reshape(input_batch_shape).float()
        )
        output = m.forward(input_batch)
        self.assertEqual(list(output.shape), [batch_size, m.output_dim])

    def test__cross_diagonal_linear_layer__forward__anti_diagonal_flag__True(self):
        c = copy.deepcopy(self.cfg)
        overrides = LinearLayerConfig(anti_diagonal_flag=False)
        m = DynamicDiagonalLinearLayer(c, overrides)
        batch_size = 2
        input_batch_shape = (batch_size, self.cfg.input_dim)
        input_batch = (
            torch.arange(prod(input_batch_shape)).reshape(input_batch_shape).float()
        )
        output = m.forward(input_batch)
        self.assertEqual(list(output.shape), [batch_size, m.output_dim])

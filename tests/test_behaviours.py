import copy
import torch
import unittest
from math import prod
import torch.nn as nn

from Emperor.layers.layers import ParameterLayerConfig
from Emperor.layers.utils.behaviours import (
    DynamicDiagonalParametersBehaviour,
)
from Emperor.layers.utils.mixture import MixtureConfig
from Emperor.layers.utils.routers import RouterConfig
from Emperor.layers.utils.samplers import SamplerConfig
from Emperor.config import ModelConfig


class TestDefaultLinearLayers(unittest.TestCase):
    def setUp(self):
        # MODEL WISE CONFI
        BATCH_SIZE = 2
        INPUT_DIM = 4
        HIDDEN_DIM = 5
        OUTPUT_DIM = 6
        GATHER_FREQUENCY_FLAG = False

        # PARAMETER GENRETOR ROUTER OPITONS
        ROUTER_INPUT_DIM = HIDDEN_DIM
        ROUTER_HIDDEN_DIM = 8
        ROUTER_OUTPUT_DIM = 9
        ROUTER_NOISY_TOPK_FLAG = False
        ROUTER_ACTIVATION_FUNCTION = nn.ReLU()
        ROUTER_NUM_LAYERNUM_LAYERSS = 5

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
            ),
            sampler_model_config=SamplerConfig(
                top_k=SAMPLER_TOP_K,
                threshold=SAMPLER_THRESHOLD,
                num_topk_samples=SAMPLER_NUM_TOPK_SAMPLES,
                normalize_probabilities_flag=SAMPLER_NORMALIZE_PROBABILITIES_FLAG,
                noisy_topk_flag=SAMPLER_NOISY_TOPK_FLAG,
                num_experts=SAMPLER_ROUTER_OUTPUT_DIM,
                coefficient_of_variation_loss_weight=SAMPLER_COEFFICIENT_OF_VARIATION_WEIGHT,
                switch_loss_weight=SAMPLER_SWITCH_WEIGHT,
                zero_centred_loss_weight=SAMPLER_ZERO_CENTRED_WEIGHT,
                mutual_information_loss_weight=SAMPLER_MUTUAL_INFORMATION_WEIGHT,
            ),
            mixture_model_config=MixtureConfig(
                input_dim=MIXTURE_INPUT_DIM,
                output_dim=MIXTURE_OUTPUT_DIM,
                depth_dim=MIXTURE_DEPTH_DIM,
                top_k=MIXTURE_TOP_K,
                bias_parameters_flag=MIXTURE_BIAS_PARAMETERS_FLAG,
                weighted_parameters_flag=MIXTURE_WEIGHTED_PARAMETERS_FLAG,
                num_experts=MIXTURE_ROUTER_OUTPUT_DIM,
                dynamic_diagonal_params_flag=MIXTURE_CROSS_DIAGONAL_FLAG,
            ),
            parameter_generator_model_config=ParameterLayerConfig(
                bias_parameters_flag=PARAMETER_GENERATOR_BIAS_PARAMETER_FLAG,
            ),
        )

        self.parameter_generator_cfg = self.cfg.mixture_model_config

    def test__initialization__anti_diagonal_flag__False(self):
        c = copy.deepcopy(self.cfg)
        weight_shape = (c.input_dim, c.output_dim)
        weight_params = torch.arange(prod(weight_shape)).reshape(weight_shape).float()

        m = DynamicDiagonalParametersBehaviour(
            c.input_dim,
            c.output_dim,
            weight_params=weight_params,
            anti_diagonal_flag=False,
        )

        self.assertEqual(m.input_dim, c.input_dim)
        self.assertEqual(m.output_dim, c.output_dim)
        self.assertIsNone(m.anti_diagonal_model)

    def test__initialization__anti_diagonal_flag__True(self):
        c = copy.deepcopy(self.cfg)
        weight_shape = (c.input_dim, c.output_dim)
        weight_params = torch.arange(prod(weight_shape)).reshape(weight_shape).float()
        bias_shape = (c.output_dim,)
        bias_params = torch.arange(prod(bias_shape)).reshape(bias_shape).float()

        m = DynamicDiagonalParametersBehaviour(
            c.input_dim,
            c.output_dim,
            weight_params=weight_params,
            bias_params=bias_params,
            anti_diagonal_flag=True,
        )

        self.assertEqual(m.input_dim, c.input_dim)
        self.assertEqual(m.output_dim, c.output_dim)

    def test__get_diagonal_padding_shape__input_dim_equals_output_dim(self):
        c = copy.deepcopy(self.cfg)
        c.input_dim = 10
        c.output_dim = 10
        weight_shape = (c.input_dim, c.output_dim)
        weight_params = torch.arange(prod(weight_shape)).reshape(weight_shape).float()

        m = DynamicDiagonalParametersBehaviour(
            c.input_dim,
            c.output_dim,
            weight_params=weight_params,
            anti_diagonal_flag=True,
        )
        padding_shape = (
            m._DynamicDiagonalParametersBehaviour__get_diagonal_padding_shape()
        )
        self.assertIsNone(padding_shape)

    def test__get_diagonal_padding_shape__input_dim_greater_than_output_dim(self):
        c = copy.deepcopy(self.cfg)
        c.input_dim = 15
        c.output_dim = 10
        weight_shape = (c.input_dim, c.output_dim)
        weight_params = torch.arange(prod(weight_shape)).reshape(weight_shape).float()

        m = DynamicDiagonalParametersBehaviour(
            c.input_dim,
            c.output_dim,
            weight_params=weight_params,
            anti_diagonal_flag=True,
        )
        padding_shape = (
            m._DynamicDiagonalParametersBehaviour__get_diagonal_padding_shape()
        )
        padding_size = abs(c.input_dim - c.output_dim)
        diagonal_padding_shape = (0, 0, 0, padding_size)

        self.assertEqual(padding_shape, diagonal_padding_shape)

    def test__get_diagonal_padding_shape__input_dim_smaller_than_output_dim(self):
        c = copy.deepcopy(self.cfg)
        c.input_dim = 10
        c.output_dim = 15
        weight_shape = (c.input_dim, c.output_dim)
        weight_params = torch.arange(prod(weight_shape)).reshape(weight_shape).float()

        m = DynamicDiagonalParametersBehaviour(
            c.input_dim,
            c.output_dim,
            weight_params=weight_params,
            anti_diagonal_flag=True,
        )
        padding_shape = (
            m._DynamicDiagonalParametersBehaviour__get_diagonal_padding_shape()
        )
        padding_size = abs(c.input_dim - c.output_dim)
        diagonal_padding_shape = (0, padding_size, 0, 0)

        self.assertEqual(padding_shape, diagonal_padding_shape)

    def test__init_diagonal_models__anti_diagonal_flag__False(self):
        c = copy.deepcopy(self.cfg)
        weight_shape = (c.input_dim, c.output_dim)
        weight_params = torch.arange(prod(weight_shape)).reshape(weight_shape).float()

        m = DynamicDiagonalParametersBehaviour(
            c.input_dim,
            c.output_dim,
            weight_params=weight_params,
            anti_diagonal_flag=False,
            dynamic_bias_flag=False,
        )
        diagonal_model, anti_diagonal_model, bias_model = (
            m._DynamicDiagonalParametersBehaviour__init_diagonal_models()
        )

        self.assertIsInstance(diagonal_model, (nn.Linear, nn.Sequential))
        self.assertIsNone(anti_diagonal_model)
        self.assertIsNone(bias_model)

    def test__init_diagonal_models__anti_diagonal_flag__True(self):
        c = copy.deepcopy(self.cfg)
        weight_shape = (c.input_dim, c.output_dim)
        weight_params = torch.arange(prod(weight_shape)).reshape(weight_shape).float()

        m = DynamicDiagonalParametersBehaviour(
            c.input_dim,
            c.output_dim,
            weight_params=weight_params,
            anti_diagonal_flag=True,
        )
        diagonal_model, anti_diagonal_model, bias_model = (
            m._DynamicDiagonalParametersBehaviour__init_diagonal_models()
        )

        self.assertIsInstance(diagonal_model, (nn.Linear, nn.Sequential))
        self.assertIsInstance(anti_diagonal_model, (nn.Linear, nn.Sequential))
        self.assertIsNone(bias_model)

    def test__init_diagonal_models__dynamic_bias_flag__True(self):
        c = copy.deepcopy(self.cfg)
        weight_shape = (c.input_dim, c.output_dim)
        weight_params = torch.arange(prod(weight_shape)).reshape(weight_shape).float()

        m = DynamicDiagonalParametersBehaviour(
            c.input_dim,
            c.output_dim,
            weight_params=weight_params,
            anti_diagonal_flag=True,
            dynamic_bias_flag=True,
        )
        diagonal_model, anti_diagonal_model, bias_model = (
            m._DynamicDiagonalParametersBehaviour__init_diagonal_models()
        )

        self.assertIsInstance(diagonal_model, (nn.Linear, nn.Sequential))
        self.assertIsInstance(anti_diagonal_model, (nn.Linear, nn.Sequential))
        self.assertIsInstance(bias_model, (nn.Linear, nn.Sequential))

    def test__convert_to_diagonal_matrix__padding_shape__None(self):
        c = copy.deepcopy(self.cfg)
        c.input_dim = 4
        c.output_dim = 4
        batch_size = 2
        weight_shape = (c.input_dim, c.output_dim)
        weight_params = torch.arange(prod(weight_shape)).reshape(weight_shape).float()

        m = DynamicDiagonalParametersBehaviour(
            c.input_dim,
            c.output_dim,
            weight_params=weight_params,
        )

        vector_matrix_shape = (batch_size, c.output_dim)
        vector_matrix = (
            torch.arange(prod(vector_matrix_shape)).reshape(vector_matrix_shape).float()
        )

        output = m._DynamicDiagonalParametersBehaviour__convert_to_diagonal_matrix(
            vector_matrix
        )

        self.assertEqual(list(output.shape), [batch_size, c.output_dim, c.output_dim])

    def test__convert_to_diagonal_matrix__padding_shape__not_None(self):
        c = copy.deepcopy(self.cfg)
        c.input_dim = 4
        c.output_dim = 6
        batch_size = 2
        weight_shape = (c.input_dim, c.output_dim)
        weight_params = torch.arange(prod(weight_shape)).reshape(weight_shape).float()

        m = DynamicDiagonalParametersBehaviour(
            c.input_dim,
            c.output_dim,
            weight_params=weight_params,
        )

        vector_matrix_shape = (batch_size, c.output_dim)
        vector_matrix = (
            torch.arange(prod(vector_matrix_shape)).reshape(vector_matrix_shape).float()
        )

        output = m._DynamicDiagonalParametersBehaviour__convert_to_diagonal_matrix(
            vector_matrix
        )
        diff = abs(c.input_dim - c.output_dim)

        self.assertEqual(
            list(output.shape), [batch_size, c.output_dim, c.output_dim + diff]
        )

    def test__add_anti_diagonal_matrix__anti_diagonal_flag__False(self):
        c = copy.deepcopy(self.cfg)
        batch_size = 2
        weight_shape = (c.input_dim, c.output_dim)
        weight_params = torch.arange(prod(weight_shape)).reshape(weight_shape).float()

        m = DynamicDiagonalParametersBehaviour(
            c.input_dim,
            c.output_dim,
            weight_params=weight_params,
            anti_diagonal_flag=False,
        )

        logits_shape = (batch_size, c.output_dim)
        logits = torch.arange(prod(logits_shape)).reshape(logits_shape).float()

        output = m._DynamicDiagonalParametersBehaviour__add_anti_diagonal_matrix(
            logits, weight_params
        )

        self.assertTrue(torch.allclose(weight_params, output))

    def test__add_anti_diagonal_matrix__anti_diagonal_flag__True(self):
        c = copy.deepcopy(self.cfg)
        batch_size = 2
        weight_shape = (c.input_dim, c.output_dim)
        weight_params = torch.arange(prod(weight_shape)).reshape(weight_shape).float()

        m = DynamicDiagonalParametersBehaviour(
            c.input_dim,
            c.output_dim,
            anti_diagonal_flag=True,
            dynamic_bias_flag=False,
            weight_params=weight_params,
        )

        logits_shape = (batch_size, c.input_dim)
        logits = torch.arange(prod(logits_shape)).reshape(logits_shape).float()

        output = m._DynamicDiagonalParametersBehaviour__add_anti_diagonal_matrix(
            logits, weight_params
        )

        self.assertEqual(list(output.shape), [batch_size, c.input_dim, c.output_dim])

    def test__add_diagonal_matrix(self):
        c = copy.deepcopy(self.cfg)
        batch_size = 2
        weight_shape = (c.input_dim, c.output_dim)
        weight_params = torch.arange(prod(weight_shape)).reshape(weight_shape).float()

        m = DynamicDiagonalParametersBehaviour(
            c.input_dim,
            c.output_dim,
            weight_params=weight_params,
        )

        logits_shape = (batch_size, c.input_dim)
        logits = torch.arange(prod(logits_shape)).reshape(logits_shape).float()

        output = m._DynamicDiagonalParametersBehaviour__add_diagonal_matrix(logits)

        self.assertEqual(list(output.shape), [batch_size, c.input_dim, c.output_dim])

    def test__maybe_update_bias_parameters__dynamic_bias_flag__False(self):
        c = copy.deepcopy(self.cfg)
        batch_size = 2
        weight_shape = (c.input_dim, c.output_dim)
        weight_params = torch.arange(prod(weight_shape)).reshape(weight_shape).float()
        bias_shape = (c.output_dim,)
        bias_params = torch.arange(prod(bias_shape)).reshape(bias_shape).float()

        m = DynamicDiagonalParametersBehaviour(
            c.input_dim,
            c.output_dim,
            weight_params=weight_params,
            bias_params=bias_params,
            dynamic_bias_flag=False,
        )

        logits_shape = (batch_size, c.input_dim)
        logits = torch.arange(prod(logits_shape)).reshape(logits_shape).float()
        output = m._DynamicDiagonalParametersBehaviour__maybe_update_bias_parameters(
            logits
        )

        self.assertTrue(torch.allclose(output, bias_params))

    def test__maybe_update_bias_parameters__dynamic_bias_flag__True(self):
        c = copy.deepcopy(self.cfg)
        batch_size = 2
        weight_shape = (c.input_dim, c.output_dim)
        weight_params = torch.arange(prod(weight_shape)).reshape(weight_shape).float()
        bias_shape = (c.output_dim,)
        bias_params = torch.arange(prod(bias_shape)).reshape(bias_shape).float()

        m = DynamicDiagonalParametersBehaviour(
            c.input_dim,
            c.output_dim,
            weight_params=weight_params,
            bias_params=bias_params,
            dynamic_bias_flag=True,
        )

        logits_shape = (batch_size, c.input_dim)
        logits = torch.arange(prod(logits_shape)).reshape(logits_shape).float()
        output = m._DynamicDiagonalParametersBehaviour__maybe_update_bias_parameters(
            logits
        )
        bias_scalars = m.bias_model(logits)
        bias_scaling_factor, bias_offset = bias_scalars.chunk(2, dim=-1)
        expected_output = bias_scaling_factor * bias_params + bias_offset

        self.assertTrue(torch.allclose(output, expected_output))

    def test__forward__with__bias_parametes(self):
        c = copy.deepcopy(self.cfg)
        batch_size = 2
        weight_shape = (c.input_dim, c.output_dim)
        weight_params = torch.arange(prod(weight_shape)).reshape(weight_shape).float()
        bias_shape = (c.output_dim,)
        bias_params = torch.arange(prod(bias_shape)).reshape(bias_shape).float()

        m = DynamicDiagonalParametersBehaviour(
            c.input_dim,
            c.output_dim,
            weight_params=weight_params,
            bias_params=bias_params,
            anti_diagonal_flag=True,
            dynamic_bias_flag=True,
        )

        logits_shape = (batch_size, c.input_dim)
        logits = torch.arange(prod(logits_shape)).reshape(logits_shape).float()
        weight_params, bias_params = m.forward(logits)
        self.assertEqual(
            list(weight_params.shape), [batch_size, c.input_dim, c.output_dim]
        )
        self.assertEqual(list(bias_params.shape), [batch_size, c.output_dim])

    def test__forward__no__bias_parametes(self):
        c = copy.deepcopy(self.cfg)
        batch_size = 2
        weight_shape = (c.input_dim, c.output_dim)
        weight_params = torch.arange(prod(weight_shape)).reshape(weight_shape).float()

        m = DynamicDiagonalParametersBehaviour(
            c.input_dim,
            c.output_dim,
            weight_params=weight_params,
            anti_diagonal_flag=True,
            dynamic_bias_flag=True,
        )

        logits_shape = (batch_size, c.input_dim)
        logits = torch.arange(prod(logits_shape)).reshape(logits_shape).float()

        weight_params, bias_params = m.forward(logits)
        self.assertListEqual(
            list(weight_params.shape), [batch_size, c.input_dim, c.output_dim]
        )
        self.assertIsNone(bias_params)

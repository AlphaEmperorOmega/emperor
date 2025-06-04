import copy
import unittest
import torch
from math import prod
import torch.nn as nn
from Emperor.components.parameter_generators.parameter_generators import (
    GeneratorParameter,
    MatrixParameter,
    ParameterGeneratorConfig,
    VectorParameter,
)
from Emperor.components.parameter_generators.utils.mixture import (
    MixtureConfig,
)
from Emperor.components.parameter_generators.utils.routers import (
    RouterConfig,
    RouterModel,
    VectorRouterModel,
)
from Emperor.components.parameter_generators.utils.samplers import SamplerConfig
from Emperor.config import ModelConfig


class TestVectorParameter(unittest.TestCase):
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

        # PARAMETER GENRETOR SAMPLER OPITONS
        SAMPLER_TOP_K = 3
        SAMPLER_THRESHOLD = 0.1
        SAMPLER_DYNAMIC_TOPK_THRESHOLD = (1 / ROUTER_OUTPUT_DIM) * 1e-6
        SAMPLER_NUM_TOPK_SAMPLES = 0
        SAMPLER_NORMALIZE_PROBABILITIES_FLAG = False
        SAMPLER_NOISY_TOPK_FLAG = ROUTER_NOISY_TOPK_FLAG
        SAMPLER_ROUTER_OUTPUT_DIM = ROUTER_OUTPUT_DIM
        SAMPLER_BOOLEAN_MASK_FLAG = False

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
            coefficient_of_variation_loss_weight=COEFFICIENT_OF_VARIATION_LOSS_WEIGHT,
            switch_loss_weight=SWITCH_LOSS_WEIGHT,
            zero_centered_loss_weight=ZERO_CENTERED_LOSS_WEIGHT,
            mutual_information_loss_weight=MUTUAL_INFORMATION_LOSS_WEIGHT,
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
                dynamic_topk_threshold=SAMPLER_DYNAMIC_TOPK_THRESHOLD,
                num_topk_samples=SAMPLER_NUM_TOPK_SAMPLES,
                normalize_probabilities_flag=SAMPLER_NORMALIZE_PROBABILITIES_FLAG,
                noisy_topk_flag=SAMPLER_NOISY_TOPK_FLAG,
                router_output_dim=SAMPLER_ROUTER_OUTPUT_DIM,
                boolean_mask_flag=SAMPLER_BOOLEAN_MASK_FLAG,
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
            parameter_generator_model_config=ParameterGeneratorConfig(
                bias_parameters_flag=PARAMETER_GENERATOR_BIAS_PARAMETER_FLAG,
            ),
        )

        self.parameter_generator_cfg = self.cfg.mixture_model_config

    def test__init_with_main_config(self):
        m = VectorParameter(self.cfg)

        self.assertEqual(
            m.bias_parameters_flag, self.parameter_generator_cfg.bias_parameters_flag
        )

    def test__init_bias_router_model__bias_parameters_flag__False(self):
        c = copy.deepcopy(self.cfg)
        orverrides = ParameterGeneratorConfig(
            bias_parameters_flag=False,
        )
        m = VectorParameter(c, orverrides)

        model = m._init_bias_router_model(c)
        self.assertIsNone(model)

    def test__init_bias_router_model__bias_parameters_flag__True(self):
        c = copy.deepcopy(self.cfg)
        orverrides = ParameterGeneratorConfig(
            bias_parameters_flag=True,
        )
        m = VectorParameter(c, orverrides)

        model = m._init_bias_router_model(c)
        self.assertIsInstance(model, VectorRouterModel)

    def test__compute_logits__compute_bias_flag__False(self):
        c = copy.deepcopy(self.cfg)
        orverrides = ParameterGeneratorConfig(
            bias_parameters_flag=True,
        )
        m = VectorParameter(c, orverrides)

        batch_size = 2
        input_batch_shape = (batch_size, self.cfg.router_model_config.input_dim)
        input_batch = (
            torch.arange(prod(input_batch_shape)).reshape(input_batch_shape).float()
        )

        output = m._compute_logits(input_batch, compute_bias_flag=False)

        self.assertEqual(
            list(output.shape),
            [
                c.router_model_config.input_dim,
                batch_size,
                c.router_model_config.output_dim,
            ],
        )

    def test__compute_logits__compute_bias_flag__True(self):
        c = copy.deepcopy(self.cfg)
        orverrides = ParameterGeneratorConfig(
            bias_parameters_flag=True,
        )
        m = VectorParameter(c, orverrides)

        batch_size = 2
        input_batch_shape = (batch_size, self.cfg.router_model_config.input_dim)
        input_batch = (
            torch.arange(prod(input_batch_shape)).reshape(input_batch_shape).float()
        )

        output = m._compute_logits(input_batch, compute_bias_flag=True)

        self.assertEqual(
            list(output.shape),
            [
                c.router_model_config.output_dim,
                batch_size,
                c.router_model_config.output_dim,
            ],
        )

    def test__compute_probabilities_and_indices__compute_bias_flag__False(self):
        c = copy.deepcopy(self.cfg)
        m = VectorParameter(c)

        batch_size = 2
        input_batch_shape = (batch_size, self.cfg.router_model_config.input_dim)
        input_batch = (
            torch.arange(prod(input_batch_shape)).reshape(input_batch_shape).float()
        )

        probabilities, indices = m._compute_probabilities_and_indices(input_batch)
        self.assertEqual(
            list(probabilities.shape),
            [
                c.mixture_model_config.input_dim,
                batch_size,
                c.sampler_model_config.top_k,
            ],
        )
        self.assertEqual(
            list(indices.shape),
            [
                c.mixture_model_config.input_dim,
                batch_size,
                c.sampler_model_config.top_k,
            ],
        )

    def test__compute_probabilities_and_indices__compute_bias_flag__True(self):
        c = copy.deepcopy(self.cfg)
        c.parameter_generator_model_config.bias_parameters_flag = True
        m = VectorParameter(c)

        batch_size = 2
        input_batch_shape = (batch_size, self.cfg.router_model_config.input_dim)
        input_batch = (
            torch.arange(prod(input_batch_shape)).reshape(input_batch_shape).float()
        )

        probabilities, indices = m._compute_probabilities_and_indices(
            input_batch,
            compute_bias_flag=c.parameter_generator_model_config.bias_parameters_flag,
        )
        self.assertEqual(
            list(probabilities.shape),
            [
                c.mixture_model_config.output_dim,
                batch_size,
                c.sampler_model_config.top_k,
            ],
        )
        self.assertEqual(
            list(indices.shape),
            [
                c.mixture_model_config.output_dim,
                batch_size,
                c.sampler_model_config.top_k,
            ],
        )

    def test__compute_bias_probabilities_and_indices__bias_parameters_flag__False(self):
        c = copy.deepcopy(self.cfg)
        orverrides = ParameterGeneratorConfig(
            bias_parameters_flag=False,
        )
        m = VectorParameter(c, orverrides)

        batch_size = 2
        input_batch_shape = (batch_size, self.cfg.router_model_config.input_dim)
        input_batch = (
            torch.arange(prod(input_batch_shape)).reshape(input_batch_shape).float()
        )

        bias_probs, bias_indices = m._compute_bias_probabilities_and_indices(
            input_batch
        )
        self.assertIsNone(bias_probs)
        self.assertIsNone(bias_indices)

    def test__compute_bias_probabilities_and_indices__bias_parameters_flag__True(self):
        c = copy.deepcopy(self.cfg)
        orverrides = ParameterGeneratorConfig(
            bias_parameters_flag=True,
        )
        m = VectorParameter(c, orverrides)

        batch_size = 2
        input_batch_shape = (batch_size, self.cfg.router_model_config.input_dim)
        input_batch = (
            torch.arange(prod(input_batch_shape)).reshape(input_batch_shape).float()
        )

        bias_probs, bias_indices = m._compute_bias_probabilities_and_indices(
            input_batch
        )
        self.assertEqual(
            list(bias_probs.shape),
            [
                c.router_model_config.output_dim,
                batch_size,
                c.sampler_model_config.top_k,
            ],
        )
        self.assertEqual(
            list(bias_indices.shape),
            [
                c.router_model_config.output_dim,
                batch_size,
                c.sampler_model_config.top_k,
            ],
        )

    def test__forward__bias_parameters_flag__False(self):
        c = copy.deepcopy(self.cfg)
        c.mixture_model_config.bias_parameters_flag = False
        c.parameter_generator_model_config.bias_parameters_flag = False
        m = VectorParameter(c)

        batch_size = 2
        input_batch_shape = (batch_size, self.cfg.router_model_config.input_dim)
        input_batch = (
            torch.arange(prod(input_batch_shape)).reshape(input_batch_shape).float()
        )

        weight_mixture, bias_mixture = m(input_batch)

        self.assertEqual(
            list(weight_mixture.shape),
            [
                batch_size,
                c.mixture_model_config.input_dim,
                c.mixture_model_config.output_dim,
            ],
        )
        self.assertIsNone(bias_mixture)

    def test__forward__bias_parameters_flag__True(self):
        c = copy.deepcopy(self.cfg)
        c.mixture_model_config.bias_parameters_flag = True
        c.parameter_generator_model_config.bias_parameters_flag = True
        m = VectorParameter(c)

        batch_size = 2
        input_batch_shape = (batch_size, self.cfg.router_model_config.input_dim)
        input_batch = (
            torch.arange(prod(input_batch_shape)).reshape(input_batch_shape).float()
        )

        weight_mixture, bias_mixture = m(input_batch)

        self.assertEqual(
            list(weight_mixture.shape),
            [
                batch_size,
                c.mixture_model_config.input_dim,
                c.mixture_model_config.output_dim,
            ],
        )
        self.assertEqual(
            list(bias_mixture.shape),
            [
                batch_size,
                self.cfg.mixture_model_config.output_dim,
            ],
        )


class TestMatrixParameter(unittest.TestCase):
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

        # PARAMETER GENRETOR SAMPLER OPITONS
        SAMPLER_TOP_K = 3
        SAMPLER_THRESHOLD = 0.1
        SAMPLER_DYNAMIC_TOPK_THRESHOLD = (1 / ROUTER_OUTPUT_DIM) * 1e-6
        SAMPLER_NUM_TOPK_SAMPLES = 0
        SAMPLER_NORMALIZE_PROBABILITIES_FLAG = False
        SAMPLER_NOISY_TOPK_FLAG = ROUTER_NOISY_TOPK_FLAG
        SAMPLER_ROUTER_OUTPUT_DIM = ROUTER_OUTPUT_DIM
        SAMPLER_BOOLEAN_MASK_FLAG = False

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
            coefficient_of_variation_loss_weight=COEFFICIENT_OF_VARIATION_LOSS_WEIGHT,
            switch_loss_weight=SWITCH_LOSS_WEIGHT,
            zero_centered_loss_weight=ZERO_CENTERED_LOSS_WEIGHT,
            mutual_information_loss_weight=MUTUAL_INFORMATION_LOSS_WEIGHT,
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
                dynamic_topk_threshold=SAMPLER_DYNAMIC_TOPK_THRESHOLD,
                num_topk_samples=SAMPLER_NUM_TOPK_SAMPLES,
                normalize_probabilities_flag=SAMPLER_NORMALIZE_PROBABILITIES_FLAG,
                noisy_topk_flag=SAMPLER_NOISY_TOPK_FLAG,
                router_output_dim=SAMPLER_ROUTER_OUTPUT_DIM,
                boolean_mask_flag=SAMPLER_BOOLEAN_MASK_FLAG,
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
            parameter_generator_model_config=ParameterGeneratorConfig(
                bias_parameters_flag=PARAMETER_GENERATOR_BIAS_PARAMETER_FLAG,
            ),
        )

        self.parameter_generator_cfg = self.cfg.mixture_model_config

    def test__init_with_main_config(self):
        m = MatrixParameter(self.cfg)

        self.assertEqual(
            m.bias_parameters_flag, self.parameter_generator_cfg.bias_parameters_flag
        )

    def test__init_bias_router_model__bias_parameters_flag__False(self):
        c = copy.deepcopy(self.cfg)
        orverrides = ParameterGeneratorConfig(
            bias_parameters_flag=False,
        )
        m = MatrixParameter(c, orverrides)

        model = m._init_bias_router_model(c)
        self.assertIsNone(model)

    def test__init_bias_router_model__bias_parameters_flag__True(self):
        c = copy.deepcopy(self.cfg)
        orverrides = ParameterGeneratorConfig(
            bias_parameters_flag=True,
        )
        m = MatrixParameter(c, orverrides)

        model = m._init_bias_router_model(c)
        self.assertIsInstance(model, RouterModel)

    def test__compute_logits__compute_bias_flag__False(self):
        c = copy.deepcopy(self.cfg)
        orverrides = ParameterGeneratorConfig(
            bias_parameters_flag=True,
        )
        m = MatrixParameter(c, orverrides)

        batch_size = 2
        input_batch_shape = (batch_size, self.cfg.router_model_config.input_dim)
        input_batch = (
            torch.arange(prod(input_batch_shape)).reshape(input_batch_shape).float()
        )

        output = m._compute_logits(input_batch, compute_bias_flag=False)

        self.assertEqual(
            list(output.shape),
            [
                batch_size,
                c.router_model_config.output_dim,
            ],
        )

    def test__compute_logits__compute_bias_flag__True(self):
        c = copy.deepcopy(self.cfg)
        orverrides = ParameterGeneratorConfig(
            bias_parameters_flag=True,
        )
        m = MatrixParameter(c, orverrides)

        batch_size = 2
        input_batch_shape = (batch_size, self.cfg.router_model_config.input_dim)
        input_batch = (
            torch.arange(prod(input_batch_shape)).reshape(input_batch_shape).float()
        )

        output = m._compute_logits(input_batch, compute_bias_flag=True)

        self.assertEqual(
            list(output.shape),
            [
                batch_size,
                c.router_model_config.output_dim,
            ],
        )

    def test__compute_probabilities_and_indices__compute_bias_flag__False(self):
        c = copy.deepcopy(self.cfg)
        m = MatrixParameter(c)

        batch_size = 2
        input_batch_shape = (batch_size, self.cfg.router_model_config.input_dim)
        input_batch = (
            torch.arange(prod(input_batch_shape)).reshape(input_batch_shape).float()
        )

        probabilities, indices = m._compute_probabilities_and_indices(input_batch)
        self.assertEqual(
            list(probabilities.shape),
            [
                batch_size,
                c.sampler_model_config.top_k,
            ],
        )
        self.assertEqual(
            list(indices.shape),
            [
                batch_size,
                c.sampler_model_config.top_k,
            ],
        )

    def test__compute_probabilities_and_indices__compute_bias_flag__True(self):
        c = copy.deepcopy(self.cfg)
        c.parameter_generator_model_config.bias_parameters_flag = True
        m = MatrixParameter(c)

        batch_size = 2
        input_batch_shape = (batch_size, self.cfg.router_model_config.input_dim)
        input_batch = (
            torch.arange(prod(input_batch_shape)).reshape(input_batch_shape).float()
        )

        probabilities, indices = m._compute_probabilities_and_indices(
            input_batch,
            compute_bias_flag=c.parameter_generator_model_config.bias_parameters_flag,
        )
        self.assertEqual(
            list(probabilities.shape),
            [
                batch_size,
                c.sampler_model_config.top_k,
            ],
        )
        self.assertEqual(
            list(indices.shape),
            [
                batch_size,
                c.sampler_model_config.top_k,
            ],
        )

    def test__compute_bias_probabilities_and_indices__bias_parameters_flag__False(self):
        c = copy.deepcopy(self.cfg)
        orverrides = ParameterGeneratorConfig(
            bias_parameters_flag=False,
        )
        m = MatrixParameter(c, orverrides)

        batch_size = 2
        input_batch_shape = (batch_size, self.cfg.router_model_config.input_dim)
        input_batch = (
            torch.arange(prod(input_batch_shape)).reshape(input_batch_shape).float()
        )

        bias_probs, bias_indices = m._compute_bias_probabilities_and_indices(
            input_batch
        )
        self.assertIsNone(bias_probs)
        self.assertIsNone(bias_indices)

    def test__compute_bias_probabilities_and_indices__bias_parameters_flag__True(self):
        c = copy.deepcopy(self.cfg)
        orverrides = ParameterGeneratorConfig(
            bias_parameters_flag=True,
        )
        m = MatrixParameter(c, orverrides)

        batch_size = 2
        input_batch_shape = (batch_size, self.cfg.router_model_config.input_dim)
        input_batch = (
            torch.arange(prod(input_batch_shape)).reshape(input_batch_shape).float()
        )

        bias_probs, bias_indices = m._compute_bias_probabilities_and_indices(
            input_batch
        )
        self.assertEqual(
            list(bias_probs.shape),
            [
                batch_size,
                c.sampler_model_config.top_k,
            ],
        )
        self.assertEqual(
            list(bias_indices.shape),
            [
                batch_size,
                c.sampler_model_config.top_k,
            ],
        )

    def test__forward__bias_parameters_flag__False(self):
        c = copy.deepcopy(self.cfg)
        c.mixture_model_config.bias_parameters_flag = False
        c.parameter_generator_model_config.bias_parameters_flag = False
        m = MatrixParameter(c)

        batch_size = 2
        input_batch_shape = (batch_size, self.cfg.router_model_config.input_dim)
        input_batch = (
            torch.arange(prod(input_batch_shape)).reshape(input_batch_shape).float()
        )

        weight_mixture, bias_mixture = m(input_batch)

        self.assertEqual(
            list(weight_mixture.shape),
            [
                batch_size,
                c.mixture_model_config.input_dim,
                c.mixture_model_config.output_dim,
            ],
        )
        self.assertIsNone(bias_mixture)

    def test__forward__bias_parameters_flag__True(self):
        c = copy.deepcopy(self.cfg)
        c.mixture_model_config.bias_parameters_flag = True
        c.parameter_generator_model_config.bias_parameters_flag = True
        m = MatrixParameter(c)

        batch_size = 2
        input_batch_shape = (batch_size, self.cfg.router_model_config.input_dim)
        input_batch = (
            torch.arange(prod(input_batch_shape)).reshape(input_batch_shape).float()
        )

        weight_mixture, bias_mixture = m(input_batch)

        self.assertEqual(
            list(weight_mixture.shape),
            [
                batch_size,
                c.mixture_model_config.input_dim,
                c.mixture_model_config.output_dim,
            ],
        )
        self.assertEqual(
            list(bias_mixture.shape),
            [
                batch_size,
                self.cfg.mixture_model_config.output_dim,
            ],
        )


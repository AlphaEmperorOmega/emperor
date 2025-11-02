import copy
import unittest
import torch
from math import prod
import torch.nn as nn

from Emperor.generators.layers import (
    GeneratorParameterLayer,
    ParameterLayerConfig,
    VectorParameterLayer,
    MatrixParameterLayer,
)
from Emperor.generators.utils.behaviours import (
    DynamicDiagonalParametersBehaviour,
)
from Emperor.generators.utils.mixture import (
    MixtureConfig,
)
from Emperor.generators.utils.routers import (
    RouterConfig,
    RouterModel,
    VectorRouterModel,
)
from Emperor.generators.utils.samplers import (
    SamplerConfig,
)
from Emperor.config import ModelConfig


class TestVectorParameterLayer(unittest.TestCase):
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
        ROUTER_DYNAMIC_LINEAR_MODEL_FLAG = False

        # PARAMETER GENRETOR SAMPLER OPITONS
        SAMPLER_TOP_K = 3
        SAMPLER_THRESHOLD = 0.0
        SAMPLER_FILTER_THRESHOLD = False
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
                num_experts=ROUTER_OUTPUT_DIM,
                noisy_topk_flag=ROUTER_NOISY_TOPK_FLAG,
                activation=ROUTER_ACTIVATION_FUNCTION,
                num_layers=ROUTER_NUM_LAYERNUM_LAYERSS,
                diagonal_model_type_flag=ROUTER_DYNAMIC_LINEAR_MODEL_FLAG,
                residual_flag=False,
            ),
            sampler_model_config=SamplerConfig(
                top_k=SAMPLER_TOP_K,
                threshold=SAMPLER_THRESHOLD,
                filter_above_threshold=SAMPLER_FILTER_THRESHOLD,
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
                time_tracker_flag=False,
                dynamic_diagonal_params_flag=False,
            ),
        )

        self.parameter_generator_cfg = self.cfg.mixture_model_config

    def test__init_with_main_config(self):
        m = VectorParameterLayer(self.cfg)

        self.assertEqual(
            m.bias_parameters_flag, self.parameter_generator_cfg.bias_parameters_flag
        )

    def test__init_bias_router_model__bias_parameters_flag__False(self):
        c = copy.deepcopy(self.cfg)
        orverrides = ParameterLayerConfig(
            bias_parameters_flag=False,
        )
        m = VectorParameterLayer(c, orverrides)

        model = m._init_bias_router_model(c)
        self.assertIsNone(model)

    def test__init_bias_router_model__bias_parameters_flag__True(self):
        c = copy.deepcopy(self.cfg)
        orverrides = ParameterLayerConfig(
            bias_parameters_flag=True,
        )
        m = VectorParameterLayer(c, orverrides)

        model = m._init_bias_router_model(c)
        self.assertIsInstance(model, VectorRouterModel)

    def test__compute_logits__compute_bias_flag__False(self):
        c = copy.deepcopy(self.cfg)
        orverrides = ParameterLayerConfig(
            bias_parameters_flag=True,
        )
        m = VectorParameterLayer(c, orverrides)

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
                c.router_model_config.num_experts,
            ],
        )

    def test__compute_logits__compute_bias_flag__True(self):
        c = copy.deepcopy(self.cfg)
        orverrides = ParameterLayerConfig(
            bias_parameters_flag=True,
        )
        m = VectorParameterLayer(c, orverrides)

        batch_size = 2
        input_batch_shape = (batch_size, self.cfg.router_model_config.input_dim)
        input_batch = (
            torch.arange(prod(input_batch_shape)).reshape(input_batch_shape).float()
        )

        output = m._compute_logits(input_batch, compute_bias_flag=True)

        self.assertEqual(
            list(output.shape),
            [
                c.router_model_config.num_experts,
                batch_size,
                c.router_model_config.num_experts,
            ],
        )

    def test__compute_probabilities_and_indices__sparse(self):
        c = copy.deepcopy(self.cfg)
        c.sampler_model_config.top_k = 1
        c.mixture_model_config.top_k = 1
        m = VectorParameterLayer(c)

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
            ],
        )
        self.assertEqual(
            list(indices.shape),
            [
                c.mixture_model_config.input_dim,
                batch_size,
            ],
        )

    def test__compute_probabilities_and_indices__topk(self):
        c = copy.deepcopy(self.cfg)
        c.sampler_model_config.top_k = 3
        c.mixture_model_config.top_k = 3
        m = VectorParameterLayer(c)

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

    def test__compute_probabilities_and_indices__full_mixture(self):
        c = copy.deepcopy(self.cfg)
        c.sampler_model_config.top_k = c.mixture_model_config.depth_dim
        c.mixture_model_config.top_k = c.mixture_model_config.depth_dim
        c.mixture_model_config.weighted_parameters_flag = True
        c.sampler_model_config.mutual_information_loss_weight = 0.0
        m = VectorParameterLayer(c)

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
                c.mixture_model_config.depth_dim,
            ],
        )
        self.assertIsNone(indices)

    def test__compute_probabilities_and_indices__compute_bias_flag__False(self):
        c = copy.deepcopy(self.cfg)
        m = VectorParameterLayer(c)

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
        m = VectorParameterLayer(c)

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
        orverrides = ParameterLayerConfig(
            bias_parameters_flag=False,
        )
        m = VectorParameterLayer(c, orverrides)

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
        orverrides = ParameterLayerConfig(
            bias_parameters_flag=True,
        )
        m = VectorParameterLayer(c, orverrides)

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
                c.router_model_config.num_experts,
                batch_size,
                c.sampler_model_config.top_k,
            ],
        )
        self.assertEqual(
            list(bias_indices.shape),
            [
                c.router_model_config.num_experts,
                batch_size,
                c.sampler_model_config.top_k,
            ],
        )

    def test__generate_parameters__bias_parameters_flag__False(self):
        c = copy.deepcopy(self.cfg)
        c.mixture_model_config.bias_parameters_flag = False
        c.parameter_generator_model_config.bias_parameters_flag = False
        m = VectorParameterLayer(c)

        batch_size = 2
        input_batch_shape = (batch_size, self.cfg.router_model_config.input_dim)
        input_batch = (
            torch.arange(prod(input_batch_shape)).reshape(input_batch_shape).float()
        )

        weight_mixture, bias_mixture = m._generate_parameters(input_batch)

        self.assertEqual(
            list(weight_mixture.shape),
            [
                batch_size,
                c.mixture_model_config.input_dim,
                c.mixture_model_config.output_dim,
            ],
        )
        self.assertIsNone(bias_mixture)

    def test__generate_parameters__bias_parameters_flag__True(self):
        c = copy.deepcopy(self.cfg)
        c.mixture_model_config.bias_parameters_flag = True
        c.parameter_generator_model_config.bias_parameters_flag = True
        m = VectorParameterLayer(c)

        batch_size = 2
        input_batch_shape = (batch_size, self.cfg.router_model_config.input_dim)
        input_batch = (
            torch.arange(prod(input_batch_shape)).reshape(input_batch_shape).float()
        )

        weight_mixture, bias_mixture = m._generate_parameters(input_batch)

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

    def test__apply_generated_biases__bias_parameters_flag__False(self):
        c = copy.deepcopy(self.cfg)
        c.mixture_model_config.bias_parameters_flag = False
        c.parameter_generator_model_config.bias_parameters_flag = False
        m = VectorParameterLayer(c)

        batch_size = 2
        weighted_inputs_shape = (batch_size, self.cfg.mixture_model_config.output_dim)
        weighted_inputs = (
            torch.arange(prod(weighted_inputs_shape))
            .reshape(weighted_inputs_shape)
            .float()
        )

        generated_biases_shape = (batch_size, self.cfg.mixture_model_config.output_dim)
        generated_biases = (
            torch.arange(prod(generated_biases_shape))
            .reshape(generated_biases_shape)
            .float()
        )

        output = m._ParameterLayerBase__apply_generated_biases(
            weighted_inputs, generated_biases
        )

        self.assertEqual(
            list(output.shape),
            [
                batch_size,
                c.mixture_model_config.output_dim,
            ],
        )
        self.assertTrue(torch.allclose(weighted_inputs, output))

    def test__apply_generated_biases__bias_parameters_flag__True(self):
        c = copy.deepcopy(self.cfg)
        c.mixture_model_config.bias_parameters_flag = True
        c.parameter_generator_model_config.bias_parameters_flag = True
        m = VectorParameterLayer(c)

        batch_size = 2
        weighted_inputs_shape = (batch_size, self.cfg.mixture_model_config.output_dim)
        weighted_inputs = (
            torch.arange(prod(weighted_inputs_shape))
            .reshape(weighted_inputs_shape)
            .float()
        )

        generated_biases_shape = (batch_size, self.cfg.mixture_model_config.output_dim)
        generated_biases = (
            torch.arange(prod(generated_biases_shape))
            .reshape(generated_biases_shape)
            .float()
        )

        output = m._ParameterLayerBase__apply_generated_biases(
            weighted_inputs, generated_biases
        )

        self.assertEqual(
            list(output.shape),
            [
                batch_size,
                c.mixture_model_config.output_dim,
            ],
        )

        for i in range(batch_size):
            expected_output = weighted_inputs[i] + generated_biases[i]
            actual_output = output[i]
            # print()
            # print(f"Input sample {i} : \n", weighted_inputs[i])
            # print(f"Selected weights: \n", generated_biases[i])
            # print("Expected result: \n", expected_output)
            # print("Actual result: \n", actual_output)
            # print()
            self.assertTrue(torch.allclose(expected_output, actual_output))

    def test__apply_generated_weights(self):
        c = copy.deepcopy(self.cfg)
        c.mixture_model_config.bias_parameters_flag = False
        c.parameter_generator_model_config.bias_parameters_flag = False
        m = VectorParameterLayer(c)

        batch_size = 2
        weighted_inputs_shape = (batch_size, self.cfg.mixture_model_config.input_dim)
        weighted_inputs = (
            torch.arange(prod(weighted_inputs_shape))
            .reshape(weighted_inputs_shape)
            .float()
        )

        generated_weights_shape = (
            batch_size,
            c.mixture_model_config.input_dim,
            c.mixture_model_config.output_dim,
        )
        generated_weights = (
            torch.arange(prod(generated_weights_shape))
            .reshape(generated_weights_shape)
            .float()
        )

        output = m._ParameterLayerBase__apply_generated_weights(
            weighted_inputs, generated_weights
        )

        self.assertEqual(
            list(output.shape),
            [
                batch_size,
                c.mixture_model_config.output_dim,
            ],
        )

        for i in range(batch_size):
            expected_output = torch.matmul(weighted_inputs[i], generated_weights[i])
            actual_output = output[i]
            # print()
            # print(f"Input sample {i} : \n", weighted_inputs[i])
            # print(f"Selected weights: \n", generated_weights[i])
            # print("Expected result: \n", expected_output)
            # print("Actual result: \n", actual_output)
            # print()
            self.assertTrue(torch.allclose(expected_output, actual_output))

    def test__compute_layer_output__bias_parameters_flag__False(self):
        c = copy.deepcopy(self.cfg)
        c.mixture_model_config.bias_parameters_flag = False
        c.parameter_generator_model_config.bias_parameters_flag = False
        m = VectorParameterLayer(c)

        batch_size = 2
        weighted_inputs_shape = (batch_size, self.cfg.mixture_model_config.input_dim)
        weighted_inputs = (
            torch.arange(prod(weighted_inputs_shape))
            .reshape(weighted_inputs_shape)
            .float()
        )
        output, updated_skip_mask, total_loss = m._compute_layer_output(weighted_inputs)

        self.assertEqual(
            list(output.shape),
            [
                batch_size,
                c.mixture_model_config.output_dim,
            ],
        )

    def test__compute_layer_output__bias_parameters_flag__True(self):
        c = copy.deepcopy(self.cfg)
        c.mixture_model_config.bias_parameters_flag = True
        c.parameter_generator_model_config.bias_parameters_flag = True
        m = VectorParameterLayer(c)

        batch_size = 2
        weighted_inputs_shape = (batch_size, self.cfg.mixture_model_config.input_dim)
        weighted_inputs = (
            torch.arange(prod(weighted_inputs_shape))
            .reshape(weighted_inputs_shape)
            .float()
        )

        output, updated_skip_mask, total_loss = m._compute_layer_output(weighted_inputs)
        self.assertEqual(
            list(output.shape),
            [
                batch_size,
                c.mixture_model_config.output_dim,
            ],
        )

    def test__create_diagonal_params_model__False(self):
        c = copy.deepcopy(self.cfg)
        overrides = ParameterLayerConfig(
            dynamic_diagonal_params_flag=False,
        )
        m = VectorParameterLayer(c, overrides)

        self.assertIsNone(m.dyagonal_params_model)

    def test__create_diagonal_params_model__True(self):
        c = copy.deepcopy(self.cfg)
        overrides = ParameterLayerConfig(
            dynamic_diagonal_params_flag=True,
        )
        m = VectorParameterLayer(c, overrides)

        self.assertIsInstance(
            m.dyagonal_params_model, DynamicDiagonalParametersBehaviour
        )

    def test__add_dynamic_diagonal_params__dynamic_diagonal_params_flag__False(self):
        c = copy.deepcopy(self.cfg)
        overrides = ParameterLayerConfig(
            dynamic_diagonal_params_flag=False,
        )
        m = VectorParameterLayer(c, overrides)

        batch_size = 2
        input_dim = c.mixture_model_config.input_dim
        output_dim = c.mixture_model_config.output_dim
        shape = (2, input_dim, output_dim)
        input_batch = torch.randn(shape)
        input_weight_params = torch.ones(shape)
        shape = (batch_size, output_dim)
        input_bias_params = torch.ones(shape)

        weight_params, bias_params = m._ParameterLayerBase__add_dynamic_diagonal_params(
            input_batch, input_weight_params, input_bias_params
        )
        self.assertTrue(torch.allclose(weight_params, input_weight_params))
        self.assertTrue(torch.allclose(bias_params, input_bias_params))

    def test__add_dynamic_diagonal_params__dynamic_diagonal_params_flag__True(self):
        c = copy.deepcopy(self.cfg)
        overrides = ParameterLayerConfig(
            dynamic_diagonal_params_flag=True,
        )
        c.mixture_model_config.input_dim = 5
        c.mixture_model_config.output_dim = 5
        m = VectorParameterLayer(c, overrides)

        batch_size = 2
        input_dim = c.mixture_model_config.input_dim
        output_dim = c.mixture_model_config.output_dim
        shape = (batch_size, input_dim)
        input_batch = torch.randn(shape)
        shape = (batch_size, input_dim, output_dim)
        input_weight_params = torch.ones(shape)
        shape = (batch_size, output_dim)
        input_bias_params = torch.ones(shape)

        weight_params, bias_params = m._ParameterLayerBase__add_dynamic_diagonal_params(
            input_batch, input_weight_params, input_bias_params
        )
        self.assertFalse(torch.all(weight_params == 1.0))
        self.assertFalse(torch.all(bias_params == 1.0))


class TestMatrixParameterLayer(unittest.TestCase):
    def setUp(self):
        # MODEL WISE CONFI
        BATCH_SIZE = 2
        INPUT_DIM = 4
        HIDDEN_DIM = 5
        OUTPUT_DIM = 6
        GATHER_FREQUENCY_FLAG = False

        # AUXILIARY LOSSES OPITONS
        COEFFICIENT_OF_VARIATION_LOSS_WEIGHT: float = 0.9
        SWITCH_LOSS_WEIGHT: float = 0.9
        ZERO_CENTERED_LOSS_WEIGHT: float = 0.9
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
        SAMPLER_THRESHOLD = 0.0
        SAMPLER_FILTER_THRESHOLD = False
        SAMPLER_DYNAMIC_TOPK_THRESHOLD = (1 / ROUTER_OUTPUT_DIM) * 1e-6
        SAMPLER_NUM_TOPK_SAMPLES = 0
        SAMPLER_NORMALIZE_PROBABILITIES_FLAG = False
        SAMPLER_NOISY_TOPK_FLAG = ROUTER_NOISY_TOPK_FLAG
        SAMPLER_ROUTER_OUTPUT_DIM = ROUTER_OUTPUT_DIM

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
                num_experts=ROUTER_OUTPUT_DIM,
                noisy_topk_flag=ROUTER_NOISY_TOPK_FLAG,
                activation=ROUTER_ACTIVATION_FUNCTION,
                num_layers=ROUTER_NUM_LAYERNUM_LAYERSS,
                diagonal_model_type_flag=ROUTER_DYNAMIC_LINEAR_MODEL_FLAG,
                residual_flag=False,
            ),
            sampler_model_config=SamplerConfig(
                top_k=SAMPLER_TOP_K,
                threshold=SAMPLER_THRESHOLD,
                num_topk_samples=SAMPLER_NUM_TOPK_SAMPLES,
                filter_above_threshold=SAMPLER_FILTER_THRESHOLD,
                normalize_probabilities_flag=SAMPLER_NORMALIZE_PROBABILITIES_FLAG,
                noisy_topk_flag=SAMPLER_NOISY_TOPK_FLAG,
                num_experts=SAMPLER_ROUTER_OUTPUT_DIM,
                coefficient_of_variation_loss_weight=COEFFICIENT_OF_VARIATION_LOSS_WEIGHT,
                switch_loss_weight=SWITCH_LOSS_WEIGHT,
                zero_centred_loss_weight=ZERO_CENTERED_LOSS_WEIGHT,
                mutual_information_loss_weight=MUTUAL_INFORMATION_LOSS_WEIGHT,
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

    def test__init_with_main_config(self):
        m = MatrixParameterLayer(self.cfg)

        self.assertEqual(
            m.bias_parameters_flag, self.parameter_generator_cfg.bias_parameters_flag
        )

    def test__init_bias_router_model__bias_parameters_flag__False(self):
        c = copy.deepcopy(self.cfg)
        orverrides = ParameterLayerConfig(
            bias_parameters_flag=False,
        )
        m = MatrixParameterLayer(c, orverrides)

        model = m._init_bias_router_model(c)
        self.assertIsNone(model)

    def test__init_bias_router_model__bias_parameters_flag__True(self):
        c = copy.deepcopy(self.cfg)
        orverrides = ParameterLayerConfig(
            bias_parameters_flag=True,
        )
        m = MatrixParameterLayer(c, orverrides)

        model = m._init_bias_router_model(c)
        self.assertIsInstance(model, RouterModel)

    def test__compute_logits__compute_bias_flag__False(self):
        c = copy.deepcopy(self.cfg)
        orverrides = ParameterLayerConfig(
            bias_parameters_flag=True,
        )
        m = MatrixParameterLayer(c, orverrides)

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
                c.router_model_config.num_experts,
            ],
        )

    def test__compute_logits__compute_bias_flag__True(self):
        c = copy.deepcopy(self.cfg)
        orverrides = ParameterLayerConfig(
            bias_parameters_flag=True,
        )
        m = MatrixParameterLayer(c, orverrides)

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
                c.router_model_config.num_experts,
            ],
        )

    def test__compute_probabilities_and_indices__compute_bias_flag__False(self):
        c = copy.deepcopy(self.cfg)
        m = MatrixParameterLayer(c)

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
        m = MatrixParameterLayer(c)

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
        orverrides = ParameterLayerConfig(
            bias_parameters_flag=False,
        )
        m = MatrixParameterLayer(c, orverrides)

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
        orverrides = ParameterLayerConfig(
            bias_parameters_flag=True,
        )
        m = MatrixParameterLayer(c, orverrides)

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

    def test__generate_parameters__bias_parameters_flag__False(self):
        c = copy.deepcopy(self.cfg)
        c.mixture_model_config.bias_parameters_flag = False
        c.parameter_generator_model_config.bias_parameters_flag = False
        m = MatrixParameterLayer(c)

        batch_size = 2
        input_batch_shape = (batch_size, self.cfg.router_model_config.input_dim)
        input_batch = (
            torch.arange(prod(input_batch_shape)).reshape(input_batch_shape).float()
        )

        weight_mixture, bias_mixture = m._generate_parameters(input_batch)

        self.assertEqual(
            list(weight_mixture.shape),
            [
                batch_size,
                c.mixture_model_config.input_dim,
                c.mixture_model_config.output_dim,
            ],
        )
        self.assertIsNone(bias_mixture)

    def test__generate_parameters__bias_parameters_flag__True(self):
        c = copy.deepcopy(self.cfg)
        c.mixture_model_config.bias_parameters_flag = True
        c.parameter_generator_model_config.bias_parameters_flag = True
        m = MatrixParameterLayer(c)

        batch_size = 2
        input_batch_shape = (batch_size, self.cfg.router_model_config.input_dim)
        input_batch = (
            torch.arange(prod(input_batch_shape)).reshape(input_batch_shape).float()
        )

        weight_mixture, bias_mixture = m._generate_parameters(input_batch)

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

    def test__apply_generated_biases__bias_parameters_flag__False(self):
        c = copy.deepcopy(self.cfg)
        c.mixture_model_config.bias_parameters_flag = False
        c.parameter_generator_model_config.bias_parameters_flag = False
        m = MatrixParameterLayer(c)

        batch_size = 2
        weighted_inputs_shape = (batch_size, self.cfg.mixture_model_config.output_dim)
        weighted_inputs = (
            torch.arange(prod(weighted_inputs_shape))
            .reshape(weighted_inputs_shape)
            .float()
        )

        generated_biases_shape = (batch_size, self.cfg.mixture_model_config.output_dim)
        generated_biases = (
            torch.arange(prod(generated_biases_shape))
            .reshape(generated_biases_shape)
            .float()
        )

        output = m._ParameterLayerBase__apply_generated_biases(
            weighted_inputs, generated_biases
        )

        self.assertEqual(
            list(output.shape),
            [
                batch_size,
                c.mixture_model_config.output_dim,
            ],
        )
        self.assertTrue(torch.allclose(weighted_inputs, output))

    def test__apply_generated_biases__bias_parameters_flag__True(self):
        c = copy.deepcopy(self.cfg)
        c.mixture_model_config.bias_parameters_flag = True
        c.parameter_generator_model_config.bias_parameters_flag = True
        m = MatrixParameterLayer(c)

        batch_size = 2
        weighted_inputs_shape = (batch_size, self.cfg.mixture_model_config.output_dim)
        weighted_inputs = (
            torch.arange(prod(weighted_inputs_shape))
            .reshape(weighted_inputs_shape)
            .float()
        )

        generated_biases_shape = (batch_size, self.cfg.mixture_model_config.output_dim)
        generated_biases = (
            torch.arange(prod(generated_biases_shape))
            .reshape(generated_biases_shape)
            .float()
        )

        output = m._ParameterLayerBase__apply_generated_biases(
            weighted_inputs, generated_biases
        )

        self.assertEqual(
            list(output.shape),
            [
                batch_size,
                c.mixture_model_config.output_dim,
            ],
        )

        for i in range(batch_size):
            expected_output = weighted_inputs[i] + generated_biases[i]
            actual_output = output[i]
            # print()
            # print(f"Input sample {i} : \n", weighted_inputs[i])
            # print(f"Selected weights: \n", generated_biases[i])
            # print("Expected result: \n", expected_output)
            # print("Actual result: \n", actual_output)
            # print()
            self.assertTrue(torch.allclose(expected_output, actual_output))

    def test__apply_generated_weights(self):
        c = copy.deepcopy(self.cfg)
        m = MatrixParameterLayer(c)

        batch_size = 2
        weighted_inputs_shape = (batch_size, self.cfg.mixture_model_config.input_dim)
        weighted_inputs = (
            torch.arange(prod(weighted_inputs_shape))
            .reshape(weighted_inputs_shape)
            .float()
        )

        generated_weights_shape = (
            batch_size,
            c.mixture_model_config.input_dim,
            c.mixture_model_config.output_dim,
        )
        generated_weights = (
            torch.arange(prod(generated_weights_shape))
            .reshape(generated_weights_shape)
            .float()
        )

        output = m._ParameterLayerBase__apply_generated_weights(
            weighted_inputs, generated_weights
        )

        self.assertEqual(
            list(output.shape),
            [
                batch_size,
                c.mixture_model_config.output_dim,
            ],
        )

        for i in range(batch_size):
            expected_output = torch.matmul(weighted_inputs[i], generated_weights[i])
            actual_output = output[i]
            # print()
            # print(f"Input sample {i} : \n", weighted_inputs[i])
            # print(f"Selected weights: \n", generated_weights[i])
            # print("Expected result: \n", expected_output)
            # print("Actual result: \n", actual_output)
            # print()
            self.assertTrue(torch.allclose(expected_output, actual_output))

    def test__compute_layer_output__bias_parameters_flag__False(self):
        c = copy.deepcopy(self.cfg)
        c.mixture_model_config.bias_parameters_flag = False
        c.parameter_generator_model_config.bias_parameters_flag = False
        m = MatrixParameterLayer(c)

        batch_size = 2
        weighted_inputs_shape = (batch_size, self.cfg.mixture_model_config.input_dim)
        weighted_inputs = (
            torch.arange(prod(weighted_inputs_shape))
            .reshape(weighted_inputs_shape)
            .float()
        )
        output, updated_skip_mask, total_loss = m._compute_layer_output(weighted_inputs)

        self.assertEqual(
            list(output.shape),
            [
                batch_size,
                c.mixture_model_config.output_dim,
            ],
        )

    def test__compute_layer_output__bias_parameters_flag__True(self):
        c = copy.deepcopy(self.cfg)
        c.mixture_model_config.bias_parameters_flag = True
        c.parameter_generator_model_config.bias_parameters_flag = True
        m = MatrixParameterLayer(c)

        batch_size = 2
        weighted_inputs_shape = (batch_size, self.cfg.mixture_model_config.input_dim)
        weighted_inputs = (
            torch.arange(prod(weighted_inputs_shape))
            .reshape(weighted_inputs_shape)
            .float()
        )
        output, updated_skip_mask, total_loss = m._compute_layer_output(weighted_inputs)

        self.assertEqual(
            list(output.shape),
            [
                batch_size,
                c.mixture_model_config.output_dim,
            ],
        )

    def test__create_diagonal_params_model__False(self):
        c = copy.deepcopy(self.cfg)
        overrides = ParameterLayerConfig(
            dynamic_diagonal_params_flag=False,
        )
        m = MatrixParameterLayer(c, overrides)

        self.assertIsNone(m.dyagonal_params_model)

    def test__create_diagonal_params_model__True(self):
        c = copy.deepcopy(self.cfg)
        overrides = ParameterLayerConfig(
            dynamic_diagonal_params_flag=True,
        )
        m = MatrixParameterLayer(c, overrides)

        self.assertIsInstance(
            m.dyagonal_params_model, DynamicDiagonalParametersBehaviour
        )

    def test__add_dynamic_diagonal_params__dynamic_diagonal_params_flag__False(self):
        c = copy.deepcopy(self.cfg)
        overrides = ParameterLayerConfig(
            dynamic_diagonal_params_flag=False,
        )
        m = MatrixParameterLayer(c, overrides)

        batch_size = 2
        input_dim = c.mixture_model_config.input_dim
        output_dim = c.mixture_model_config.output_dim
        shape = (2, input_dim, output_dim)
        input_batch = torch.randn(shape)
        input_weight_params = torch.ones(shape)
        shape = (batch_size, output_dim)
        input_bias_params = torch.ones(shape)

        weight_params, bias_params = m._ParameterLayerBase__add_dynamic_diagonal_params(
            input_batch, input_weight_params, input_bias_params
        )
        self.assertTrue(torch.allclose(weight_params, input_weight_params))
        self.assertTrue(torch.allclose(bias_params, input_bias_params))

    def test__add_dynamic_diagonal_params__dynamic_diagonal_params_flag__True(self):
        c = copy.deepcopy(self.cfg)
        overrides = ParameterLayerConfig(
            dynamic_diagonal_params_flag=True,
        )
        c.mixture_model_config.input_dim = 5
        c.mixture_model_config.output_dim = 5
        m = MatrixParameterLayer(c, overrides)

        batch_size = 2
        input_dim = c.mixture_model_config.input_dim
        output_dim = c.mixture_model_config.output_dim
        shape = (batch_size, input_dim)
        input_batch = torch.randn(shape)
        shape = (batch_size, input_dim, output_dim)
        input_weight_params = torch.ones(shape)
        shape = (batch_size, output_dim)
        input_bias_params = torch.ones(shape)

        weight_params, bias_params = m._ParameterLayerBase__add_dynamic_diagonal_params(
            input_batch, input_weight_params, input_bias_params
        )
        self.assertFalse(torch.all(weight_params == 1.0))
        self.assertFalse(torch.all(bias_params == 1.0))


class TestGeneratorParameterLayer(unittest.TestCase):
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
        ROUTER_DYNAMIC_LINEAR_MODEL_FLAG = True

        # PARAMETER GENRETOR SAMPLER OPITONS
        SAMPLER_TOP_K = 3
        SAMPLER_THRESHOLD = 0.0
        SAMPLER_FILTER_THRESHOLD = False
        SAMPLER_DYNAMIC_TOPK_THRESHOLD = (1 / ROUTER_OUTPUT_DIM) * 1e-6
        SAMPLER_NUM_TOPK_SAMPLES = 0
        SAMPLER_NORMALIZE_PROBABILITIES_FLAG = False
        SAMPLER_NOISY_TOPK_FLAG = ROUTER_NOISY_TOPK_FLAG
        SAMPLER_ROUTER_OUTPUT_DIM = ROUTER_OUTPUT_DIM

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
                num_experts=ROUTER_OUTPUT_DIM,
                noisy_topk_flag=ROUTER_NOISY_TOPK_FLAG,
                activation=ROUTER_ACTIVATION_FUNCTION,
                num_layers=ROUTER_NUM_LAYERNUM_LAYERSS,
                diagonal_model_type_flag=ROUTER_DYNAMIC_LINEAR_MODEL_FLAG,
                residual_flag=False,
            ),
            sampler_model_config=SamplerConfig(
                top_k=SAMPLER_TOP_K,
                threshold=SAMPLER_THRESHOLD,
                filter_above_threshold=SAMPLER_FILTER_THRESHOLD,
                num_topk_samples=SAMPLER_NUM_TOPK_SAMPLES,
                normalize_probabilities_flag=SAMPLER_NORMALIZE_PROBABILITIES_FLAG,
                noisy_topk_flag=SAMPLER_NOISY_TOPK_FLAG,
                num_experts=SAMPLER_ROUTER_OUTPUT_DIM,
                coefficient_of_variation_loss_weight=0.0,
                switch_loss_weight=0.0,
                zero_centred_loss_weight=0.0,
                mutual_information_loss_weight=0.0,
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

    def test__init_with_main_config(self):
        m = GeneratorParameterLayer(self.cfg)

        self.assertEqual(
            m.bias_parameters_flag, self.parameter_generator_cfg.bias_parameters_flag
        )

    def test__init_bias_router_model__bias_parameters_flag__False(self):
        c = copy.deepcopy(self.cfg)
        orverrides = ParameterLayerConfig(
            bias_parameters_flag=False,
        )
        m = GeneratorParameterLayer(c, orverrides)

        model = m._init_bias_router_model(c)
        self.assertIsNone(model)

    def test__init_bias_router_model__bias_parameters_flag__True(self):
        c = copy.deepcopy(self.cfg)
        orverrides = ParameterLayerConfig(
            bias_parameters_flag=True,
        )
        m = GeneratorParameterLayer(c, orverrides)

        model = m._init_bias_router_model(c)
        self.assertIsInstance(model, RouterModel)

    def test__compute_logits__compute_bias_flag__False(self):
        c = copy.deepcopy(self.cfg)
        orverrides = ParameterLayerConfig(
            bias_parameters_flag=True,
        )
        m = GeneratorParameterLayer(c, orverrides)

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
                c.router_model_config.num_experts,
            ],
        )

    def test__compute_logits__compute_bias_flag__True(self):
        c = copy.deepcopy(self.cfg)
        orverrides = ParameterLayerConfig(
            bias_parameters_flag=True,
        )
        m = GeneratorParameterLayer(c, orverrides)

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
                c.router_model_config.num_experts,
            ],
        )

    def test__compute_probabilities_and_indices__compute_bias_flag__False(self):
        c = copy.deepcopy(self.cfg)
        m = GeneratorParameterLayer(c)

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
        m = GeneratorParameterLayer(c)

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
        orverrides = ParameterLayerConfig(
            bias_parameters_flag=False,
        )
        m = GeneratorParameterLayer(c, orverrides)

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
        orverrides = ParameterLayerConfig(
            bias_parameters_flag=True,
        )
        m = GeneratorParameterLayer(c, orverrides)

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

    def test__generate_parameters__bias_parameters_flag__False(self):
        c = copy.deepcopy(self.cfg)
        c.mixture_model_config.bias_parameters_flag = False
        c.parameter_generator_model_config.bias_parameters_flag = False
        m = GeneratorParameterLayer(c)

        batch_size = 2
        input_batch_shape = (batch_size, self.cfg.router_model_config.input_dim)
        input_batch = (
            torch.arange(prod(input_batch_shape)).reshape(input_batch_shape).float()
        )

        weight_mixture, bias_mixture = m._generate_parameters(input_batch)

        self.assertEqual(
            list(weight_mixture.shape),
            [
                batch_size,
                c.mixture_model_config.input_dim,
                c.mixture_model_config.output_dim,
            ],
        )
        self.assertIsNone(bias_mixture)

    def test__generate_parameters__bias_parameters_flag__True(self):
        c = copy.deepcopy(self.cfg)
        c.mixture_model_config.bias_parameters_flag = True
        c.parameter_generator_model_config.bias_parameters_flag = True
        m = GeneratorParameterLayer(c)

        batch_size = 2
        input_batch_shape = (batch_size, self.cfg.router_model_config.input_dim)
        input_batch = (
            torch.arange(prod(input_batch_shape)).reshape(input_batch_shape).float()
        )

        weight_mixture, bias_mixture = m._generate_parameters(input_batch)

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

    def test__apply_generated_biases__bias_parameters_flag__False(self):
        c = copy.deepcopy(self.cfg)
        c.mixture_model_config.bias_parameters_flag = False
        c.parameter_generator_model_config.bias_parameters_flag = False
        m = GeneratorParameterLayer(c)

        batch_size = 2
        weighted_inputs_shape = (batch_size, self.cfg.mixture_model_config.output_dim)
        weighted_inputs = (
            torch.arange(prod(weighted_inputs_shape))
            .reshape(weighted_inputs_shape)
            .float()
        )

        generated_biases_shape = (batch_size, self.cfg.mixture_model_config.output_dim)
        generated_biases = (
            torch.arange(prod(generated_biases_shape))
            .reshape(generated_biases_shape)
            .float()
        )

        output = m._ParameterLayerBase__apply_generated_biases(
            weighted_inputs, generated_biases
        )

        self.assertEqual(
            list(output.shape),
            [
                batch_size,
                c.mixture_model_config.output_dim,
            ],
        )
        self.assertTrue(torch.allclose(weighted_inputs, output))

    def test__apply_generated_biases__bias_parameters_flag__True(self):
        c = copy.deepcopy(self.cfg)
        c.mixture_model_config.bias_parameters_flag = True
        c.parameter_generator_model_config.bias_parameters_flag = True
        m = GeneratorParameterLayer(c)

        batch_size = 2
        weighted_inputs_shape = (batch_size, self.cfg.mixture_model_config.output_dim)
        weighted_inputs = (
            torch.arange(prod(weighted_inputs_shape))
            .reshape(weighted_inputs_shape)
            .float()
        )

        generated_biases_shape = (batch_size, self.cfg.mixture_model_config.output_dim)
        generated_biases = (
            torch.arange(prod(generated_biases_shape))
            .reshape(generated_biases_shape)
            .float()
        )

        output = m._ParameterLayerBase__apply_generated_biases(
            weighted_inputs, generated_biases
        )

        self.assertEqual(
            list(output.shape),
            [
                batch_size,
                c.mixture_model_config.output_dim,
            ],
        )

        for i in range(batch_size):
            expected_output = weighted_inputs[i] + generated_biases[i]
            actual_output = output[i]
            # print()
            # print(f"Input sample {i} : \n", weighted_inputs[i])
            # print(f"Selected weights: \n", generated_biases[i])
            # print("Expected result: \n", expected_output)
            # print("Actual result: \n", actual_output)
            # print()
            self.assertTrue(torch.allclose(expected_output, actual_output))

    def test__create_diagonal_params_model__False(self):
        c = copy.deepcopy(self.cfg)
        overrides = ParameterLayerConfig(
            dynamic_diagonal_params_flag=False,
        )
        m = GeneratorParameterLayer(c, overrides)

        self.assertIsNone(m.dyagonal_params_model)

    def test__create_diagonal_params_model__True(self):
        c = copy.deepcopy(self.cfg)
        overrides = ParameterLayerConfig(
            dynamic_diagonal_params_flag=True,
        )
        m = GeneratorParameterLayer(c, overrides)

        self.assertIsInstance(
            m.dyagonal_params_model, DynamicDiagonalParametersBehaviour
        )

    def test__add_dynamic_diagonal_params__dynamic_diagonal_params_flag__False(self):
        c = copy.deepcopy(self.cfg)
        overrides = ParameterLayerConfig(
            dynamic_diagonal_params_flag=False,
        )
        m = GeneratorParameterLayer(c, overrides)

        batch_size = 2
        input_dim = c.mixture_model_config.input_dim
        output_dim = c.mixture_model_config.output_dim
        shape = (2, input_dim, output_dim)
        input_batch = torch.randn(shape)
        input_weight_params = torch.ones(shape)
        shape = (batch_size, output_dim)
        input_bias_params = torch.ones(shape)

        weight_params, bias_params = m._ParameterLayerBase__add_dynamic_diagonal_params(
            input_batch, input_weight_params, input_bias_params
        )
        self.assertTrue(torch.allclose(weight_params, input_weight_params))
        self.assertTrue(torch.allclose(bias_params, input_bias_params))

    def test__add_dynamic_diagonal_params__dynamic_diagonal_params_flag__True(self):
        c = copy.deepcopy(self.cfg)
        overrides = ParameterLayerConfig(
            dynamic_diagonal_params_flag=True,
        )
        c.mixture_model_config.input_dim = 5
        c.mixture_model_config.output_dim = 5
        m = GeneratorParameterLayer(c, overrides)

        batch_size = 2
        input_dim = c.mixture_model_config.input_dim
        output_dim = c.mixture_model_config.output_dim
        shape = (batch_size, input_dim)
        input_batch = torch.randn(shape)
        shape = (batch_size, input_dim, output_dim)
        input_weight_params = torch.ones(shape)
        shape = (batch_size, output_dim)
        input_bias_params = torch.ones(shape)

        weight_params, bias_params = m._ParameterLayerBase__add_dynamic_diagonal_params(
            input_batch, input_weight_params, input_bias_params
        )
        self.assertFalse(torch.all(weight_params == 1.0))
        self.assertFalse(torch.all(bias_params == 1.0))

import copy
import unittest
import torch
import torch.nn as nn
from Emperor.config import ModelConfig
from Emperor.experts.experts import (
    ExpertsModule,
    ExpertsModuleConfig,
    MixtureOfExpertsConfig,
    MixtureOfExperts,
)
from Emperor.layers.layers import ParameterLayerConfig
from Emperor.layers.utils.base import LayerBlock
from Emperor.layers.utils.enums import ActivationFunctionOptions, LayerTypes
from Emperor.layers.utils.mixture import MixtureConfig
from Emperor.layers.utils.routers import RouterConfig
from Emperor.layers.utils.samplers import SamplerConfig


class TestExpertsModule(unittest.TestCase):
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
        SAMPLER_COEFFICIENT_OF_VARIATION_WEIGHT = 0.1
        SAMPLER_SWITCH_WEIGHT = 0.1
        SAMPLER_ZERO_CENTRED_WEIGHT = 0.1
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
            mixture_of_experts_config=MixtureOfExpertsConfig(
                weighted_parameters_flag=True,
            ),
            input_moe_layer_config=ExpertsModuleConfig(
                input_dim=ROUTER_INPUT_DIM,
                output_dim=64,
                dropout_probability=0.1,
                layer_norm_flag=True,
                activation=ActivationFunctionOptions.GELU,
                model_type=LayerTypes.DYNAMIC_BASE,
                num_experts=SAMPLER_ROUTER_OUTPUT_DIM,
            ),
            output_moe_layer_config=ExpertsModuleConfig(
                input_dim=64,
                output_dim=ROUTER_INPUT_DIM,
                dropout_probability=0.1,
                layer_norm_flag=True,
                activation=ActivationFunctionOptions.GELU,
                model_type=LayerTypes.DYNAMIC_BASE,
                num_experts=SAMPLER_ROUTER_OUTPUT_DIM,
            ),
        )

    def test__init_input_layer_with_default_config(self):
        c = copy.deepcopy(self.cfg)
        config = c.input_moe_layer_config
        m = ExpertsModule(c)

        self.assertIsInstance(m, ExpertsModule)
        self.assertEqual(m.input_dim, config.input_dim)
        self.assertEqual(m.output_dim, config.output_dim)
        self.assertEqual(m.dropout_probability, config.dropout_probability)
        self.assertEqual(m.layer_norm_flag, config.layer_norm_flag)
        self.assertEqual(m.activation, config.activation)
        self.assertEqual(m.model_type, config.model_type)
        self.assertEqual(m.num_experts, config.num_experts)

    def test__init_output_layer_with_default_config(self):
        c = copy.deepcopy(self.cfg)
        config = c.output_moe_layer_config
        m = ExpertsModule(c, is_output_layer_flag=True)

        self.assertIsInstance(m, ExpertsModule)
        self.assertEqual(m.input_dim, config.input_dim)
        self.assertEqual(m.output_dim, config.output_dim)
        self.assertEqual(m.dropout_probability, config.dropout_probability)
        self.assertEqual(m.layer_norm_flag, config.layer_norm_flag)
        self.assertEqual(m.activation, config.activation)
        self.assertEqual(m.model_type, config.model_type)
        self.assertEqual(m.num_experts, config.num_experts)

    def test__resolve_config_type(self):
        c = copy.deepcopy(self.cfg)

        m_input = ExpertsModule(c)
        input_config_type = m_input._ExpertsModule__resolve_config_type()
        m_output = ExpertsModule(c, is_output_layer_flag=True)
        output_config_type = m_output._ExpertsModule__resolve_config_type()

        self.assertEqual(m_input.cfg, c.input_moe_layer_config)
        self.assertEqual(m_output.cfg, c.output_moe_layer_config)
        self.assertEqual(input_config_type, "input_moe_layer_config")
        self.assertEqual(output_config_type, "output_moe_layer_config")

    def test__create_experts(self):
        c = copy.deepcopy(self.cfg)
        m = ExpertsModule(c)

        expert_models = m._ExpertsModule__create_experts(c)

        self.assertEqual(len(m.expert_modules), m.num_experts)
        for expert in expert_models:
            self.assertIsInstance(expert, LayerBlock)

    def test__create_experts_with_different_dimensions(self):
        c = copy.deepcopy(self.cfg)
        c.input_moe_layer_config.input_dim = 64
        c.input_moe_layer_config.output_dim = 128
        c.input_moe_layer_config.num_experts = 8
        config = c.input_moe_layer_config

        m = ExpertsModule(c)
        expert_models = m._ExpertsModule__create_experts(c)
        self.assertEqual(len(m.expert_modules), m.num_experts)
        for expert in expert_models:
            input_dim, output_dim = expert.model.weight_params.shape
            bias_dim = (
                expert.model.bias_params.shape[0]
                if expert.model.bias_params is not None
                else None
            )
            self.assertIsInstance(expert, LayerBlock)
            self.assertEqual(input_dim, config.input_dim)
            self.assertEqual(output_dim, config.output_dim)
            if expert.model.bias_params is not None:
                self.assertEqual(bias_dim, config.output_dim)

    def test__create_experts_with_different_dimensions__LinearLayer(self):
        c = copy.deepcopy(self.cfg)
        c.input_moe_layer_config.model_type = LayerTypes.BASE
        config = c.input_moe_layer_config

        m = ExpertsModule(c)
        expert_models = m._ExpertsModule__create_experts(c)
        self.assertEqual(len(m.expert_modules), m.num_experts)
        for expert in expert_models:
            input_dim, output_dim = expert.model.weight_params.shape
            bias_dim = (
                expert.model.bias_params.shape[0]
                if expert.model.bias_params is not None
                else None
            )
            self.assertIsInstance(expert, LayerBlock)
            self.assertEqual(input_dim, config.input_dim)
            self.assertEqual(output_dim, config.output_dim)
            if expert.model.bias_params is not None:
                self.assertEqual(bias_dim, config.output_dim)

    def test__create_experts_with_different_dimensions__DynamicDiagonalLinearLayer(
        self,
    ):
        c = copy.deepcopy(self.cfg)
        c.input_moe_layer_config.model_type = LayerTypes.DYNAMIC_BASE
        config = c.input_moe_layer_config

        m = ExpertsModule(c)
        expert_models = m._ExpertsModule__create_experts(c)
        self.assertEqual(len(m.expert_modules), m.num_experts)
        for expert in expert_models:
            input_dim, output_dim = expert.model.weight_params.shape
            bias_dim = (
                expert.model.bias_params.shape[0]
                if expert.model.bias_params is not None
                else None
            )
            self.assertIsInstance(expert, LayerBlock)
            self.assertEqual(input_dim, config.input_dim)
            self.assertEqual(output_dim, config.output_dim)
            if expert.model.bias_params is not None:
                self.assertEqual(bias_dim, config.output_dim)

    def test__create_experts_with_different_dimensions__VectorParameterLayer(self):
        c = copy.deepcopy(self.cfg)
        c.input_moe_layer_config.model_type = LayerTypes.VECTOR
        config = c.input_moe_layer_config

        m = ExpertsModule(c)
        expert_models = m._ExpertsModule__create_experts(c)
        self.assertEqual(len(m.expert_modules), m.num_experts)
        for expert in expert_models:
            input_dim, _, output_dim = expert.model.mixture.weight_bank.shape
            self.assertIsInstance(expert, LayerBlock)
            self.assertEqual(input_dim, config.input_dim)
            self.assertEqual(output_dim, config.output_dim)

    def test__create_experts_with_different_dimensions__MatrixParameterLayer(self):
        c = copy.deepcopy(self.cfg)
        c.input_moe_layer_config.model_type = LayerTypes.MATRIX
        config = c.input_moe_layer_config

        m = ExpertsModule(c)
        expert_models = m._ExpertsModule__create_experts(c)
        self.assertEqual(len(m.expert_modules), m.num_experts)
        for expert in expert_models:
            _, input_dim, output_dim = expert.model.mixture.weight_bank.shape
            self.assertIsInstance(expert, LayerBlock)
            self.assertEqual(input_dim, config.input_dim)
            self.assertEqual(output_dim, config.output_dim)

    def test__create_experts_with_different_dimensions__GeneratorParameterLayer(self):
        c = copy.deepcopy(self.cfg)
        c.input_moe_layer_config.model_type = LayerTypes.GENERATOR
        config = c.input_moe_layer_config

        m = ExpertsModule(c)
        expert_models = m._ExpertsModule__create_experts(c)
        self.assertEqual(len(m.expert_modules), m.num_experts)
        for expert in expert_models:
            _, input_dim, _ = expert.model.mixture.input_weight_bank.shape
            _, _, output_dim = expert.model.mixture.output_weight_bank.shape
            self.assertIsInstance(expert, LayerBlock)
            self.assertEqual(input_dim, config.input_dim)
            self.assertEqual(output_dim, config.output_dim)

    def test__get_expert_indices(self):
        c = copy.deepcopy(self.cfg)
        m = ExpertsModule(c)

        top_k = c.sampler_model_config.top_k
        indices = torch.stack([torch.randperm(m.num_experts)[:top_k] for _ in range(5)])

        for expert_index in range(m.num_experts):
            output = m._ExpertsModule__get_expert_indices(indices, expert_index)
            self.assertIsInstance(output, torch.Tensor)

    def test__forward__LinearLayer(self):
        c = copy.deepcopy(self.cfg)
        config = c.input_moe_layer_config
        m = ExpertsModule(c)

        top_k = c.sampler_model_config.top_k
        batch_size = 5
        for _ in range(3):
            indices = torch.stack(
                [torch.randperm(m.num_experts)[:top_k] for _ in range(batch_size)]
            )
            input_batch = torch.randn(batch_size, config.input_dim)

            output = m.compute_expert_outputs(input_batch, indices)

            self.assertIsInstance(output, torch.Tensor)
            self.assertEqual(
                list(output.shape), [batch_size * top_k, config.output_dim]
            )

    def test__forward__DynamicDiagonalLinearLayer(self):
        c = copy.deepcopy(self.cfg)
        c.input_moe_layer_config.model_type = LayerTypes.DYNAMIC_BASE
        config = c.input_moe_layer_config
        m = ExpertsModule(c)

        top_k = c.sampler_model_config.top_k
        batch_size = 5
        for _ in range(3):
            indices = torch.stack(
                [torch.randperm(m.num_experts)[:top_k] for _ in range(batch_size)]
            )
            input_batch = torch.randn(batch_size, config.input_dim)

            output = m.compute_expert_outputs(input_batch, indices)

            self.assertIsInstance(output, torch.Tensor)
            self.assertEqual(
                list(output.shape), [batch_size * top_k, config.output_dim]
            )

    def test__forward__VectorParameterLayer(self):
        c = copy.deepcopy(self.cfg)
        c.input_moe_layer_config.model_type = LayerTypes.VECTOR
        config = c.input_moe_layer_config
        m = ExpertsModule(c)

        top_k = c.sampler_model_config.top_k
        batch_size = 5
        for _ in range(3):
            indices = torch.stack(
                [torch.randperm(m.num_experts)[:top_k] for _ in range(batch_size)]
            )
            input_batch = torch.randn(batch_size, config.input_dim)

            output = m.compute_expert_outputs(input_batch, indices)
            expert_outputs, loss = output

            self.assertIsInstance(output, tuple)
            self.assertIsInstance(expert_outputs, torch.Tensor)
            self.assertIsInstance(loss, torch.Tensor)
            self.assertTrue(loss > 0)
            self.assertEqual(
                list(expert_outputs.shape), [batch_size * top_k, config.output_dim]
            )

    def test__forward__GeneratorParameterLayer(self):
        c = copy.deepcopy(self.cfg)
        c.input_moe_layer_config.model_type = LayerTypes.MATRIX
        config = c.input_moe_layer_config
        m = ExpertsModule(c)

        top_k = c.sampler_model_config.top_k
        batch_size = 5
        for _ in range(3):
            indices = torch.stack(
                [torch.randperm(m.num_experts)[:top_k] for _ in range(batch_size)]
            )
            input_batch = torch.randn(batch_size, config.input_dim)

            output = m.compute_expert_outputs(input_batch, indices)
            expert_outputs, loss = output

            self.assertIsInstance(output, tuple)
            self.assertIsInstance(expert_outputs, torch.Tensor)
            self.assertIsInstance(loss, torch.Tensor)
            self.assertTrue(loss > 0)
            self.assertEqual(
                list(expert_outputs.shape), [batch_size * top_k, config.output_dim]
            )

    def test__forward__MatrixParameterLayer(self):
        c = copy.deepcopy(self.cfg)
        c.input_moe_layer_config.model_type = LayerTypes.GENERATOR
        config = c.input_moe_layer_config
        m = ExpertsModule(c)

        top_k = c.sampler_model_config.top_k
        batch_size = 5
        for _ in range(3):
            indices = torch.stack(
                [torch.randperm(m.num_experts)[:top_k] for _ in range(batch_size)]
            )
            input_batch = torch.randn(batch_size, config.input_dim)

            output = m.compute_expert_outputs(input_batch, indices)
            expert_outputs, loss = output

            self.assertIsInstance(output, tuple)
            self.assertIsInstance(expert_outputs, torch.Tensor)
            self.assertIsInstance(loss, torch.Tensor)
            self.assertTrue(loss > 0)
            self.assertEqual(
                list(expert_outputs.shape), [batch_size * top_k, config.output_dim]
            )


class TestMixtureOfExperts(unittest.TestCase):
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
        SAMPLER_COEFFICIENT_OF_VARIATION_WEIGHT = 0.1
        SAMPLER_SWITCH_WEIGHT = 0.1
        SAMPLER_ZERO_CENTRED_WEIGHT = 0.1
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
            mixture_of_experts_config=MixtureOfExpertsConfig(
                weighted_parameters_flag=True,
            ),
            input_moe_layer_config=ExpertsModuleConfig(
                input_dim=ROUTER_INPUT_DIM,
                output_dim=64,
                dropout_probability=0.1,
                layer_norm_flag=True,
                activation=ActivationFunctionOptions.GELU,
                model_type=LayerTypes.DYNAMIC_BASE,
                num_experts=SAMPLER_ROUTER_OUTPUT_DIM,
            ),
            output_moe_layer_config=ExpertsModuleConfig(
                input_dim=64,
                output_dim=ROUTER_INPUT_DIM,
                dropout_probability=0.1,
                layer_norm_flag=True,
                activation=ActivationFunctionOptions.GELU,
                model_type=LayerTypes.DYNAMIC_BASE,
                num_experts=SAMPLER_ROUTER_OUTPUT_DIM,
            ),
        )

    def test__init_input_layer_with_default_config(self):
        c = copy.deepcopy(self.cfg)
        config = c.mixture_of_experts_config
        m = MixtureOfExperts(c)

        self.assertIsInstance(m, MixtureOfExperts)
        self.assertEqual(m.weighted_parameters_flag, config.weighted_parameters_flag)

    def test__prepare_inputs(self):
        c = copy.deepcopy(self.cfg)
        m = MixtureOfExperts(c)

        batch_size = 5
        sequence_length = 6
        embedding_dim = 7
        input_batch = torch.randn(batch_size, sequence_length, embedding_dim)
        output, skip_mask = m._MixtureOfExperts__prepare_inputs(input_batch)
        output_combined_dims, output_embedding_dim = output.shape

        self.assertEqual(output_combined_dims, batch_size * sequence_length)
        self.assertEqual(output_embedding_dim, embedding_dim)
        self.assertIsNone(skip_mask)

    def test__prepare_inputs__with__skip_mask(self):
        c = copy.deepcopy(self.cfg)
        m = MixtureOfExperts(c)

        batch_size = 5
        sequence_length = 6
        embedding_dim = 7
        input_batch = torch.randn(batch_size, sequence_length, embedding_dim)
        skip_mask = torch.randint(0, 2, (batch_size, sequence_length))
        output, reshaped_skip_ask = m._MixtureOfExperts__prepare_inputs(
            input_batch, skip_mask
        )
        output_combined_dims, output_embedding_dim = output.shape

        self.assertEqual(output_combined_dims, batch_size * sequence_length)
        self.assertEqual(output_embedding_dim, embedding_dim)
        self.assertTrue(torch.allclose(reshaped_skip_ask, skip_mask.view(-1, 1)))

    def test__resolve_output_shape__2d_input_tensor(self):
        c = copy.deepcopy(self.cfg)
        m = MixtureOfExperts(c)

        batch_size = 5
        embedding_dim = 7
        input_batch = torch.randn(batch_size, embedding_dim)
        m._MixtureOfExperts__resolve_output_shape(input_batch)

        self.assertEqual(m.batch_size, batch_size)
        self.assertEqual(m.sequence_length, 1)
        self.assertEqual(m.output_shape, [batch_size, -1])

    def test__resolve_output_shape__3d_input_tensor(self):
        c = copy.deepcopy(self.cfg)
        m = MixtureOfExperts(c)

        batch_size = 5
        sequence_length = 6
        embedding_dim = 7
        input_batch = torch.randn(batch_size, sequence_length, embedding_dim)
        m._MixtureOfExperts__resolve_output_shape(input_batch)

        self.assertEqual(batch_size, m.batch_size)
        self.assertEqual(sequence_length, m.sequence_length)
        self.assertEqual(m.output_shape, [batch_size, sequence_length, -1])

    def test__compute_expert_mixture__weighted_parameters_flag__False(self):
        c = copy.deepcopy(self.cfg)
        overrides = MixtureOfExpertsConfig(
            weighted_parameters_flag=False,
        )
        m = MixtureOfExperts(c, overrides)

        top_k = 3
        batch_size = 5
        sequence_length = 6
        embedding_dim = 7
        input_batch = torch.randn(top_k * batch_size * sequence_length, embedding_dim)

        output = m._MixtureOfExperts__compute_expert_mixture(input_batch)

        self.assertEqual(batch_size, m.batch_size)

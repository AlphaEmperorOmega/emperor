import copy
import unittest
import torch
import torch.nn as nn
from Emperor.attention.utils.utils import (
    AttentionMask,
    AttentionProcessor,
    AttentionProjector,
    AttentionUtils,
    AttentionValidator,
)
from Emperor.config import ModelConfig
from Emperor.experts.experts import (
    MixtureOfExpertsConfig,
    MixtureOfExpertsFeedForwardConfig,
)
from Emperor.layers.layers import ParameterLayerConfig
from Emperor.layers.utils.mixture import MixtureConfig
from Emperor.layers.utils.routers import RouterConfig
from Emperor.layers.utils.samplers import SamplerConfig
from Emperor.layers.utils.enums import ActivationFunctionOptions, LayerTypes
from Emperor.attention.attention import MultiHeadAttention, MultiHeadAttentionConfig


class TestMultiHeadAttention(unittest.TestCase):
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
            mixture_of_experts_config=MixtureOfExpertsFeedForwardConfig(
                weighted_parameters_flag=True,
            ),
            input_moe_layer_config=MixtureOfExpertsConfig(
                input_dim=ROUTER_INPUT_DIM,
                output_dim=64,
                top_k=MIXTURE_TOP_K,
                dropout_probability=0.1,
                layer_norm_flag=True,
                activation=ActivationFunctionOptions.GELU,
                model_type=LayerTypes.DYNAMIC_BASE,
                num_experts=SAMPLER_ROUTER_OUTPUT_DIM,
                compute_expert_mixture_flag=False,
                weighted_parameters_flag=False,
            ),
            output_moe_layer_config=MixtureOfExpertsConfig(
                input_dim=64,
                output_dim=ROUTER_INPUT_DIM,
                top_k=MIXTURE_TOP_K,
                dropout_probability=0.1,
                layer_norm_flag=True,
                activation=ActivationFunctionOptions.GELU,
                model_type=LayerTypes.DYNAMIC_BASE,
                num_experts=SAMPLER_ROUTER_OUTPUT_DIM,
                compute_expert_mixture_flag=True,
                weighted_parameters_flag=True,
            ),
            multi_head_attention_model_config=MultiHeadAttentionConfig(
                model_type=LayerTypes.DYNAMIC_BASE,
                batch_size=BATCH_SIZE,
                num_heads=4,
                embedding_dim=64,
                target_sequence_length=16,
                source_sequence_length=32,
                target_dtype=torch.float32,
                use_separate_projection_weight=False,
                dropout_probability=0.0,
                key_value_bias_flag=False,
                zero_attention_flag=False,
                batch_first_flag=False,
                key_dim=16,
                value_dim=32,
                causal_attention_mask_flag=False,
            ),
        )

    def test__init_input_layer_with_default_config(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        m = MultiHeadAttention(c)

        self.assertIsInstance(m, MultiHeadAttention)
        self.assertEqual(m.batch_size, config.batch_size)
        self.assertEqual(m.model_type, config.model_type)
        self.assertEqual(m.num_heads, config.num_heads)
        self.assertEqual(m.embedding_dim, config.embedding_dim)
        self.assertEqual(m.target_dtype, config.target_dtype)
        self.assertEqual(m.target_sequence_length, config.target_sequence_length)
        self.assertEqual(m.source_sequence_length, config.source_sequence_length)
        self.assertEqual(
            m.use_separate_projection_weight, config.use_separate_projection_weight
        )
        self.assertEqual(m.dropout_probability, config.dropout_probability)
        self.assertEqual(m.key_value_bias_flag, config.key_value_bias_flag)
        self.assertEqual(m.zero_attention_flag, config.zero_attention_flag)
        self.assertEqual(m.batch_first_flag, config.batch_first_flag)
        self.assertEqual(m.query_dim, config.embedding_dim)
        self.assertEqual(m.key_dim, config.key_dim)
        self.assertEqual(m.value_dim, config.value_dim)

    def test__resolve_kv_dimensions__kv_zero(self):
        c = copy.deepcopy(self.cfg)
        c.multi_head_attention_model_config.key_dim = 0
        c.multi_head_attention_model_config.value_dim = 0
        config = c.multi_head_attention_model_config
        m = MultiHeadAttention(c)

        m._MultiHeadAttention__resolve_kv_dimensions()
        self.assertEqual(m.key_dim, config.embedding_dim)
        self.assertEqual(m.value_dim, config.embedding_dim)

    def test__resolve_kv_dimensions__kv_nonzero(self):
        c = copy.deepcopy(self.cfg)
        c.multi_head_attention_model_config.key_dim = 128
        c.multi_head_attention_model_config.value_dim = 256
        config = c.multi_head_attention_model_config
        m = MultiHeadAttention(c)

        m._MultiHeadAttention__resolve_head_dim()
        self.assertEqual(m.key_dim, config.key_dim)
        self.assertEqual(m.value_dim, config.value_dim)

    def test__resolve_head_dim(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        m = MultiHeadAttention(c)

        head_dim = m._MultiHeadAttention__resolve_head_dim()
        expected_head_dim = config.embedding_dim // config.num_heads
        self.assertEqual(head_dim, expected_head_dim)

    def test__resolve_head_dim__test_assertion(self):
        c = copy.deepcopy(self.cfg)
        m = MultiHeadAttention(c)
        m.num_heads = 3

        with self.assertRaises(AssertionError) as context:
            _ = m._MultiHeadAttention__resolve_head_dim()

    def test__init_attention_utils(self):
        c = copy.deepcopy(self.cfg)
        m = MultiHeadAttention(c)

        self.assertIsInstance(m.validator, AttentionValidator)
        self.assertIsInstance(m.masks, AttentionMask)
        self.assertIsInstance(m.projector, AttentionProjector)
        self.assertIsInstance(m.processor, AttentionProcessor)
        self.assertIsInstance(m.utils, AttentionUtils)

    def test__are_qkv_dimensions_equal(self):
        c = copy.deepcopy(self.cfg)
        c.multi_head_attention_model_config.embedding_dim = 32
        c.multi_head_attention_model_config.key_dim = 32
        c.multi_head_attention_model_config.value_dim = 32
        m = MultiHeadAttention(c)

        output = m._MultiHeadAttention__are_qkv_dimensions_equal()
        self.assertTrue(output)

    def test__are_qkv_dimensions_equal__different_embedding_dim(self):
        c = copy.deepcopy(self.cfg)
        c.multi_head_attention_model_config.embedding_dim = 64
        c.multi_head_attention_model_config.key_dim = 32
        c.multi_head_attention_model_config.value_dim = 32
        m = MultiHeadAttention(c)

        output = m._MultiHeadAttention__are_qkv_dimensions_equal()
        self.assertFalse(output)

    def test__are_qkv_dimensions_equal__different_key_dim(self):
        c = copy.deepcopy(self.cfg)
        c.multi_head_attention_model_config.embedding_dim = 32
        c.multi_head_attention_model_config.key_dim = 64
        c.multi_head_attention_model_config.value_dim = 32
        m = MultiHeadAttention(c)

        output = m._MultiHeadAttention__are_qkv_dimensions_equal()
        self.assertFalse(output)

    def test__are_qkv_dimensions_equal__different_value_dim(self):
        c = copy.deepcopy(self.cfg)
        c.multi_head_attention_model_config.embedding_dim = 32
        c.multi_head_attention_model_config.key_dim = 32
        c.multi_head_attention_model_config.value_dim = 64
        m = MultiHeadAttention(c)

        output = m._MultiHeadAttention__are_qkv_dimensions_equal()
        self.assertFalse(output)

    def test__create_shared_projection_models(self):
        c = copy.deepcopy(self.cfg)
        c.multi_head_attention_model_config.embedding_dim = 32
        c.multi_head_attention_model_config.key_dim = 32
        c.multi_head_attention_model_config.value_dim = 64
        m = MultiHeadAttention(c)
        qkv_model, output_model = (
            m._MultiHeadAttention__create_shared_projection_models(c)
        )

        self.assertIsNone(m.query_model)
        self.assertIsNone(m.key_model)
        self.assertIsNone(m.value_model)
        self.assertIsInstance(qkv_model, m.model_type.value)
        self.assertIsInstance(output_model, m.model_type.value)

    def test__create_independend_projection_models(self):
        c = copy.deepcopy(self.cfg)
        # The config is changed to because of `__are_qkv_dimensions_equal`
        # this needs to evaluate to True because it's called in the constructor
        c.multi_head_attention_model_config.embedding_dim = 32
        c.multi_head_attention_model_config.key_dim = 32
        c.multi_head_attention_model_config.value_dim = 32
        m = MultiHeadAttention(c)
        query_model, key_model, value_model, output_model = (
            m._MultiHeadAttention__create_independend_projection_models(c)
        )

        self.assertIsInstance(query_model, m.model_type.value)
        self.assertIsInstance(key_model, m.model_type.value)
        self.assertIsInstance(value_model, m.model_type.value)
        self.assertIsInstance(output_model, m.model_type.value)
        self.assertIsNone(m.qkv_model)


class TestAttentionValidator(unittest.TestCase):
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
            mixture_of_experts_config=MixtureOfExpertsFeedForwardConfig(
                weighted_parameters_flag=True,
            ),
            input_moe_layer_config=MixtureOfExpertsConfig(
                input_dim=ROUTER_INPUT_DIM,
                output_dim=64,
                top_k=MIXTURE_TOP_K,
                dropout_probability=0.1,
                layer_norm_flag=True,
                activation=ActivationFunctionOptions.GELU,
                model_type=LayerTypes.DYNAMIC_BASE,
                num_experts=SAMPLER_ROUTER_OUTPUT_DIM,
                compute_expert_mixture_flag=False,
                weighted_parameters_flag=False,
            ),
            output_moe_layer_config=MixtureOfExpertsConfig(
                input_dim=64,
                output_dim=ROUTER_INPUT_DIM,
                top_k=MIXTURE_TOP_K,
                dropout_probability=0.1,
                layer_norm_flag=True,
                activation=ActivationFunctionOptions.GELU,
                model_type=LayerTypes.DYNAMIC_BASE,
                num_experts=SAMPLER_ROUTER_OUTPUT_DIM,
                compute_expert_mixture_flag=True,
                weighted_parameters_flag=True,
            ),
            multi_head_attention_model_config=MultiHeadAttentionConfig(
                model_type=LayerTypes.DYNAMIC_BASE,
                batch_size=BATCH_SIZE,
                num_heads=4,
                embedding_dim=64,
                target_sequence_length=16,
                source_sequence_length=32,
                target_dtype=torch.float32,
                use_separate_projection_weight=False,
                dropout_probability=0.0,
                key_value_bias_flag=False,
                zero_attention_flag=False,
                batch_first_flag=False,
                key_dim=16,
                value_dim=32,
                causal_attention_mask_flag=False,
            ),
        )

    # def test__init_input_layer_with_default_config(self):
    #     c = copy.deepcopy(self.cfg)
    #     config = c.multi_head_attention_model_config
    #     m = AttentionValidator(c)
    #
    #     self.assertIsInstance(m, AttentionValidator)

import copy
import unittest
import torch
from torch._C import dtype
import torch.nn as nn
from torch.nn.functional import scaled_dot_product_attention
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


class TestAttention(unittest.TestCase):
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


class TestMultiHeadAttention__init(TestAttention):
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

    def test__forward(self):
        # This will be tested after all components are tested
        pass


class TestMultIHeadAttention____resolve_kv_dimensions(TestAttention):
    def test__kv_zero(self):
        c = copy.deepcopy(self.cfg)
        c.multi_head_attention_model_config.key_dim = 0
        c.multi_head_attention_model_config.value_dim = 0
        config = c.multi_head_attention_model_config
        m = MultiHeadAttention(c)

        m._MultiHeadAttention__resolve_kv_dimensions()
        self.assertEqual(m.key_dim, config.embedding_dim)
        self.assertEqual(m.value_dim, config.embedding_dim)

    def test__kv_nonzero(self):
        c = copy.deepcopy(self.cfg)
        c.multi_head_attention_model_config.key_dim = 128
        c.multi_head_attention_model_config.value_dim = 256
        config = c.multi_head_attention_model_config
        m = MultiHeadAttention(c)

        m._MultiHeadAttention__resolve_head_dim()
        self.assertEqual(m.key_dim, config.key_dim)
        self.assertEqual(m.value_dim, config.value_dim)


class TestMultIHeadAttention____resolve_head_dim(TestAttention):
    def test__computed_head_dim(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        m = MultiHeadAttention(c)

        head_dim = m._MultiHeadAttention__resolve_head_dim()
        expected_head_dim = config.embedding_dim // config.num_heads
        self.assertEqual(head_dim, expected_head_dim)

    def test__if_assertion_is_raised(self):
        c = copy.deepcopy(self.cfg)
        m = MultiHeadAttention(c)
        m.num_heads = 3

        with self.assertRaises(AssertionError) as context:
            _ = m._MultiHeadAttention__resolve_head_dim()


class TestMultIHeadAttention____initialize_attention_components(TestAttention):
    def test__ensure_componets_are_initialzied(self):
        c = copy.deepcopy(self.cfg)
        m = MultiHeadAttention(c)

        self.assertIsInstance(m.validator, AttentionValidator)
        self.assertIsInstance(m.masks, AttentionMask)
        self.assertIsInstance(m.projector, AttentionProjector)
        self.assertIsInstance(m.processor, AttentionProcessor)
        self.assertIsInstance(m.utils, AttentionUtils)


class TestMultIHeadAttention____are_qkv_dimensions_equal(TestAttention):
    def test__different_embedding_dim(self):
        c = copy.deepcopy(self.cfg)
        c.multi_head_attention_model_config.embedding_dim = 64
        c.multi_head_attention_model_config.key_dim = 32
        c.multi_head_attention_model_config.value_dim = 32
        m = MultiHeadAttention(c)

        output = m._MultiHeadAttention__are_qkv_dimensions_equal()
        self.assertFalse(output)

    def test__different_key_dim(self):
        c = copy.deepcopy(self.cfg)
        c.multi_head_attention_model_config.embedding_dim = 32
        c.multi_head_attention_model_config.key_dim = 64
        c.multi_head_attention_model_config.value_dim = 32
        m = MultiHeadAttention(c)

        output = m._MultiHeadAttention__are_qkv_dimensions_equal()
        self.assertFalse(output)

    def test__different_value_dim(self):
        c = copy.deepcopy(self.cfg)
        # This exists here to ensure because of `register_parameter` in the
        # this method is allready called when `MultiHeadAttention` is initialized
        c.multi_head_attention_model_config.embedding_dim = 32
        c.multi_head_attention_model_config.key_dim = 32
        c.multi_head_attention_model_config.value_dim = 64
        m = MultiHeadAttention(c)

        output = m._MultiHeadAttention__are_qkv_dimensions_equal()
        self.assertFalse(output)

    def test__embd_key_value_same_dim(self):
        c = copy.deepcopy(self.cfg)
        # This exists here to ensure because of `register_parameter` in the
        # this method is allready called when `MultiHeadAttention` is initialized
        c.multi_head_attention_model_config.embedding_dim = 32
        c.multi_head_attention_model_config.key_dim = 32
        c.multi_head_attention_model_config.value_dim = 32
        m = MultiHeadAttention(c)

        output = m._MultiHeadAttention__are_qkv_dimensions_equal()
        self.assertTrue(output)


class TestMultIHeadAttention____build_shared_projection_models(TestAttention):
    def test__shared_model_inizialization(self):
        c = copy.deepcopy(self.cfg)
        # This exists here to ensure because of `register_parameter` in the
        # this method is allready called when `MultiHeadAttention` is initialized
        c.multi_head_attention_model_config.embedding_dim = 32
        c.multi_head_attention_model_config.key_dim = 32
        c.multi_head_attention_model_config.value_dim = 32
        m = MultiHeadAttention(c)
        qkv_model, output_model = m._MultiHeadAttention__build_shared_projection_models(
            c
        )

        self.assertIsNone(m.query_model)
        self.assertIsNone(m.key_model)
        self.assertIsNone(m.value_model)
        self.assertIsInstance(qkv_model, m.model_type.value)
        self.assertIsInstance(output_model, m.model_type.value)


class TestMultIHeadAttention____build_separate_projection_models(TestAttention):
    def test__separate_models_initializations(self):
        c = copy.deepcopy(self.cfg)
        c.multi_head_attention_model_config.embedding_dim = 32
        c.multi_head_attention_model_config.key_dim = 64
        c.multi_head_attention_model_config.value_dim = 32
        m = MultiHeadAttention(c)
        query_model, key_model, value_model, output_model = (
            m._MultiHeadAttention__build_separate_projection_models(c)
        )

        self.assertIsInstance(query_model, m.model_type.value)
        self.assertIsInstance(key_model, m.model_type.value)
        self.assertIsInstance(value_model, m.model_type.value)
        self.assertIsInstance(output_model, m.model_type.value)
        self.assertIsNone(m.qkv_model)


class TestMultIHeadAttention____build_projection_models(TestAttention):
    def test__same_qkv_dim(self):
        c = copy.deepcopy(self.cfg)
        c.multi_head_attention_model_config.embedding_dim = 32
        c.multi_head_attention_model_config.key_dim = 32
        c.multi_head_attention_model_config.value_dim = 32
        m = MultiHeadAttention(c)

        self.assertIsInstance(m.qkv_model, m.model_type.value)
        self.assertIsInstance(m.output_model, m.model_type.value)
        self.assertIsNone(m.query_model)
        self.assertIsNone(m.key_model)
        self.assertIsNone(m.value_model)

    def test__different_qkv_dim(self):
        c = copy.deepcopy(self.cfg)
        c.multi_head_attention_model_config.embedding_dim = 32
        c.multi_head_attention_model_config.key_dim = 64
        c.multi_head_attention_model_config.value_dim = 32
        m = MultiHeadAttention(c)

        self.assertIsInstance(m.query_model, m.model_type.value)
        self.assertIsInstance(m.key_model, m.model_type.value)
        self.assertIsInstance(m.value_model, m.model_type.value)
        self.assertIsInstance(m.output_model, m.model_type.value)
        self.assertIsNone(m.qkv_model)


class TestAttentionUtils____transpose_shared_qkv(TestAttention):
    def test__same_tensor_for_kv(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        validator = AttentionValidator(config)
        m = AttentionUtils(config, validator)

        batch_size = config.batch_size
        target_sequence_length = config.target_sequence_length
        embedding_dim = config.embedding_dim

        query = key = torch.randn(target_sequence_length, batch_size, embedding_dim)

        output_q, output_k, output_v = m._AttentionUtils__transpose_shared_qkv(
            query, key
        )

        self.assertEqual(
            output_q.shape, (batch_size, target_sequence_length, embedding_dim)
        )
        self.assertEqual(
            output_k.shape, (batch_size, target_sequence_length, embedding_dim)
        )
        self.assertEqual(
            output_v.shape, (batch_size, target_sequence_length, embedding_dim)
        )
        self.assertTrue(torch.equal(output_q, output_k))
        self.assertTrue(torch.equal(output_k, output_v))

    def test__different_tensors_for_kv(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        validator = AttentionValidator(config)
        m = AttentionUtils(config, validator)

        batch_size = config.batch_size
        target_sequence_length = config.target_sequence_length
        source_sequence_length = config.target_sequence_length
        embedding_dim = config.embedding_dim

        query = torch.randn(target_sequence_length, batch_size, embedding_dim)
        key = torch.randn(source_sequence_length, batch_size, embedding_dim)

        output_q, output_k, output_v = m._AttentionUtils__transpose_shared_qkv(
            query, key
        )

        self.assertEqual(
            output_q.shape, (batch_size, target_sequence_length, embedding_dim)
        )
        self.assertEqual(
            output_k.shape, (batch_size, source_sequence_length, embedding_dim)
        )
        self.assertEqual(
            output_v.shape, (batch_size, source_sequence_length, embedding_dim)
        )
        self.assertTrue(torch.all(output_q != output_k))
        self.assertTrue(torch.all(output_q != output_v))
        self.assertTrue(torch.equal(output_k, output_v))


class TestAttentionUtils__maybe_transpose_qkv(TestAttention):
    def test__batch_first__False(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        config.batch_first_flag = False
        validator = AttentionValidator(config)
        m = AttentionUtils(config, validator)

        batch_size = config.batch_size
        target_sequence_length = config.target_sequence_length
        embedding_dim = config.embedding_dim

        query = key = value = torch.randn(
            target_sequence_length, batch_size, embedding_dim
        )

        output_q, output_k, output_v = m.maybe_transpose_qkv(query, key, value)

        self.assertTrue(torch.equal(output_q, query))
        self.assertTrue(torch.equal(output_k, key))
        self.assertTrue(torch.equal(output_v, value))

    def test__batch_first__True(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        config.batch_first_flag = True
        validator = AttentionValidator(config)
        m = AttentionUtils(config, validator)

        batch_size = config.batch_size
        target_sequence_length = config.target_sequence_length
        source_sequence_length = config.target_sequence_length
        embedding_dim = config.embedding_dim

        query = key = value = torch.randn(
            target_sequence_length, batch_size, embedding_dim
        )
        output_q, output_k, output_v = m.maybe_transpose_qkv(query, key, value)

        self.assertEqual(
            output_q.shape, (batch_size, target_sequence_length, embedding_dim)
        )
        self.assertEqual(
            output_k.shape, (batch_size, target_sequence_length, embedding_dim)
        )
        self.assertEqual(
            output_v.shape, (batch_size, target_sequence_length, embedding_dim)
        )
        self.assertTrue(torch.equal(output_q, output_k))
        self.assertTrue(torch.equal(output_k, output_v))
        self.assertTrue(torch.equal(output_q, output_v))

    def test__batch_first__True__same_kv(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        config.batch_first_flag = True
        validator = AttentionValidator(config)
        m = AttentionUtils(config, validator)

        batch_size = config.batch_size
        target_sequence_length = config.target_sequence_length
        source_sequence_length = config.target_sequence_length
        embedding_dim = config.embedding_dim

        query = torch.randn(target_sequence_length, batch_size, embedding_dim)
        key = value = torch.randn(source_sequence_length, batch_size, embedding_dim)
        output_q, output_k, output_v = m.maybe_transpose_qkv(query, key, value)

        self.assertEqual(
            output_q.shape, (batch_size, target_sequence_length, embedding_dim)
        )
        self.assertEqual(
            output_k.shape, (batch_size, source_sequence_length, embedding_dim)
        )
        self.assertEqual(
            output_v.shape, (batch_size, source_sequence_length, embedding_dim)
        )
        self.assertTrue(torch.equal(output_k, output_v))
        self.assertTrue(torch.all(output_q != output_k))
        self.assertTrue(torch.all(output_q != output_v))

    def test__batch_first__True__different__qkv(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        config.batch_first_flag = True
        validator = AttentionValidator(config)
        m = AttentionUtils(config, validator)

        batch_size = config.batch_size
        source_sequence_length = config.source_sequence_length
        target_sequence_length = config.source_sequence_length
        embedding_dim = config.embedding_dim

        query = torch.randn(target_sequence_length, batch_size, embedding_dim)
        key = torch.randn(source_sequence_length, batch_size, embedding_dim)
        value = torch.randn(source_sequence_length, batch_size, embedding_dim)
        output_q, output_k, output_v = m.maybe_transpose_qkv(query, key, value)

        self.assertEqual(
            output_q.shape, (batch_size, target_sequence_length, embedding_dim)
        )
        self.assertEqual(
            output_k.shape, (batch_size, source_sequence_length, embedding_dim)
        )
        self.assertEqual(
            output_v.shape, (batch_size, source_sequence_length, embedding_dim)
        )
        self.assertTrue(torch.all(output_k != output_v))
        self.assertTrue(torch.all(output_q != output_k))
        self.assertTrue(torch.all(output_q != output_v))


class TestAttentionUtils__add_batch_dimension_if_missing(TestAttention):
    def test__batched_input(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        config.batch_first_flag = True
        validator = AttentionValidator(config)
        m = AttentionUtils(config, validator)

        batch_size = config.batch_size
        num_heads = config.num_heads
        source_sequence_length = config.source_sequence_length
        target_sequence_length = config.source_sequence_length
        embedding_dim = config.embedding_dim

        query = torch.randn(target_sequence_length, batch_size, embedding_dim)
        key = torch.randn(source_sequence_length, batch_size, embedding_dim)
        value = torch.randn(source_sequence_length, batch_size, embedding_dim)
        key_padding_mask = torch.randint(0, 1, (batch_size, source_sequence_length))
        attention_mask = torch.randn(
            batch_size * num_heads, target_sequence_length, source_sequence_length
        )

        output_q, output_k, output_v, output_padding, output_attention_mask = (
            m.add_batch_dimension_if_missing(
                query, key, value, key_padding_mask, attention_mask
            )
        )

        self.assertEqual(query.shape, output_q.shape)
        self.assertEqual(key.shape, output_k.shape)
        self.assertEqual(value.shape, output_v.shape)
        self.assertEqual(key_padding_mask.shape, output_padding.shape)
        self.assertEqual(attention_mask.shape, output_attention_mask.shape)

    def test__non_batched_input(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        config.batch_first_flag = True
        validator = AttentionValidator(config)
        m = AttentionUtils(config, validator)

        source_sequence_length = config.source_sequence_length
        target_sequence_length = config.source_sequence_length
        embedding_dim = config.embedding_dim

        query = torch.randn(target_sequence_length, embedding_dim)
        key = torch.randn(source_sequence_length, embedding_dim)
        value = torch.randn(source_sequence_length, embedding_dim)
        key_padding_mask = torch.randint(0, 1, (source_sequence_length,))
        attention_mask = torch.randn(target_sequence_length, source_sequence_length)

        output_q, output_k, output_v, output_padding, output_attention_mask = (
            m.add_batch_dimension_if_missing(
                query, key, value, key_padding_mask, attention_mask
            )
        )

        self.assertEqual(output_q.shape, (target_sequence_length, 1, embedding_dim))
        self.assertEqual(output_k.shape, (source_sequence_length, 1, embedding_dim))
        self.assertEqual(output_v.shape, (source_sequence_length, 1, embedding_dim))
        self.assertEqual(output_padding.shape, (1, source_sequence_length))
        self.assertEqual(
            output_attention_mask.shape,
            (1, target_sequence_length, source_sequence_length),
        )


class TestAttentionValidator____check_query_dims(TestAttention):
    def test__incorrect_1D_tensor(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        m = AttentionValidator(config)

        test_dim = config.embedding_dim

        query = torch.randn(test_dim)
        with self.assertRaises(RuntimeError) as context:
            m._AttentionValidator__check_query_dims(query)

    def test__correct_2D_tensor(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        m = AttentionValidator(config)

        target_sequence_length = m.target_sequence_length
        embedding_dim = config.embedding_dim

        query = torch.randn(target_sequence_length, embedding_dim)

        output = m._AttentionValidator__check_query_dims(query)
        self.assertIsNone(output)

    def test__correct_3D_tensor(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        m = AttentionValidator(config)

        batch_size = m.batch_size
        target_sequence_length = m.target_sequence_length
        embedding_dim = config.embedding_dim

        query = torch.randn(target_sequence_length, batch_size, embedding_dim)

        output = m._AttentionValidator__check_query_dims(query)
        self.assertIsNone(output)


class TestAttentionValidator____check_query_key_value_dimensions(TestAttention):
    def test__batched_input_flag__False__incorrect_input_dim(
        self,
    ):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        m = AttentionValidator(config)
        m.batched_input_flag = False

        batch_size = m.batch_size
        source_sequence_length = m.source_sequence_length
        embedding_dim = config.embedding_dim

        key = torch.randn(source_sequence_length, batch_size, embedding_dim)
        value = torch.randn(source_sequence_length * batch_size, embedding_dim)
        with self.assertRaises(RuntimeError) as context:
            m._AttentionValidator__check_query_key_value_dimensions(key, value)

    def test__batched_input_flag__True__incorrect_input_dim(
        self,
    ):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        m = AttentionValidator(config)
        m.batched_input_flag = True

        batch_size = m.batch_size
        source_sequence_length = m.source_sequence_length
        embedding_dim = config.embedding_dim

        key = torch.randn(source_sequence_length, batch_size, embedding_dim)
        value = torch.randn(source_sequence_length * batch_size, embedding_dim)
        with self.assertRaises(RuntimeError) as context:
            m._AttentionValidator__check_query_key_value_dimensions(key, value)

    def test__batched_input_flag__False__correct_input_dim(
        self,
    ):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        m = AttentionValidator(config)
        m.batched_input_flag = False

        batch_size = m.batch_size
        source_sequence_length = m.source_sequence_length
        embedding_dim = config.embedding_dim

        key = torch.randn(source_sequence_length * batch_size, embedding_dim)
        value = torch.randn(source_sequence_length * batch_size, embedding_dim)
        output = m._AttentionValidator__check_query_key_value_dimensions(key, value)
        self.assertIsNone(output)

    def test__batched_input_flag__True__correct_input_dim(
        self,
    ):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        m = AttentionValidator(config)
        m.batched_input_flag = True

        batch_size = m.batch_size
        source_sequence_length = m.source_sequence_length
        embedding_dim = config.embedding_dim

        key = torch.randn(source_sequence_length, batch_size, embedding_dim)
        value = torch.randn(source_sequence_length, batch_size, embedding_dim)
        output = m._AttentionValidator__check_query_key_value_dimensions(key, value)
        self.assertIsNone(output)


class TestAttentionValidator____check_key_padding_mask_dimensions(TestAttention):
    def test__None_as_input(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        m = AttentionValidator(config)

        output = m._AttentionValidator__check_key_padding_mask_dimensions(None)
        self.assertIsNone(output)

    def test__incorrect_3D_key_padding_mask_dims(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        m = AttentionValidator(config)

        batch_size = m.batch_size
        source_sequence_length = m.source_sequence_length
        embedding_dim = config.embedding_dim

        key_padding_mask = torch.randn(
            source_sequence_length, batch_size, embedding_dim
        )
        with self.assertRaises(RuntimeError) as context:
            m._AttentionValidator__check_key_padding_mask_dimensions(key_padding_mask)

    def test__batched_input_flag__True__with__2D_key_padding_mask_shape(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        m = AttentionValidator(config)
        m.batched_input_flag = True

        batch_size = m.batch_size
        source_sequence_length = m.source_sequence_length

        key_padding_mask = torch.randn(batch_size, source_sequence_length)
        output = m._AttentionValidator__check_key_padding_mask_dimensions(
            key_padding_mask
        )
        self.assertIsNone(output)

    def test__batched_input_flag__False__with__1D_key_padding_mask(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        m = AttentionValidator(config)
        m.batched_input_flag = False

        source_sequence_length = m.source_sequence_length

        key_padding_mask = torch.randn(source_sequence_length)
        output = m._AttentionValidator__check_key_padding_mask_dimensions(
            key_padding_mask
        )
        self.assertIsNone(output)


class TestAttentionValidator____check_attention_mask(TestAttention):
    def test__None_as_input(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        m = AttentionValidator(config)
        m.batched_input_flag = True

        output = m._AttentionValidator__check_attention_mask(None)
        self.assertIsNone(output)

    def test__1D_incorrect_input_dims(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        m = AttentionValidator(config)
        m.batched_input_flag = True

        source_sequence_length = m.source_sequence_length

        attention_mask = torch.randn(source_sequence_length)
        with self.assertRaises(RuntimeError) as context:
            m._AttentionValidator__check_attention_mask(attention_mask)

    def test__2D__incorrect_mask_shape__correct_input_dims(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        m = AttentionValidator(config)
        m.batched_input_flag = True

        source_sequence_length = 12
        target_sequence_length = m.target_sequence_length

        attention_mask = torch.randn(source_sequence_length, target_sequence_length)

        with self.assertRaises(RuntimeError) as context:
            m._AttentionValidator__check_attention_mask(attention_mask)

    def test__2D__correct_mask_shape__correct_input_dims(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        m = AttentionValidator(config)
        m.batched_input_flag = True

        source_sequence_length = m.source_sequence_length
        target_sequence_length = m.target_sequence_length

        attention_mask = torch.randn(source_sequence_length, target_sequence_length)

        output = m._AttentionValidator__check_attention_mask(attention_mask)
        self.assertIsNone(output)

    def test__3D__incorrect_mask_shape__correct_input_dims(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        m = AttentionValidator(config)
        m.batched_input_flag = True

        batch_size = 2
        num_heads = 3
        source_sequence_length = 12
        target_sequence_length = m.target_sequence_length

        attention_mask = torch.randn(
            batch_size * num_heads, source_sequence_length, target_sequence_length
        )

        with self.assertRaises(RuntimeError) as context:
            m._AttentionValidator__check_attention_mask(attention_mask)

    def test__3D__correct_mask_shape__correct_input_dims(
        self,
    ):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        m = AttentionValidator(config)
        m.batched_input_flag = True

        batch_size = m.batch_size
        num_heads = m.num_heads
        source_sequence_length = m.source_sequence_length
        target_sequence_length = m.target_sequence_length

        attention_mask = torch.randn(
            batch_size * num_heads, source_sequence_length, target_sequence_length
        )

        output = m._AttentionValidator__check_attention_mask(attention_mask)
        self.assertIsNone(output)

    def test__4D_incorrect_input_dims(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        m = AttentionValidator(config)
        m.batched_input_flag = True

        batch_size = 2
        num_heads = 3
        source_sequence_length = 12
        target_sequence_length = m.target_sequence_length

        attention_mask = torch.randn(
            num_heads,
            batch_size,
            source_sequence_length,
            target_sequence_length,
        )

        with self.assertRaises(RuntimeError) as context:
            m._AttentionValidator__check_attention_mask(attention_mask)


class TestAttentionValidator____resolve_attention_mask_shape(TestAttention):
    def test__ensure_correct_shape_for_2D_mask(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        m = AttentionValidator(config)
        m.batched_input_flag = True

        source_sequence_length = m.source_sequence_length
        target_sequence_length = m.target_sequence_length

        expected_attention_mask_shape = (
            source_sequence_length,
            target_sequence_length,
        )

        attention_mask = torch.randn(
            source_sequence_length,
            target_sequence_length,
        )

        output = m._AttentionValidator__resolve_attention_mask_shape(attention_mask)
        self.assertEqual(output, expected_attention_mask_shape)

    def test__ensure_correct_shape_for_3D_mask(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        m = AttentionValidator(config)
        m.batched_input_flag = True

        batch_size = m.batch_size
        num_heads = m.num_heads
        source_sequence_length = m.source_sequence_length
        target_sequence_length = m.target_sequence_length

        expected_attention_mask_shape = (
            batch_size * num_heads,
            source_sequence_length,
            target_sequence_length,
        )

        attention_mask = torch.randn(
            batch_size * num_heads,
            source_sequence_length,
            target_sequence_length,
        )

        output = m._AttentionValidator__resolve_attention_mask_shape(attention_mask)
        self.assertEqual(output, expected_attention_mask_shape)


class TestAttentionValidator____ensure_attention_mask_if_causal(TestAttention):
    def test__None_as_input(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        m = AttentionValidator(config)
        m.causal_attention_mask_flag = True

        with self.assertRaises(RuntimeError) as context:
            m._AttentionValidator__ensure_attention_mask_if_causal(None)

    def test__causal_attention_mask_flag__True__and__None_as_input(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        m = AttentionValidator(config)
        m.causal_attention_mask_flag = True

        with self.assertRaises(RuntimeError) as context:
            m._AttentionValidator__ensure_attention_mask_if_causal(None)

    def test__causal_attention_mask_flag__False__and__attention_mask(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        m = AttentionValidator(config)
        m.causal_attention_mask_flag = False

        batch_size = m.batch_size
        num_heads = m.num_heads
        source_sequence_length = m.source_sequence_length
        target_sequence_length = m.target_sequence_length

        attention_mask = torch.randn(
            batch_size * num_heads,
            source_sequence_length,
            target_sequence_length,
        )

        output = m._AttentionValidator__ensure_attention_mask_if_causal(attention_mask)
        self.assertIsNone(output)


class TestAttentionValidator__multi_head_attention_input_shapes(TestAttention):
    def test__all_inputs_batched(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        m = AttentionValidator(config)

        batch_size = m.batch_size
        num_heads = m.num_heads
        source_sequence_length = m.source_sequence_length
        target_sequence_length = m.target_sequence_length
        embedding_dim = config.embedding_dim

        query = torch.randn(target_sequence_length, batch_size, embedding_dim)
        key = torch.randn(source_sequence_length, batch_size, embedding_dim)
        value = torch.randn(source_sequence_length, batch_size, embedding_dim)
        key_padding_mask = torch.randint(0, 1, (batch_size, source_sequence_length))
        attention_mask = torch.randn(
            batch_size * num_heads, source_sequence_length, target_sequence_length
        )

        output = m.multi_head_attention_input_shapes(
            query, key, value, key_padding_mask, attention_mask
        )

        self.assertTrue(output)

    def test__all_inputs_not_batched(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        m = AttentionValidator(config)

        source_sequence_length = m.source_sequence_length
        target_sequence_length = m.target_sequence_length
        embedding_dim = config.embedding_dim

        query = torch.randn(target_sequence_length, embedding_dim)
        key = torch.randn(source_sequence_length, embedding_dim)
        value = torch.randn(source_sequence_length, embedding_dim)
        key_padding_mask = torch.randint(0, 1, (source_sequence_length,))
        attention_mask = torch.randn(source_sequence_length, target_sequence_length)

        output = m.multi_head_attention_input_shapes(
            query, key, value, key_padding_mask, attention_mask
        )

        self.assertFalse(output)

    def test__no_key_and_attention_masks(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        m = AttentionValidator(config)

        batch_size = m.batch_size
        source_sequence_length = m.source_sequence_length
        target_sequence_length = m.target_sequence_length
        embedding_dim = config.embedding_dim

        query = torch.randn(target_sequence_length, batch_size, embedding_dim)
        key = torch.randn(source_sequence_length, batch_size, embedding_dim)
        value = torch.randn(source_sequence_length, batch_size, embedding_dim)

        output = m.multi_head_attention_input_shapes(query, key, value)

        self.assertTrue(output)


class TestAttentionValidator__is_mask_float_or_bool(TestAttention):
    def test__incorect_int_input(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        m = AttentionValidator(config)
        m.batched_input_flag = False

        mask = torch.randint(0, 10, (10, 10))
        maks_name = "test_mask"

        with self.assertRaises(RuntimeError) as context:
            m.is_mask_float_or_bool(mask, maks_name)

    def test__correct_float_input(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        m = AttentionValidator(config)
        m.batched_input_flag = False

        mask = torch.randn(10, 10)
        maks_name = "test_mask"

        output = m.is_mask_float_or_bool(mask, maks_name)
        self.assertIsNone(output)

    def test__correct_boolean_input(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        m = AttentionValidator(config)
        m.batched_input_flag = False

        mask = torch.randn(10, 10) > 0
        maks_name = "test_mask"

        output = m.is_mask_float_or_bool(mask, maks_name)
        self.assertIsNone(output)


class TestAttentionValidator__is_mask_correct_dtype(TestAttention):
    def test__incorrect__other_dtype__check_other__True(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        m = AttentionValidator(config)
        m.batched_input_flag = False

        mask = torch.randn(10, 10, dtype=torch.float64)
        maks_name = "test_mask"
        other_type = torch.float32
        other_name = "real_maks_dtype"
        check_other = True

        with self.assertRaises(RuntimeError) as context:
            m.is_mask_correct_dtype(
                mask, maks_name, other_type, other_name, check_other
            )

    def test__incorrect__other_dtype__check_other__False(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        m = AttentionValidator(config)
        m.batched_input_flag = False

        mask = torch.randn(10, 10, dtype=torch.float64)
        maks_name = "test_mask"
        other_type = torch.float32
        other_name = "real_maks_dtype"
        check_other = False

        output = m.is_mask_correct_dtype(
            mask, maks_name, other_type, other_name, check_other
        )
        self.assertIsNone(output)

    def test__mask__and__other_type__same_dtype(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        m = AttentionValidator(config)
        m.batched_input_flag = False

        mask = torch.randn(10, 10, dtype=torch.float32)
        maks_name = "test_mask"
        other_type = torch.float32
        other_name = "real_maks_dtype"
        check_other = True

        output = m.is_mask_correct_dtype(
            mask, maks_name, other_type, other_name, check_other
        )
        self.assertIsNone(output)

    def test__other_type__None__check_other__True(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        m = AttentionValidator(config)
        m.batched_input_flag = False

        mask = torch.randn(10, 10, dtype=torch.float32)
        maks_name = "test_mask"
        other_type = None
        other_name = "real_maks_dtype"
        check_other = True

        output = m.is_mask_correct_dtype(
            mask, maks_name, other_type, other_name, check_other
        )
        self.assertIsNone(output)


class TestAttentionValidator____canonical_mask(TestAttention):
    def test__input_as_None(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        validator = AttentionValidator(config)
        m = AttentionMask(config, validator)

        mask = None
        mask_name = ""
        other_type = None
        other_name = ""
        target_type = torch.float32
        check_other = False

        output = m._AttentionMask__canonical_mask(
            mask, mask_name, other_type, other_name, target_type, check_other
        )
        self.assertIsNone(output)

    def test__boolean_mask_input(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        validator = AttentionValidator(config)
        m = AttentionMask(config, validator)

        mask = torch.randn(10, 10, dtype=torch.float32) > 0
        mask_name = "maks_to_test"
        other_type = torch.bool
        other_name = "required_mask_dtype"
        target_type = torch.float32
        check_other = True

        output = m._AttentionMask__canonical_mask(
            mask, mask_name, other_type, other_name, target_type, check_other
        )

        self.assertTrue(output.dtype == torch.float32)
        self.assertFalse(output.dtype == mask.dtype)

    def test__float_mask_input(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        validator = AttentionValidator(config)
        m = AttentionMask(config, validator)

        mask = torch.randn(10, 10, dtype=torch.float32)
        mask_name = "maks_to_test"
        other_type = torch.float32
        other_name = "required_mask_dtype"
        target_type = torch.float32
        check_other = True

        output = m._AttentionMask__canonical_mask(
            mask, mask_name, other_type, other_name, target_type, check_other
        )

        self.assertTrue(torch.equal(output, mask))


class TestAttentionMask__validate_attention_mask(TestAttention):
    def test__key_padding_mask__None(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        validator = AttentionValidator(config)
        m = AttentionMask(config, validator)
        m.causal_attention_mask_flag = True

        batch_size = config.batch_size
        num_heads = config.num_heads
        source_sequence_length = config.source_sequence_length
        target_sequence_length = config.target_sequence_length

        key_padding_mask = None
        attention_mask = torch.randn(
            batch_size * num_heads, source_sequence_length, target_sequence_length
        )
        attention_mask = attention_mask > 0
        need_weights = False

        output = m._AttentionMask__validate_attention_mask(
            key_padding_mask,
            attention_mask,
            need_weights,
        )

        self.assertIsNone(output)

    def test__boolean_attention_mask(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        validator = AttentionValidator(config)
        m = AttentionMask(config, validator)
        m.causal_attention_mask_flag = True

        batch_size = config.batch_size
        num_heads = config.num_heads
        source_sequence_length = config.source_sequence_length
        target_sequence_length = config.target_sequence_length

        key_padding_mask = torch.randint(0, 1, (batch_size, source_sequence_length))
        attention_mask = torch.randn(
            batch_size * num_heads, source_sequence_length, target_sequence_length
        )
        attention_mask = attention_mask > 0
        need_weights = True

        output = m._AttentionMask__validate_attention_mask(
            key_padding_mask,
            attention_mask,
            need_weights,
        )

        self.assertIsInstance(output, torch.Tensor)
        self.assertNotEqual(output.dtype, torch.bool)
        self.assertEqual(output.dtype, config.target_dtype)
        self.assertFalse(m.causal_attention_mask_flag)

    def test__float_attention_mask(self):
        c = copy.deepcopy(self.cfg)
        config = c.multi_head_attention_model_config
        validator = AttentionValidator(config)
        m = AttentionMask(config, validator)
        m.causal_attention_mask_flag = True

        batch_size = config.batch_size
        num_heads = config.num_heads
        source_sequence_length = config.source_sequence_length
        target_sequence_length = config.target_sequence_length

        key_padding_mask = torch.randint(0, 1, (batch_size, source_sequence_length))
        attention_mask = torch.randn(
            batch_size * num_heads, source_sequence_length, target_sequence_length
        )
        need_weights = True

        output = m._AttentionMask__validate_attention_mask(
            key_padding_mask,
            attention_mask,
            need_weights,
        )

        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.dtype, attention_mask.dtype)
        self.assertTrue(torch.equal(output, attention_mask))
        self.assertFalse(m.causal_attention_mask_flag)

    # def test__validate_padding_and_attention_masks(self):
    #     c = copy.deepcopy(self.cfg)
    #     config = c.multi_head_attention_model_config
    #     validator = AttentionValidator(config)
    #     m = AttentionMask(config, validator)
    #     m.causal_attention_mask_flag = True
    #
    #     batch_size = config.batch_size
    #     num_heads = config.num_heads
    #     source_sequence_length = config.source_sequence_length
    #     target_sequence_length = config.target_sequence_length
    #
    #     key_padding_mask = torch.randint(0, 1, (batch_size, source_sequence_length)) > 0
    #     attention_mask = (
    #         torch.randn(
    #             batch_size * num_heads, source_sequence_length, target_sequence_length
    #         )
    #         > 0
    #     )
    #     need_weights = True
    #
    #     output = m.validate_padding_and_attention_masks(
    #         key_padding_mask,
    #         attention_mask,
    #         need_weights,
    #     )
    #
    #     self.assertIsInstance(output, torch.Tensor)
    #     self.assertEqual(output.dtype, attention_mask.dtype)
    #     self.assertTrue(torch.equal(output, attention_mask))
    #     self.assertFalse(m.causal_attention_mask_flag)

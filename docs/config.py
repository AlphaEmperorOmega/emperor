import torch
import torch.nn as nn

from Emperor.adaptive.utils.layers import AdaptiveParameterLayerConfig
from Emperor.adaptive.utils.mixtures.base import AdaptiveMixtureConfig
from Emperor.attention.utils.layer import MultiHeadAttentionConfig
from Emperor.base.enums import ActivationOptions, LayerNormPositionOptions
from Emperor.base.layer import LayerStackConfig
from Emperor.config import ModelConfig
from Emperor.experts.utils.layers import MixtureOfExpertsConfig
from Emperor.linears.options import LinearLayerOptions, LinearLayerStackOptions
from Emperor.linears.utils.layers import LinearLayerConfig
from Emperor.sampler.utils.routers import RouterConfig
from Emperor.sampler.utils.samplers import SamplerConfig
from Emperor.transformer.stack import TransformerConfig
from Emperor.transformer.utils.feed_forward import (
    FeedForwardConfig,
    MixtureOfExpertsFeedForwardConfig,
)
from Emperor.neuron.neuron import (
    AxonsConfig,
    NeuronClusterConfig,
    NucleusConfig,
    TerminalConfig,
    TerminalRangeOptions,
    TerminalZAxisOffsetOptions,
)


def default_unittest_config():
    # MODEL WISE CONFI
    BATCH_SIZE = 2
    INPUT_DIM = 4
    HIDDEN_DIM = 12
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
    MIXTURE_OUTPUT_DIM = OUTPUT_DIM
    MIXTURE_DEPTH_DIM = ROUTER_OUTPUT_DIM
    MIXTURE_TOP_K = SAMPLER_TOP_K
    MIXTURE_WEIGHTED_PARAMETERS_FLAG = False
    MIXTURE_BIAS_PARAMETERS_FLAG = False
    MIXTURE_ROUTER_OUTPUT_DIM = ROUTER_OUTPUT_DIM
    MIXTURE_CROSS_DIAGONAL_FLAG = False

    # PARAMETER GENERATOR OPTIONS
    PARAMETER_GENERATOR_BIAS_PARAMETER_FLAG = MIXTURE_WEIGHTED_PARAMETERS_FLAG

    SOURCE_SEQUENCE_LENGTH = 16
    TARGET_SEQUENCE_LENGTH = 32

    return ModelConfig(
        batch_size=BATCH_SIZE,
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
        gather_frequency_flag=GATHER_FREQUENCY_FLAG,
        router_model_config=RouterConfig(
            input_dim=ROUTER_INPUT_DIM,
            num_experts=ROUTER_OUTPUT_DIM,
            noisy_topk_flag=ROUTER_NOISY_TOPK_FLAG,
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
        mixture_model_config=AdaptiveMixtureConfig(
            input_dim=MIXTURE_INPUT_DIM,
            output_dim=MIXTURE_OUTPUT_DIM,
            top_k=MIXTURE_TOP_K,
            weighted_parameters_flag=MIXTURE_WEIGHTED_PARAMETERS_FLAG,
            num_experts=MIXTURE_ROUTER_OUTPUT_DIM,
        ),
        parameter_generator_model_config=AdaptiveParameterLayerConfig(
            time_tracker_flag=False,
        ),
        linear_layer_config=LinearLayerConfig(
            input_dim=INPUT_DIM,
            output_dim=OUTPUT_DIM,
            bias_flag=True,
        ),
        mixture_of_experts_config=MixtureOfExpertsFeedForwardConfig(
            weighted_parameters_flag=True,
        ),
        input_moe_layer_config=MixtureOfExpertsConfig(
            input_dim=ROUTER_INPUT_DIM,
            output_dim=64,
            top_k=MIXTURE_TOP_K,
            num_experts=SAMPLER_ROUTER_OUTPUT_DIM,
            compute_expert_mixture_flag=False,
            weighted_parameters_flag=False,
            init_sampler_model_flag=False,
        ),
        output_moe_layer_config=MixtureOfExpertsConfig(
            input_dim=64,
            output_dim=ROUTER_INPUT_DIM,
            top_k=MIXTURE_TOP_K,
            num_experts=SAMPLER_ROUTER_OUTPUT_DIM,
            compute_expert_mixture_flag=True,
            weighted_parameters_flag=True,
            init_sampler_model_flag=False,
        ),
        multi_head_attention_model_config=MultiHeadAttentionConfig(
            model_type=LinearLayerStackOptions.ADAPTIVE,
            batch_size=BATCH_SIZE,
            num_heads=4,
            embedding_dim=HIDDEN_DIM,
            target_sequence_length=16,
            source_sequence_length=16,
            target_dtype=torch.float32,
            is_self_attention_projector_flag=False,
            dropout_probability=0.0,
            key_value_bias_flag=False,
            zero_attention_flag=False,
            query_key_projection_dim=0,
            value_projection_dim=0,
            causal_attention_mask_flag=False,
            add_key_value_bias_flag=False,
            average_attention_weights_flag=False,
            return_attention_weights_flag=False,
        ),
        transformer_feed_forward_config=FeedForwardConfig(
            num_layers=2,
            layer_stack_option=LinearLayerOptions.ADAPTIVE,
        ),
        layer_stack_config=LayerStackConfig(
            input_dim=HIDDEN_DIM,
            hidden_dim=HIDDEN_DIM,
            output_dim=HIDDEN_DIM,
            num_layers=2,
            model_type=LinearLayerOptions.ADAPTIVE,
            activation=ActivationOptions.GELU,
            layer_norm_dim=HIDDEN_DIM,
            residual_flag=False,
            adaptive_computation_flag=False,
            dropout_probability=0.0,
            layer_norm_position=LayerNormPositionOptions.NONE,
        ),
        transformer_layer_config=TransformerConfig(),
        transformer_config=TransformerConfig(
            num_layers=6,
            source_sequence_length=16,
            target_sequence_length=16,
            layer_norm_dim=HIDDEN_DIM,
            causal_attention_mask_flag=False,
        ),
        neuron_nucleus_config=NucleusConfig(
            model_type=LinearLayerOptions.ADAPTIVE,
        ),
        neuron_axon_config=AxonsConfig(
            memory_type=None,
        ),
        neuron_terminal_config=TerminalConfig(
            x_axis_position=0,
            y_axis_position=0,
            z_axis_position=0,
            xy_axis_range=TerminalRangeOptions.TWO,
            z_axis_range=TerminalRangeOptions.THREE,
            z_axis_offset=TerminalZAxisOffsetOptions.ONE,
        ),
        neuron_cluster_config=NeuronClusterConfig(
            x_axis_total_neurons=10,
            y_axis_total_neurons=10,
            z_axis_total_neurons=10,
        ),
    )

import torch.nn as nn
from torch import float32
from dataclasses import dataclass, field

from emperor.attention.utils.layer import MultiHeadAttentionConfig
from emperor.base.enums import ActivationOptions, LayerNormPositionOptions
from emperor.base.utils import ConfigBase
from emperor.experts.utils.enums import InitSamplerOptions
from emperor.transformer.utils.layers import TransformerConfig
from emperor.transformer.utils.feed_forward import FeedForwardConfig
from emperor.experts.utils.layers import MixtureOfExpertsConfig
from emperor.parametric.utils.config import ParametricLayerConfig, AdaptiveRouterOptions
from emperor.base.layer import LayerStackConfig
from emperor.parametric.utils.mixtures.base import AdaptiveMixtureConfig
from emperor.linears.options import LinearLayerOptions, LinearLayerStackOptions
from emperor.linears.utils.config import LinearLayerConfig
from emperor.sampler.utils.samplers import SamplerConfig
from emperor.sampler.utils.routers import RouterConfig
from emperor.neuron.neuron import (
    AxonsConfig,
    NeuronClusterConfig,
    NucleusConfig,
    TerminalConfig,
    TerminalRangeOptions,
)

# MODEL WISE CONFI
BATCH_SIZE = 10
SEQUENCE_LENGTH = 5
LEARNING_RATE = 1e-3
INPUT_DIM = 16
HIDDEN_DIM = 32
OUTPUT_DIM = 64
DEPTH_DIM = 5
NUM_EXPERTS = 32
THRESHOLD = 0.01
NOISY_TOPK_FLAG = False
BIAS_PARAMETER_FLAG = True
TOP_K = 3
GATHER_FREQUENCY_FLAG = False

# PARAMETER GENRETOR ROUTER OPITONS
ROUTER_INPUT_DIM = HIDDEN_DIM
ROUTER_HIDDEN_DIM = HIDDEN_DIM
ROUTER_OUTPUT_DIM = NUM_EXPERTS
ROUTER_RESIDUAL_FLAG = True
ROUTER_NOISY_TOPK_FLAG = NOISY_TOPK_FLAG
ROUTER_ACTIVATION = nn.ReLU()
ROUTER_NUM_LAYERS = 5
ROUTER_DIAGONAL_LINEAR_MODEL_FLAG = True

# PARAMETER GENRETOR SAMPLER OPITONS
SAMPLER_TOP_K = TOP_K
SAMPLER_THRESHOLD = THRESHOLD
SAMPLER_NUM_TOPK_SAMPLES = 0
SAMPLER_FILTER_THRESHOLD = False
SAMPLER_NORMALIZE_PROBABILITIES_FLAG = False
SAMPLER_NOISY_TOPK_FLAG = NOISY_TOPK_FLAG
SAMPLER_ROUTER_OUTPUT_DIM = NUM_EXPERTS
SAMPLER_COEFFICIENT_OF_VARIATION_WEIGHT = 0.0
SAMPLER_SWITCH_WEIGHT = 0.0
SAMPLER_ZERO_CENTRED_WEIGHT = 0.0
SAMPLER_MUTUAL_INFORMATION_WEIGHT = 0.0

# PARAMETER GENRETOR MIXTURE OPITONS
MIXTURE_INPUT_DIM = INPUT_DIM
MIXTURE_OUTPUT_DIM = OUTPUT_DIM
MIXTURE_DEPTH_DIM = NUM_EXPERTS
MIXTURE_TOP_K = TOP_K
MIXTURE_WEIGHTED_PARAMETERS_FLAG = False
MIXTURE_BIAS_PARAMETERS_FLAG = BIAS_PARAMETER_FLAG
MIXTURE_NUM_EXPERTS = NUM_EXPERTS
MIXTURE_ANTI_DIAGONAL_FLAG = True

# PARAMETER GENERATOR OPTIONS
PARAMETER_GENERATOR_BIAS_PARAMETER_FLAG = BIAS_PARAMETER_FLAG
PARAMETER_GENERATOR_TRACK_TIME_FLAG = False
PARAMETER_GENERATOR_DYNAMIC_DIAGONAL_PARAMS_FLAG = True


@dataclass
class ModelConfig(ConfigBase):
    batch_size: int = field(
        default=BATCH_SIZE,
        metadata={"help": "Batch size for training and inference"},
    )
    learning_rate: float = field(
        default=LEARNING_RATE,
        metadata={"help": "Batch size for training and inference"},
    )
    sequence_length: int = field(
        default=SEQUENCE_LENGTH,
        metadata={"help": "Number of tokens for each sequence in the input batch."},
    )
    input_dim: int = field(
        default=INPUT_DIM,
        metadata={"help": "Dimension of the input features"},
    )
    hidden_dim: int = field(
        default=HIDDEN_DIM,
        metadata={"help": "Dimension of the output features"},
    )
    output_dim: int = field(
        default=OUTPUT_DIM,
        metadata={"help": "Dimension of the output features"},
    )
    gather_frequency_flag: bool = field(
        default=GATHER_FREQUENCY_FLAG,
        metadata={
            "help": "Flag to control frequency of gathering operations for the purpose of visualization"
        },
    )
    router_model_config: RouterConfig = field(
        default_factory=lambda: RouterConfig(
            num_experts=ROUTER_OUTPUT_DIM,
            noisy_topk_flag=ROUTER_NOISY_TOPK_FLAG,
        ),
        metadata={"help": "`RouterModel` configuration"},
    )
    sampler_model_config: SamplerConfig = field(
        default_factory=lambda: SamplerConfig(
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
        metadata={"help": "`SamplerConfig` configuration"},
    )
    mixture_model_config: AdaptiveMixtureConfig = field(
        default_factory=lambda: AdaptiveMixtureConfig(
            input_dim=MIXTURE_INPUT_DIM,
            output_dim=MIXTURE_OUTPUT_DIM,
            top_k=MIXTURE_TOP_K,
            weighted_parameters_flag=MIXTURE_WEIGHTED_PARAMETERS_FLAG,
            num_experts=MIXTURE_NUM_EXPERTS,
        ),
        metadata={"help": "`MixtureConfig` configuration"},
    )
    parameter_generator_model_config: ParametricLayerConfig = field(
        default_factory=lambda: ParametricLayerConfig(
            time_tracker_flag=PARAMETER_GENERATOR_TRACK_TIME_FLAG,
            # dynamic_diagonal_params_flag=PARAMETER_GENERATOR_DYNAMIC_DIAGONAL_PARAMS_FLAG,
        ),
        metadata={"help": "`ParameterGeneratorConfig` configuration"},
    )
    linear_layer_config: LinearLayerConfig | LinearLayerConfig = field(
        default_factory=lambda: LinearLayerConfig(
            input_dim=INPUT_DIM,
            output_dim=OUTPUT_DIM,
            bias_flag=True,
        ),
        metadata={"help": "`LinearLayerConfig` configuration"},
    )
    mixture_of_experts_config: MixtureOfExpertsConfig = field(
        default_factory=lambda: MixtureOfExpertsConfig(
            weighted_parameters_flag=True,
        ),
        metadata={"help": "`MixtureOfExpertsConfig` configuration"},
    )
    input_moe_layer_config: MixtureOfExpertsConfig = field(
        default_factory=lambda: MixtureOfExpertsConfig(
            top_k=MIXTURE_TOP_K,
            num_experts=12,
            compute_expert_mixture_flag=False,
            weighted_parameters_flag=False,
            init_sampler_option=InitSamplerOptions.SHARED,
        ),
        metadata={"help": "`MixtureOfExpertsConfig` configuration"},
    )
    output_moe_layer_config: MixtureOfExpertsConfig = field(
        default_factory=lambda: MixtureOfExpertsConfig(
            layer_stack_option=LinearLayerStackOptions.BASE,
            top_k=MIXTURE_TOP_K,
            num_experts=12,
            compute_expert_mixture_flag=True,
            weighted_parameters_flag=True,
            init_sampler_option=InitSamplerOptions.SHARED,
        ),
        metadata={"help": "`MixtureOfExpertsConfig` configuration"},
    )
    multi_head_attention_model_config: MultiHeadAttentionConfig = field(
        default_factory=lambda: MultiHeadAttentionConfig(
            model_type=LinearLayerOptions.ADAPTIVE,
            batch_size=BATCH_SIZE,
            num_heads=NUM_EXPERTS,
            query_key_projection_dim=16,
            value_projection_dim=32,
            embedding_dim=64,
            target_sequence_length=16,
            source_sequence_length=32,
            target_dtype=float32,
            attention_option=False,
            dropout_probability=0.0,
            key_value_bias_flag=False,
            zero_attention_flag=False,
            causal_attention_mask_flag=False,
            add_key_value_bias_flag=False,
        ),
        metadata={"help": "`MultiHeadAttention` configuration"},
    )
    layer_stack_config: LayerStackConfig = field(
        default_factory=lambda: LayerStackConfig(
            input_dim=INPUT_DIM,
            hidden_dim=HIDDEN_DIM,
            output_dim=OUTPUT_DIM,
            num_layers=2,
            activation=ActivationOptions.RELU,
            model_type=nn.Linear,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            residual_flag=False,
            adaptive_computation_flag=False,
        ),
        metadata={"help": "`MultiHeadAttention` configuration"},
    )
    transformer_feed_forward_config: FeedForwardConfig = field(
        default_factory=lambda: FeedForwardConfig(
            layer_stack_option=LinearLayerOptions.ADAPTIVE,
            num_layers=1,
        ),
        metadata={"help": "`MultiHeadAttention` configuration"},
    )
    transformer_layer_config: TransformerConfig = field(
        default_factory=lambda: TransformerConfig(),
        metadata={"help": "`MultiHeadAttention` configuration"},
    )
    transformer_config: TransformerConfig | None = field(
        default_factory=lambda: TransformerConfig(
            num_layers=2,
            source_sequence_length=0,
            target_sequence_length=0,
            causal_attention_mask_flag=False,
        ),
        metadata={
            "help": "`TransformerEncoder` and `TransformerDecoder` configuration"
        },
    )
    neuron_nucleus_config: NucleusConfig | None = field(
        default_factory=lambda: NucleusConfig(
            model_type=LinearLayerOptions.ADAPTIVE,
        ),
        metadata={"help": "`Nucleus` configuration"},
    )
    neuron_axon_config: AxonsConfig | None = field(
        default_factory=lambda: AxonsConfig(
            memory_type=None,
        ),
        metadata={"help": "Neuron `Axon` configuration"},
    )
    neuron_terminal_config: TerminalConfig | None = field(
        default_factory=lambda: TerminalConfig(
            x_axis_position=0,
            y_axis_position=0,
            z_axis_position=0,
            xy_axis_range=TerminalRangeOptions.TWO,
            z_axis_range=TerminalRangeOptions.TWO,
            z_axis_offset=TerminalRangeOptions.ONE,
        ),
        metadata={"help": "Neuron `Axon` configuration"},
    )
    neuron_cluster_config: NeuronClusterConfig | None = field(
        default_factory=lambda: NeuronClusterConfig(
            x_axis_total_neurons=10,
            y_axis_total_neurons=10,
            z_axis_total_neurons=10,
        ),
        metadata={"help": "Neuron `Cluster` configuration"},
    )

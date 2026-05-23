import torch.nn as nn
from dataclasses import dataclass, field

from emperor.base.utils import ConfigBase, optional_field
from emperor.transformer.utils.layers import TransformerConfig
from emperor.transformer.utils.feed_forward import FeedForwardConfig
from emperor.experts.core.config import MixtureOfExpertsConfig
from emperor.base.layer import LayerStackConfig
from emperor.sampler.core.samplers import SamplerConfig
from emperor.sampler.core.routers import RouterConfig
from emperor.neuron.neuron import (
    AxonsConfig,
    NeuronClusterConfig,
    NucleusConfig,
    TerminalConfig,
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
    experiment_config: ConfigBase | None = field(
        default=None,
        metadata={"help": "Config used to build the model module within the layer"},
    )

    gather_frequency_flag: bool = field(
        default=GATHER_FREQUENCY_FLAG,
        metadata={
            "help": "Flag to control frequency of gathering operations for the purpose of visualization"
        },
    )
    router_model_config: "RouterConfig | None" = optional_field(
        "`RouterModel` configuration"
    )
    sampler_model_config: "SamplerConfig | None" = optional_field(
        "`SamplerConfig` configuration"
    )
    mixture_model_config: "AdaptiveMixtureConfig | None" = field(
        default=None,
        metadata={"help": "`MixtureConfig` configuration"},
    )
    parameter_generator_model_config: "ParametricLayerConfig | None" = field(
        default=None,
        metadata={"help": "`ParameterGeneratorConfig` configuration"},
    )
    input_moe_layer_config: "MixtureOfExpertsConfig | None" = optional_field(
        "`MixtureOfExpertsConfig` configuration"
    )
    output_moe_layer_config: "MixtureOfExpertsConfig | None" = optional_field(
        "`MixtureOfExpertsConfig` configuration"
    )
    multi_head_attention_model_config: "MultiHeadAttentionConfig | None" = field(
        default=None,
        metadata={"help": "`MultiHeadAttention` configuration"},
    )
    layer_stack_config: "LayerStackConfig | None" = optional_field(
        "`LayerStack` configuration"
    )
    transformer_feed_forward_config: "FeedForwardConfig | None" = optional_field(
        "`FeedForward` configuration"
    )
    transformer_layer_config: "TransformerConfig | None" = optional_field(
        "`Transformer` layer configuration"
    )
    transformer_config: "TransformerConfig | None" = optional_field(
        "`TransformerEncoder` and `TransformerDecoder` configuration"
    )
    neuron_nucleus_config: "NucleusConfig | None" = optional_field(
        "`Nucleus` configuration"
    )
    neuron_axon_config: "AxonsConfig | None" = optional_field(
        "Neuron `Axon` configuration"
    )
    neuron_terminal_config: "TerminalConfig | None" = optional_field(
        "Neuron `Terminal` configuration"
    )
    neuron_cluster_config: "NeuronClusterConfig | None" = optional_field(
        "Neuron `Cluster` configuration"
    )

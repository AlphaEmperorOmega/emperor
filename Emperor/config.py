import torch.nn as nn
from dataclasses import dataclass, field
from Emperor.base.utils import DataClassBase


from Emperor.components.parameter_generators.layers import ParameterLayerConfig
from Emperor.components.parameter_generators.utils.linears import LinearLayerConfig
from Emperor.components.parameter_generators.utils.mixture import MixtureConfig
from Emperor.components.parameter_generators.utils.samplers import SamplerConfig
from Emperor.components.parameter_generators.utils.routers import (
    RouterConfig,
)


# MODEL WISE CONFI
BATCH_SIZE = 10
INPUT_DIM = 16
HIDDEN_DIM = 32
OUTPUT_DIM = 64
GATHER_FREQUENCY_FLAG = False

# PARAMETER GENRETOR ROUTER OPITONS
ROUTER_INPUT_DIM = HIDDEN_DIM
ROUTER_HIDDEN_DIM = 32
ROUTER_OUTPUT_DIM = 128
ROUTER_NOISY_TOPK_FLAG = False
ROUTER_ACTIVATION_FUNCTION = nn.ReLU()
ROUTER_NUM_LAYERNUM_LAYERSS = 3
ROUTER_DIAGONAL_LINEAR_MODEL_FLAG = True

# PARAMETER GENRETOR SAMPLER OPITONS
SAMPLER_TOP_K = 8
SAMPLER_THRESHOLD = 0.1
SAMPLER_DYNAMIC_TOPK_THRESHOLD = (1 / ROUTER_OUTPUT_DIM) * 1e-6
SAMPLER_NUM_TOPK_SAMPLES = 0
SAMPLER_NORMALIZE_PROBABILITIES_FLAG = False
SAMPLER_NOISY_TOPK_FLAG = ROUTER_NOISY_TOPK_FLAG
SAMPLER_ROUTER_OUTPUT_DIM = ROUTER_OUTPUT_DIM
SAMPLER_BOOLEAN_MASK_FLAG = False
SAMPLER_COEFFICIENT_OF_VARIATION_WEIGHT = 0.0
SAMPLER_SWITCH_WEIGHT = 0.0
SAMPLER_ZERO_CENTRED_WEIGHT = 0.0
SAMPLER_MUTUAL_INFORMATION_WEIGHT = 0.0

# PARAMETER GENRETOR MIXTURE OPITONS
MIXTURE_INPUT_DIM = 16
MIXTURE_OUTPUT_DIM = 64
MIXTURE_DEPTH_DIM = ROUTER_OUTPUT_DIM
MIXTURE_TOP_K = SAMPLER_TOP_K
MIXTURE_WEIGHTED_PARAMETERS_FLAG = False
MIXTURE_BIAS_PARAMETERS_FLAG = False
MIXTURE_ROUTER_OUTPUT_DIM = ROUTER_OUTPUT_DIM
MIXTURE_CROSS_DIAGONAL_FLAG = False

# PARAMETER GENERATOR OPTIONS
PARAMETER_GENERATOR_BIAS_PARAMETER_FLAG = MIXTURE_WEIGHTED_PARAMETERS_FLAG
PARAMETER_GENERATOR_TRACK_TIME_FLAG = MIXTURE_WEIGHTED_PARAMETERS_FLAG


@dataclass
class ModelConfig(DataClassBase):
    batch_size: int = field(
        default=BATCH_SIZE,
        metadata={"help": "Batch size for training and inference"},
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
            input_dim=ROUTER_INPUT_DIM,
            hidden_dim=ROUTER_HIDDEN_DIM,
            output_dim=ROUTER_OUTPUT_DIM,
            noisy_topk_flag=ROUTER_NOISY_TOPK_FLAG,
            activation=ROUTER_ACTIVATION_FUNCTION,
            num_layers=ROUTER_NUM_LAYERNUM_LAYERSS,
            diagonal_linear_model_flag=ROUTER_DIAGONAL_LINEAR_MODEL_FLAG,
        ),
        metadata={"help": "`RouterModel` configuration"},
    )
    sampler_model_config: SamplerConfig = field(
        default_factory=lambda: SamplerConfig(
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
        metadata={"help": "`SamplerConfig` configuration"},
    )
    mixture_model_config: MixtureConfig = field(
        default_factory=lambda: MixtureConfig(
            input_dim=MIXTURE_INPUT_DIM,
            output_dim=MIXTURE_OUTPUT_DIM,
            depth_dim=MIXTURE_DEPTH_DIM,
            top_k=MIXTURE_TOP_K,
            bias_parameters_flag=MIXTURE_BIAS_PARAMETERS_FLAG,
            weighted_parameters_flag=MIXTURE_WEIGHTED_PARAMETERS_FLAG,
            router_output_dim=MIXTURE_ROUTER_OUTPUT_DIM,
            cross_diagonal_flag=MIXTURE_CROSS_DIAGONAL_FLAG,
        ),
        metadata={"help": "`MixtureConfig` configuration"},
    )
    parameter_generator_model_config: ParameterLayerConfig = field(
        default_factory=lambda: ParameterLayerConfig(
            bias_parameters_flag=PARAMETER_GENERATOR_BIAS_PARAMETER_FLAG,
        ),
        metadata={"help": "`ParameterGeneratorConfig` configuration"},
    )
    linear_layer_model_config: LinearLayerConfig = field(
        default_factory=lambda: LinearLayerConfig(
            input_dim=INPUT_DIM,
            output_dim=OUTPUT_DIM,
            bias_flag=True,
            anti_diagonal_flag=True,
        ),
        metadata={"help": "`LinearLayerConfig` configuration"},
    )

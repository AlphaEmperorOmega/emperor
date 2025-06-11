import torch.nn as nn
from dataclasses import dataclass, field
from Emperor.base.utils import DataClassBase


from Emperor.components.parameter_generators.layers import ParameterLayerConfig
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

# AUXILIARY LOSSES OPITONS
COEFFICIENT_OF_VARIATION_LOSS_WEIGHT: float = 0.0
SWITCH_LOSS_WEIGHT: float = 0.0
ZERO_CENTERED_LOSS_WEIGHT: float = 0.0
MUTUAL_INFORMATION_LOSS_WEIGHT: float = 0.0

# PARAMETER GENRETOR ROUTER OPITONS
ROUTER_INPUT_DIM = HIDDEN_DIM
ROUTER_HIDDEN_DIM = 32
ROUTER_OUTPUT_DIM = 128
ROUTER_NOISY_TOPK_FLAG = False
ROUTER_ACTIVATION_FUNCTION = nn.ReLU()
ROUTER_NUM_LAYERNUM_LAYERSS = 3

# PARAMETER GENRETOR SAMPLER OPITONS
SAMPLER_TOP_K = 8
SAMPLER_THRESHOLD = 0.1
SAMPLER_DYNAMIC_TOPK_THRESHOLD = (1 / ROUTER_OUTPUT_DIM) * 1e-6
SAMPLER_NUM_TOPK_SAMPLES = 0
SAMPLER_NORMALIZE_PROBABILITIES_FLAG = False
SAMPLER_NOISY_TOPK_FLAG = ROUTER_NOISY_TOPK_FLAG
SAMPLER_ROUTER_OUTPUT_DIM = ROUTER_OUTPUT_DIM
SAMPLER_BOOLEAN_MASK_FLAG = False

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
    coefficient_of_variation_loss_weight: float = field(
        default=COEFFICIENT_OF_VARIATION_LOSS_WEIGHT
    )
    switch_loss_weight: float = field(
        default=SWITCH_LOSS_WEIGHT,
        metadata={"help": ""},
    )
    zero_centered_loss_weight: float = field(
        default=ZERO_CENTERED_LOSS_WEIGHT,
        metadata={"help": ""},
    )
    mutual_information_loss_weight: float = field(
        default=MUTUAL_INFORMATION_LOSS_WEIGHT,
        metadata={"help": ""},
    )
    router_model_config: RouterConfig = field(
        default_factory=lambda: RouterConfig(
            input_dim=ROUTER_INPUT_DIM,
            hidden_dim=ROUTER_HIDDEN_DIM,
            output_dim=ROUTER_OUTPUT_DIM,
            noisy_topk_flag=ROUTER_NOISY_TOPK_FLAG,
            activation=ROUTER_ACTIVATION_FUNCTION,
            num_layers=ROUTER_NUM_LAYERNUM_LAYERSS,
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


# Flags
# MULTIPLY_BY_GATES: bool = False
# ATTENTION_PROJECTION_BIAS_FLAG: bool = False
# ADD_ZERO_ATTENTION_FLAG: bool = False
# SELF_ATTENTION_FLAG: bool = False
# ENCODER_DECODER_ATTENTION_FLAG: bool = False
#
# # Transformer Attention Inputs
# EMBEDDING_DIM: int = 784
# QUERY_INPUT_DIM: Optional[int] = None
# KEY_INPUT_DIM: Optional[int] = None
# VALUE_INPUT_DIM: Optional[int] = None
# QKV_HIDDEN_DIM: Optional[int] = 128
# HEAD_DIM: int = 64
# ATTENTION_OUTPUT_DIM: Optional[int] = 10
#
# # Transformer Config
# QKV_INPUT_DIM: Optional[int] = None
# QKV_OUTPUT_DIM: Optional[int] = None
# FFN_INPUT_DIM: Optional[int] = None
# FFN_HIDDEN_DIM: Optional[int] = None
# FFN_OUTPUT_DIM: Optional[int] = None
#
# # Additional Flags
# RETURN_RAW_FFN_OUTPUT_FLAG: bool = False
# NORMALIZE_BEFORE_FLAG: bool = False
# ADD_MEMORY_BIAS_KEY_VALUES_FLAG: bool = False
#
# # Dropout Probabilities
# ATTN_DROPOUT_PROBABILITY: float = 0.0
# FFN_DROPOUT_PROBABILITY: float = 0.0
# DROPOUT_PROBABILITY: float = 0.0
# QUANT_NOISE: float = 0.0
# QUANT_BLOCK_SIZE: int = 0
#
# # Gating
# GATING_DROPOUT: int = 0

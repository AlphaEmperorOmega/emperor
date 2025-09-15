from torch import float32
import torch.nn as nn
from dataclasses import dataclass, field

from Emperor.attention.attention import MultiHeadAttentionConfig
from Emperor.base.enums import ActivationOptions, LayerNormPositionOptions
from Emperor.base.utils import DataClassBase
from Emperor.feedForward.feed_forward import (
    FeedForwardConfig,
    MixtureOfExpertsFeedForwardConfig,
)
from Emperor.experts.experts import MixtureOfExpertsConfig
from Emperor.layers.layers import ParameterLayerConfig
from Emperor.layers.utils.base import LayerBlockStackConfig
from Emperor.layers.utils.enums import (
    # AttentionTypes,
    # FeedForwardTypes,
    LayerTypes,
    LinearLayerTypes,
)
from Emperor.layers.utils.linears import LinearLayerConfig
from Emperor.layers.utils.mixture import MixtureConfig
from Emperor.layers.utils.samplers import SamplerConfig
from Emperor.layers.utils.routers import (
    RouterConfig,
)
from Emperor.transformer.layer import TransformerLayerConfig


# MODEL WISE CONFI
BATCH_SIZE = 10
SEQUENCE_LENGTH = 5
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
class ModelConfig(DataClassBase):
    batch_size: int = field(
        default=BATCH_SIZE,
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
            input_dim=ROUTER_INPUT_DIM,
            hidden_dim=ROUTER_HIDDEN_DIM,
            output_dim=ROUTER_OUTPUT_DIM,
            residual_flag=ROUTER_RESIDUAL_FLAG,
            noisy_topk_flag=ROUTER_NOISY_TOPK_FLAG,
            activation=ROUTER_ACTIVATION,
            num_layers=ROUTER_NUM_LAYERS,
            diagonal_model_type_flag=ROUTER_DIAGONAL_LINEAR_MODEL_FLAG,
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
    mixture_model_config: MixtureConfig = field(
        default_factory=lambda: MixtureConfig(
            input_dim=MIXTURE_INPUT_DIM,
            output_dim=MIXTURE_OUTPUT_DIM,
            depth_dim=MIXTURE_DEPTH_DIM,
            top_k=MIXTURE_TOP_K,
            weighted_parameters_flag=MIXTURE_WEIGHTED_PARAMETERS_FLAG,
            bias_parameters_flag=MIXTURE_BIAS_PARAMETERS_FLAG,
            num_experts=MIXTURE_NUM_EXPERTS,
            dynamic_diagonal_params_flag=MIXTURE_ANTI_DIAGONAL_FLAG,
        ),
        metadata={"help": "`MixtureConfig` configuration"},
    )
    parameter_generator_model_config: ParameterLayerConfig = field(
        default_factory=lambda: ParameterLayerConfig(
            bias_parameters_flag=PARAMETER_GENERATOR_BIAS_PARAMETER_FLAG,
            time_tracker_flag=PARAMETER_GENERATOR_TRACK_TIME_FLAG,
            dynamic_diagonal_params_flag=PARAMETER_GENERATOR_DYNAMIC_DIAGONAL_PARAMS_FLAG,
        ),
        metadata={"help": "`ParameterGeneratorConfig` configuration"},
    )
    linear_layer_model_config: LinearLayerConfig = field(
        default_factory=lambda: LinearLayerConfig(
            input_dim=INPUT_DIM,
            output_dim=OUTPUT_DIM,
            bias_flag=True,
            anti_diagonal_flag=True,
            dynamic_bias_flag=True,
        ),
        metadata={"help": "`LinearLayerConfig` configuration"},
    )
    mixture_of_experts_config: MixtureOfExpertsFeedForwardConfig = field(
        default_factory=lambda: MixtureOfExpertsFeedForwardConfig(
            weighted_parameters_flag=True,
        ),
        metadata={"help": "`MixtureOfExpertsConfig` configuration"},
    )
    input_moe_layer_config: MixtureOfExpertsConfig = field(
        default_factory=lambda: MixtureOfExpertsConfig(
            input_dim=32,
            output_dim=64,
            top_k=MIXTURE_TOP_K,
            dropout_probability=0.1,
            layer_norm_flag=True,
            activation=ActivationOptions.GELU,
            model_type=LayerTypes.DYNAMIC_BASE,
            num_experts=12,
            compute_expert_mixture_flag=False,
            weighted_parameters_flag=False,
            init_sampler_model_flag=False,
        ),
        metadata={"help": "`MixtureOfExpertsConfig` configuration"},
    )
    output_moe_layer_config: MixtureOfExpertsConfig = field(
        default_factory=lambda: MixtureOfExpertsConfig(
            input_dim=32,
            output_dim=64,
            top_k=MIXTURE_TOP_K,
            dropout_probability=0.1,
            layer_norm_flag=True,
            activation=ActivationOptions.GELU,
            model_type=LayerTypes.DYNAMIC_BASE,
            num_experts=12,
            compute_expert_mixture_flag=True,
            weighted_parameters_flag=True,
            init_sampler_model_flag=False,
        ),
        metadata={"help": "`MixtureOfExpertsConfig` configuration"},
    )
    multi_head_attention_model_config: MultiHeadAttentionConfig = field(
        default_factory=lambda: MultiHeadAttentionConfig(
            model_type=LayerTypes.DYNAMIC_BASE,
            batch_size=BATCH_SIZE,
            num_heads=NUM_EXPERTS,
            query_key_projection_dim=16,
            value_projection_dim=32,
            embedding_dim=64,
            target_sequence_length=16,
            source_sequence_length=32,
            target_dtype=float32,
            use_separate_projection_weight_flag=False,
            dropout_probability=0.0,
            key_value_bias_flag=False,
            zero_attention_flag=False,
            causal_attention_mask_flag=False,
            add_key_value_bias_flag=False,
        ),
        metadata={"help": "`MultiHeadAttention` configuration"},
    )
    layer_block_stack_config: LayerBlockStackConfig = field(
        default_factory=lambda: LayerBlockStackConfig(
            input_dim=INPUT_DIM,
            hidden_dim=HIDDEN_DIM,
            output_dim=OUTPUT_DIM,
            num_layers=2,
            activation=ActivationOptions.RELU,
            model_type=nn.Linear,
            layer_norm_position=LayerNormPositionOptions.NONE,
            residual_flag=False,
            adaptive_computation_flag=False,
        ),
        metadata={"help": "`MultiHeadAttention` configuration"},
    )
    transformer_feed_forward_config: FeedForwardConfig = field(
        default_factory=lambda: FeedForwardConfig(
            model_type=LayerTypes.DYNAMIC_BASE,
            num_layers=1,
        ),
        metadata={"help": "`MultiHeadAttention` configuration"},
    )
    transformer_layer_config: TransformerLayerConfig = field(
        default_factory=lambda: TransformerLayerConfig(
            layer_norm_position=LayerNormPositionOptions.DEFAULT,
            dropout_probability=0.0,
            layer_norm_dim=0,
        ),
        metadata={"help": "`MultiHeadAttention` configuration"},
    )

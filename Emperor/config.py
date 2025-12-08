from torch import float32
import torch.nn as nn
from dataclasses import dataclass, field

from Emperor.attention.attention import MultiHeadAttentionConfig
from Emperor.base.enums import ActivationOptions, LayerNormPositionOptions
from Emperor.base.utils import ConfigBase
from Emperor.feedForward.feed_forward import (
    FeedForwardConfig,
    MixtureOfExpertsFeedForwardConfig,
)
from Emperor.experts.experts import MixtureOfExpertsConfig
from Emperor.adaptive.utils.layers import ParameterLayerConfig
from Emperor.base.layer import LayerStackConfig
from Emperor.adaptive.options import AdaptiveLayerOptions
from Emperor.adaptive.utils.mixtures.base import MixtureConfig
from Emperor.linears.options import LinearLayerOptions, LinearLayerStackOptions
from Emperor.linears.utils.layers import LinearLayerConfig
from Emperor.sampler.utils.samplers import SamplerConfig
from Emperor.sampler.utils.routers import RouterConfig
from Emperor.neuron.neuron import (
    AxonsConfig,
    NeuronClusterConfig,
    NucleusConfig,
    TerminalConfig,
    TerminalRangeOptions,
)
from Emperor.transformer.layer import TransformerConfig, TransformerLayerConfig


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
class ModelConfig(ConfigBase):
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
    mixture_model_config: MixtureConfig = field(
        default_factory=lambda: MixtureConfig(
            input_dim=MIXTURE_INPUT_DIM,
            output_dim=MIXTURE_OUTPUT_DIM,
            top_k=MIXTURE_TOP_K,
            weighted_parameters_flag=MIXTURE_WEIGHTED_PARAMETERS_FLAG,
            num_experts=MIXTURE_NUM_EXPERTS,
        ),
        metadata={"help": "`MixtureConfig` configuration"},
    )
    parameter_generator_model_config: ParameterLayerConfig = field(
        default_factory=lambda: ParameterLayerConfig(
            bias_parameters_flag=PARAMETER_GENERATOR_BIAS_PARAMETER_FLAG,
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
            init_sampler_model_flag=False,
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
            init_sampler_model_flag=False,
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
            use_separate_projection_weight_flag=False,
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
            layer_norm_position=LayerNormPositionOptions.NONE,
            residual_flag=False,
            adaptive_computation_flag=False,
        ),
        metadata={"help": "`MultiHeadAttention` configuration"},
    )
    transformer_feed_forward_config: FeedForwardConfig = field(
        default_factory=lambda: FeedForwardConfig(
            model_type=LinearLayerOptions.ADAPTIVE,
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
    transformer_config: TransformerConfig | None = field(
        default_factory=lambda: TransformerConfig(
            num_layers=2,
            source_sequence_length=0,
            target_sequence_length=0,
            layer_norm_dim=0,
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


# class LinearLayerConfigGenerator:
#     def __init__(
#         self,
#         input_dim: int,
#         output_dim: int,
#         bias_flag: bool,
#         anti_diagonal_flag: bool,
#         dynamic_bias_flag: bool,
#     ):
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.bias_flag = bias_flag
#         self.anti_diagonal_flag = anti_diagonal_flag
#         self.dynamic_bias_flag = dynamic_bias_flag
#
#     def build(self) -> LinearLayerConfig:
#         return LinearLayerConfig(
#             input_dim=self.input_dim,
#             output_dim=self.output_dim,
#             bias_flag=self.bias_flag,
#             anti_diagonal_flag=self.anti_diagonal_flag,
#             dynamic_bias_flag=self.dynamic_bias_flag,
#         )
#
#
# class RouterConfigGenerator:
#     def __init__(
#         self,
#         input_dim: int,
#         hidden_dim: int,
#         num_experts: int,
#         residual_flag: bool,
#         noisy_topk_flag: int,
#         activation: nn.Module,
#         num_layers: int | None,
#         diagonal_model_type_flag: bool,
#     ):
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.num_experts = num_experts
#         self.residual_flag = residual_flag
#         self.noisy_topk_flag = noisy_topk_flag
#         self.activation = activation
#         self.num_layers = num_layers
#         self.diagonal_model_type_flag = diagonal_model_type_flag
#
#     def build_router_config(self) -> RouterConfig:
#         return RouterConfig(
#             input_dim=self.input_dim,
#             hidden_dim=self.hidden_dim,
#             num_experts=self.num_experts,
#             residual_flag=self.residual_flag,
#             noisy_topk_flag=self.noisy_topk_flag,
#             activation=self.activation,
#             num_layers=self.num_layers,
#             diagonal_model_type_flag=self.diagonal_model_type_flag,
#         )
#
#
# class SamplerConfigGenerator:
#     def __init__(
#         self,
#         input_dim: int,
#         hidden_dim: int,
#         num_experts: int,
#         residual_flag: bool,
#         noisy_topk_flag: bool,
#         activation: nn.Module,
#         num_layers: int,
#         diagonal_model_type_flag: bool,
#         top_k: int,
#         threshold: float,
#         filter_above_threshold: bool,
#         num_topk_samples: int,
#         normalize_probabilities_flag: bool,
#         coefficient_of_variation_loss_weight: float,
#         switch_loss_weight: float,
#         zero_centred_loss_weight: float,
#         mutual_information_loss_weight: float,
#     ):
#         super().__init__(
#             input_dim,
#             hidden_dim,
#             num_experts,
#             residual_flag,
#             noisy_topk_flag,
#             activation,
#             num_layers,
#             diagonal_model_type_flag,
#         )
#         self.top_k = top_k
#         self.threshold = threshold
#         self.filter_above_threshold = filter_above_threshold
#         self.num_topk_samples = num_topk_samples
#         self.normalize_probabilities_flag = normalize_probabilities_flag
#         self.coefficient_of_variation_loss_weight = coefficient_of_variation_loss_weight
#         self.switch_loss_weight = switch_loss_weight
#         self.zero_centred_loss_weight = zero_centred_loss_weight
#         self.mutual_information_loss_weight = mutual_information_loss_weight
#
#     def build_sampler_config(self) -> SamplerConfig:
#         return SamplerConfig(
#             top_k=self.top_k,
#             threshold=self.threshold,
#             filter_above_threshold=self.filter_above_threshold,
#             num_topk_samples=self.num_topk_samples,
#             normalize_probabilities_flag=self.normalize_probabilities_flag,
#             noisy_topk_flag=self.noisy_topk_flag,
#             num_experts=self.num_experts,
#             coefficient_of_variation_loss_weight=self.coefficient_of_variation_loss_weight,
#             switch_loss_weight=self.switch_loss_weight,
#             zero_centred_loss_weight=self.zero_centred_loss_weight,
#             mutual_information_loss_weight=self.mutual_information_loss_weight,
#             router_model_config=super().build_router_config(),
#         )
#
#
# class ParameterLayerMixtureConfigGenerator(SamplerConfigGenerator):
#     def __init__(
#         self,
#         input_dim: int,
#         hidden_dim: int,
#         num_experts: int,
#         residual_flag: bool,
#         noisy_topk_flag: bool,
#         activation: nn.Module,
#         num_layers: int,
#         diagonal_model_type_flag: bool,
#         top_k: int,
#         threshold: float,
#         filter_above_threshold: bool,
#         num_topk_samples: int,
#         normalize_probabilities_flag: bool,
#         coefficient_of_variation_loss_weight: float,
#         switch_loss_weight: float,
#         zero_centred_loss_weight: float,
#         mutual_information_loss_weight: float,
#         output_dim: int,
#         weighted_parameters_flag: bool,
#         bias_parameters_flag: bool,
#         dynamic_diagonal_params_flag: bool,
#     ):
#         super().__init__(
#             input_dim,
#             hidden_dim,
#             num_experts,
#             residual_flag,
#             noisy_topk_flag,
#             activation,
#             num_layers,
#             diagonal_model_type_flag,
#             top_k,
#             threshold,
#             filter_above_threshold,
#             num_topk_samples,
#             normalize_probabilities_flag,
#             coefficient_of_variation_loss_weight,
#             switch_loss_weight,
#             zero_centred_loss_weight,
#             mutual_information_loss_weight,
#         )
#         self.output_dim = output_dim
#         self.weighted_parameters_flag = weighted_parameters_flag
#         self.bias_parameters_flag = bias_parameters_flag
#         self.dynamic_diagonal_params_flag = dynamic_diagonal_params_flag
#
#     def build_mixture_config(self) -> MixtureConfig:
#         return MixtureConfig(
#             input_dim=self.input_dim,
#             output_dim=self.output_dim,
#             depth_dim=self.num_experts,
#             top_k=self.top_k,
#             weighted_parameters_flag=self.weighted_parameters_flag,
#             bias_parameters_flag=self.bias_parameters_flag,
#             num_experts=self.num_experts,
#             dynamic_diagonal_params_flag=self.dynamic_diagonal_params_flag,
#             router_model_config=super().build_router_config(),
#             sampler_model_config=super().build_sampler_config(),
#         )
#
#
# class ParameterLayerConfigGenerator(ParameterLayerMixtureConfigGenerator):
#     def __init__(
#         input_dim: int,
#         hidden_dim: int,
#         num_experts: int,
#         residual_flag: bool,
#         noisy_topk_flag: bool,
#         activation: nn.Module,
#         num_layers: int,
#         diagonal_model_type_flag: bool,
#         top_k: int,
#         threshold: float,
#         filter_above_threshold: bool,
#         num_topk_samples: int,
#         normalize_probabilities_flag: bool,
#         coefficient_of_variation_loss_weight: float,
#         switch_loss_weight: float,
#         zero_centred_loss_weight: float,
#         mutual_information_loss_weight: float,
#         output_dim: int,
#         weighted_parameters_flag: bool,
#         bias_parameters_flag: bool,
#         dynamic_diagonal_params_flag: bool,
#     ):
#         super().__init__(
#             input_dim,
#             hidden_dim,
#             num_experts,
#             residual_flag,
#             noisy_topk_flag,
#             activation,
#             num_layers,
#             diagonal_model_type_flag,
#             top_k,
#             threshold,
#             filter_above_threshold,
#             num_topk_samples,
#             normalize_probabilities_flag,
#             coefficient_of_variation_loss_weight,
#             switch_loss_weight,
#             zero_centred_loss_weight,
#             mutual_information_loss_weight,
#             output_dim,
#             weighted_parameters_flag,
#             bias_parameters_flag,
#             dynamic_diagonal_params_flag,
#         )
#
#     def build_parameter_generator_flag(self) -> ParameterLayerConfig:
#         return ParameterLayerConfig(
#             bias_parameters_flag=self.bias_parameters_flag,
#             time_tracker_flag=self.time_tracker_flag,
#             dynamic_diagonal_params_flag=self.dynamic_diagonal_params_flag,
#             router_model_config=super().build_router_config(),
#             sampler_model_config=super().build_sampler_config(),
#             mixture_model_config=super().build_mixture_config(),
#         )
#
#
# class MixtureOfExpertsConfigGenerator(ParameterLayerConfigGenerator):
#     def __init__(
#         self,
#         input_dim: int,
#         hidden_dim: int,
#         num_experts: int,
#         residual_flag: bool,
#         noisy_topk_flag: bool,
#         activation: nn.Module,
#         num_layers: int,
#         diagonal_model_type_flag: bool,
#         top_k: int,
#         threshold: float,
#         filter_above_threshold: bool,
#         num_topk_samples: int,
#         normalize_probabilities_flag: bool,
#         coefficient_of_variation_loss_weight: float,
#         switch_loss_weight: float,
#         zero_centred_loss_weight: float,
#         mutual_information_loss_weight: float,
#         output_dim: int,
#         weighted_parameters_flag: bool,
#         bias_parameters_flag: bool,
#         dynamic_diagonal_params_flag: bool,
#         # Mixture of experts
#         expert_top_k: int,
#         dropout_probability: float,
#         expert_layer_norm_flag: bool,
#         expert_activation: ActivationOptions,
#         model_type: LayerTypes,
#         expert_num_experts: int,
#         compute_expert_mixture_flag: bool,
#         init_sampler_model_flag: bool,
#     ):
#         super().__init__(
#             input_dim,
#             hidden_dim,
#             num_experts,
#             residual_flag,
#             noisy_topk_flag,
#             activation,
#             num_layers,
#             diagonal_model_type_flag,
#             top_k,
#             threshold,
#             filter_above_threshold,
#             num_topk_samples,
#             normalize_probabilities_flag,
#             coefficient_of_variation_loss_weight,
#             switch_loss_weight,
#             zero_centred_loss_weight,
#             mutual_information_loss_weight,
#             output_dim,
#             weighted_parameters_flag,
#             bias_parameters_flag,
#             dynamic_diagonal_params_flag,
#         )
#
#         self.expert_top_k = expert_top_k
#         self.dropout_probability = dropout_probability
#         self.expert_layer_norm_flag = expert_layer_norm_flag
#         self.expert_activation = expert_activation
#         self.model_type = model_type
#         self.expert_num_experts = expert_num_experts
#         self.compute_expert_mixture_flag = compute_expert_mixture_flag
#         self.init_sampler_model_flag = init_sampler_model_flag
#
#     def build(self) -> MixtureOfExpertsConfig:
#         return MixtureOfExpertsConfig(
#             top_k=self.expert_top_k,
#             dropout_probability=self.dropout_probability,
#             layer_norm_flag=self.expert_layer_norm_flag,
#             activation=self.expert_activation,
#             model_type=self.model_type,
#             num_experts=self.expert_num_experts,
#             compute_expert_mixture_flag=self.compute_expert_mixture_flag,
#             weighted_parameters_flag=self.weighted_parameters_flag,
#             init_sampler_model_flag=self.init_sampler_model_flag,
#             router_model_config=super().build_router_config(),
#             sampler_model_config=super().build_sampler_config(),
#             mixture_model_config=super().build_mixture_config(),
#             parameter_generator_model_config=super().build_parameter_generator_flag(),
#         )
#

# class ParameterLayerConfigGenerator(LinearLayerConfigGenerator):
#     def __init__(
#         self,
#         input_dim: int,
#         output_dim: int,
#         bias_flag: bool,
#         anti_diagonal_flag: bool,
#         dynamic_bias_flag: bool,
#         bias_parameters_flag: bool,
#         time_tracker_flag: bool,
#         dynamic_diagonal_params_flag: bool,
#         linear_layer_model_type: LinearLayerTypes,
#     ):
#         super().__init__(
#             input_dim,
#             output_dim,
#             bias_flag,
#             anti_diagonal_flag,
#             dynamic_bias_flag,
#         )
#         self.bias_parameters_flag = bias_parameters_flag
#         self.time_tracker_flag = time_tracker_flag
#         self.dynamic_diagonal_params_flag = dynamic_diagonal_params_flag
#         self.linear_layer_model_type = linear_layer_model_type
#
#     def build(self) -> ParameterLayerConfig:
#         return ParameterLayerConfig(
#             bias_parameters_flag=self.bias_parameters_flag,
#             time_tracker_flag=self.time_tracker_flag,
#             dynamic_diagonal_params_flag=self.dynamic_diagonal_params_flag,
#             linear_layer_model_type=self.linear_layer_model_type,
#             linear_layer_config=super().build(),
#         )


# class MixtureOfExpertsConfigGenerator(ParameterLayerConfigGenerator):
#     def __init__(
#         self,
#         # Lienar layer options
#         input_dim: int,
#         output_dim: int,
#         bias_flag: bool,
#         anti_diagonal_flag: bool,
#         dynamic_bias_flag: bool,
#         # Parameter layer options
#         bias_parameters_flag: bool,
#         time_tracker_flag: bool,
#         dynamic_diagonal_params_flag: bool,
#         linear_layer_model_type: LinearLayerTypes,
#         # Muxtire of epxerts layer
#         top_k: int,
#         dropout_probability: float,
#         layer_norm_flag: bool,
#         activation: ActivationOptions,
#         num_experts: int,
#         compute_expert_mixture_flag: bool,
#         weighted_parameters_flag: bool,
#         init_sampler_model_flag: bool,
#     ):
#         super().__init__(
#             input_dim,
#             output_dim,
#             bias_flag,
#             anti_diagonal_flag,
#             dynamic_bias_flag,
#             bias_parameters_flag,
#             time_tracker_flag,
#             dynamic_diagonal_params_flag,
#             linear_layer_model_type,
#         )
#         self.top_k = top_k
#         self.dropout_probability = dropout_probability
#         self.layer_norm_flag = layer_norm_flag
#         self.activation = activation
#         self.num_experts = num_experts
#         self.compute_expert_mixture_flag = compute_expert_mixture_flag
#         self.weighted_parameters_flag = weighted_parameters_flag
#         self.init_sampler_model_flag = init_sampler_model_flag
#
#     def build(self) -> MixtureOfExpertsConfig:
#         return MixtureOfExpertsConfig(
#             top_k=MIXTURE_TOP_K,
#             dropout_probability=self.dropout_probability,
#             layer_norm_flag=True,
#             activation=ActivationOptions.GELU,
#             model_type=LayerTypes.DYNAMIC_BASE,
#             num_experts=12,
#             compute_expert_mixture_flag=True,
#             weighted_parameters_flag=True,
#             init_sampler_model_flag=False,
#         )

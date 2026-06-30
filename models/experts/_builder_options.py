from dataclasses import dataclass

from emperor.base.layer.gate import GateConfig, LayerGateOptions
from emperor.base.layer.residual import ResidualConnectionOptions
from emperor.base.options import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
)
from emperor.experts.core.options import (
    DroppedTokenOptions,
    ExpertWeightingPositionOptions,
    RoutingInitializationMode,
)
from emperor.halting.options import HaltingHiddenStateModeOptions


@dataclass(frozen=True)
class ExpertsStackOptions:
    hidden_dim: int
    bias_flag: bool
    layer_norm_position: LayerNormPositionOptions
    num_layers: int
    activation: ActivationOptions
    residual_connection_option: ResidualConnectionOptions
    dropout_probability: float
    last_layer_bias_option: LastLayerBiasOptions
    apply_output_pipeline_flag: bool


@dataclass(frozen=True)
class ExpertsControllerStackOptions:
    hidden_dim: int
    num_layers: int
    last_layer_bias_option: LastLayerBiasOptions
    apply_output_pipeline_flag: bool
    activation: ActivationOptions
    layer_norm_position: LayerNormPositionOptions
    residual_connection_option: ResidualConnectionOptions
    dropout_probability: float
    bias_flag: bool


@dataclass(frozen=True)
class ExpertsMixtureOptions:
    top_k: int
    num_experts: int
    capacity_factor: float
    dropped_token_behavior: DroppedTokenOptions
    compute_expert_mixture_flag: bool
    weighted_parameters_flag: bool
    weighting_position_option: ExpertWeightingPositionOptions
    routing_initialization_mode: RoutingInitializationMode


@dataclass(frozen=True)
class ExpertsSamplerOptions:
    threshold: float
    filter_above_threshold: bool
    num_topk_samples: int
    normalize_probabilities_flag: bool
    noisy_topk_flag: bool
    coefficient_of_variation_loss_weight: float
    switch_loss_weight: float
    zero_centred_loss_weight: float
    mutual_information_loss_weight: float


@dataclass(frozen=True)
class ExpertsRouterOptions:
    noisy_topk_flag: bool


@dataclass(frozen=True)
class ExpertsLayerControllerOptions:
    stack_gate_flag: bool
    gate_option: LayerGateOptions | None
    gate_activation: ActivationOptions | None
    gate_stack_options: ExpertsControllerStackOptions
    stack_halting_flag: bool
    halting_threshold: float
    halting_dropout: float
    halting_hidden_state_mode: HaltingHiddenStateModeOptions
    halting_stack_options: ExpertsControllerStackOptions
    halting_output_dim: int
    shared_gate_config: GateConfig | None = None


@dataclass(frozen=True)
class ExpertsRecurrentControllerOptions:
    recurrent_flag: bool
    recurrent_max_steps: int
    recurrent_layer_norm_position: LayerNormPositionOptions
    recurrent_gate_flag: bool
    recurrent_gate_option: LayerGateOptions | None
    recurrent_gate_activation: ActivationOptions | None
    recurrent_halting_flag: bool


@dataclass(frozen=True)
class ExpertsAdaptiveGeneratorStackOptions:
    hidden_dim: int
    num_layers: int
    last_layer_bias_option: LastLayerBiasOptions
    apply_output_pipeline_flag: bool
    activation: ActivationOptions
    layer_norm_position: LayerNormPositionOptions
    residual_connection_option: ResidualConnectionOptions
    dropout_probability: float

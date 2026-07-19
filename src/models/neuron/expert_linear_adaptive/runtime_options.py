from dataclasses import dataclass

from emperor.base.layer.residual import ResidualConnectionOptions
from emperor.base.options import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
)
from emperor.halting.options import HaltingHiddenStateModeOptions
from emperor.neuron.core.options import TerminalRangeOptions, TerminalZAxisOffsetOptions


@dataclass(frozen=True)
class NeuronClusterCapacityOptions:
    x_axis_total_neurons: int
    y_axis_total_neurons: int
    z_axis_total_neurons: int
    initial_x_axis_total_neurons: int | None
    initial_y_axis_total_neurons: int | None
    initial_z_axis_total_neurons: int | None
    max_steps: int
    growth_threshold: int | None


@dataclass(frozen=True)
class NeuronTerminalOptions:
    xy_axis_range: TerminalRangeOptions
    z_axis_range: TerminalRangeOptions
    z_axis_offset: TerminalZAxisOffsetOptions
    top_k: int


@dataclass(frozen=True)
class NeuronSubmoduleStackOptions:
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
class NeuronTerminalSamplerOptions:
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
class ClusterRouteHaltingOptions:
    enabled: bool
    threshold: float
    dropout: float
    hidden_state_mode: HaltingHiddenStateModeOptions
    stack_options: NeuronSubmoduleStackOptions
    output_dim: int


# Preserve historical class paths for serialization compatibility.
NeuronClusterCapacityOptions.__module__ = (
    "models.neuron.expert_linear_adaptive._neuron_options"
)
NeuronTerminalOptions.__module__ = (
    "models.neuron.expert_linear_adaptive._neuron_options"
)
NeuronSubmoduleStackOptions.__module__ = (
    "models.neuron.expert_linear_adaptive._neuron_options"
)
NeuronTerminalSamplerOptions.__module__ = (
    "models.neuron.expert_linear_adaptive._neuron_options"
)
ClusterRouteHaltingOptions.__module__ = (
    "models.neuron.expert_linear_adaptive._neuron_options"
)

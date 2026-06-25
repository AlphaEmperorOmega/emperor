import copy
from typing import Any

from emperor.base.layer import LayerConfig, LayerStackConfig
from emperor.halting.config import StickBreakingConfig
from emperor.linears.core.config import LinearLayerConfig
from emperor.neuron.core.config import (
    AxonsConfig,
    NeuronClusterConfig,
    NeuronConfig,
    NucleusConfig,
    TerminalConfig,
)
from emperor.sampler.core.config import RouterConfig, SamplerConfig

from models.neuron.neuron_linear.experiment_config import HiddenBlockConfig


class NeuronControlConfigFactory:
    def __init__(self, builder: Any) -> None:
        self.builder = builder

    def build(
        self,
        source_hidden_model_config,
        hidden_dim: int,
    ) -> NeuronClusterConfig:
        hidden_block_config = HiddenBlockConfig(
            input_dim=hidden_dim,
            output_dim=hidden_dim,
            model_config=copy.deepcopy(source_hidden_model_config),
        )
        terminal_sampler_config = self._build_terminal_sampler_config(hidden_dim)
        neuron_config = NeuronConfig(
            nucleus_config=NucleusConfig(model_config=hidden_block_config),
            axons_config=AxonsConfig(memory_config=None),
            terminal_config=TerminalConfig(
                input_dim=hidden_dim,
                xy_axis_range=self.builder.cluster_terminal_xy_axis_range,
                z_axis_range=self.builder.cluster_terminal_z_axis_range,
                z_axis_offset=self.builder.cluster_terminal_z_axis_offset,
                sampler_config=terminal_sampler_config,
            ),
        )
        return NeuronClusterConfig(
            x_axis_total_neurons=self.builder.cluster_x_axis_total_neurons,
            y_axis_total_neurons=self.builder.cluster_y_axis_total_neurons,
            z_axis_total_neurons=self.builder.cluster_z_axis_total_neurons,
            initial_x_axis_total_neurons=(
                self.builder.cluster_initial_x_axis_total_neurons
            ),
            initial_y_axis_total_neurons=(
                self.builder.cluster_initial_y_axis_total_neurons
            ),
            initial_z_axis_total_neurons=(
                self.builder.cluster_initial_z_axis_total_neurons
            ),
            entry_sampler_config=None,
            max_steps=self.builder.cluster_max_steps,
            growth_threshold=self.builder.cluster_growth_threshold,
            halting_config=self._build_cluster_halting_config(hidden_dim),
            neuron_config=neuron_config,
        )

    def _build_terminal_sampler_config(self, hidden_dim: int) -> SamplerConfig:
        num_experts = self._terminal_num_experts()
        top_k = min(max(1, self.builder.cluster_terminal_top_k), num_experts)
        return SamplerConfig(
            top_k=top_k,
            threshold=self.builder.cluster_terminal_sampler_threshold,
            filter_above_threshold=(
                self.builder.cluster_terminal_sampler_filter_above_threshold
            ),
            num_topk_samples=min(
                self.builder.cluster_terminal_sampler_num_topk_samples,
                top_k,
            ),
            normalize_probabilities_flag=(
                self.builder.cluster_terminal_sampler_normalize_probabilities_flag
            ),
            noisy_topk_flag=self.builder.cluster_terminal_sampler_noisy_topk_flag,
            num_experts=num_experts,
            coefficient_of_variation_loss_weight=(
                self.builder.cluster_terminal_sampler_coefficient_of_variation_loss_weight
            ),
            switch_loss_weight=(
                self.builder.cluster_terminal_sampler_switch_loss_weight
            ),
            zero_centred_loss_weight=(
                self.builder.cluster_terminal_sampler_zero_centred_loss_weight
            ),
            mutual_information_loss_weight=(
                self.builder.cluster_terminal_sampler_mutual_information_loss_weight
            ),
            router_config=self._build_router_config(hidden_dim, num_experts),
        )

    def _build_router_config(
        self,
        hidden_dim: int,
        num_experts: int,
    ) -> RouterConfig:
        router_hidden_dim = self.builder.cluster_terminal_router_hidden_dim or max(
            hidden_dim,
            num_experts,
        )
        return RouterConfig(
            input_dim=hidden_dim,
            num_experts=num_experts,
            noisy_topk_flag=self.builder.cluster_terminal_sampler_noisy_topk_flag,
            model_config=LayerStackConfig(
                input_dim=hidden_dim,
                hidden_dim=router_hidden_dim,
                output_dim=num_experts,
                num_layers=self.builder.cluster_terminal_router_num_layers,
                last_layer_bias_option=(
                    self.builder.cluster_terminal_router_last_layer_bias_option
                ),
                apply_output_pipeline_flag=(
                    self.builder.cluster_terminal_router_apply_output_pipeline_flag
                ),
                layer_config=LayerConfig(
                    activation=self.builder.cluster_terminal_router_activation,
                    residual_connection_option=(
                        self.builder.cluster_terminal_router_residual_connection_option
                    ),
                    dropout_probability=(
                        self.builder.cluster_terminal_router_dropout_probability
                    ),
                    layer_norm_position=(
                        self.builder.cluster_terminal_router_layer_norm_position
                    ),
                    gate_config=None,
                    halting_config=None,
                    memory_config=None,
                    layer_model_config=LinearLayerConfig(
                        bias_flag=self.builder.cluster_terminal_router_bias_flag
                    ),
                ),
            ),
        )

    def _build_cluster_halting_config(
        self,
        hidden_dim: int,
    ) -> StickBreakingConfig | None:
        if not self.builder.cluster_halting_flag:
            return None
        return StickBreakingConfig(
            input_dim=hidden_dim,
            threshold=self.builder.cluster_halting_threshold,
            halting_dropout=self.builder.cluster_halting_dropout,
            hidden_state_mode=self.builder.cluster_halting_hidden_state_mode,
            halting_gate_config=LayerStackConfig(
                input_dim=hidden_dim,
                hidden_dim=self.builder.cluster_halting_stack_hidden_dim,
                output_dim=self.builder.cluster_halting_output_dim,
                num_layers=self.builder.cluster_halting_stack_num_layers,
                last_layer_bias_option=(
                    self.builder.cluster_halting_stack_last_layer_bias_option
                ),
                apply_output_pipeline_flag=(
                    self.builder.cluster_halting_stack_apply_output_pipeline_flag
                ),
                layer_config=LayerConfig(
                    activation=self.builder.cluster_halting_stack_activation,
                    residual_connection_option=(
                        self.builder.cluster_halting_stack_residual_connection_option
                    ),
                    dropout_probability=(
                        self.builder.cluster_halting_stack_dropout_probability
                    ),
                    layer_norm_position=(
                        self.builder.cluster_halting_stack_layer_norm_position
                    ),
                    gate_config=None,
                    halting_config=None,
                    memory_config=None,
                    layer_model_config=LinearLayerConfig(
                        bias_flag=self.builder.cluster_halting_stack_bias_flag
                    ),
                ),
            ),
        )

    def _terminal_num_experts(self) -> int:
        xy_range = self._enum_or_int_value(
            self.builder.cluster_terminal_xy_axis_range
        )
        z_range = self._enum_or_int_value(self.builder.cluster_terminal_z_axis_range)
        return (xy_range * 2 + 1) ** 2 * (z_range + 1)

    def _enum_or_int_value(self, value) -> int:
        return int(value.value if hasattr(value, "value") else value)

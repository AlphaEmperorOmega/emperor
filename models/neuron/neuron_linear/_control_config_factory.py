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
        terminal_options = self.builder.terminal_options
        neuron_config = NeuronConfig(
            nucleus_config=NucleusConfig(model_config=hidden_block_config),
            axons_config=AxonsConfig(memory_config=None),
            terminal_config=TerminalConfig(
                input_dim=hidden_dim,
                xy_axis_range=terminal_options.xy_axis_range,
                z_axis_range=terminal_options.z_axis_range,
                z_axis_offset=terminal_options.z_axis_offset,
                sampler_config=terminal_sampler_config,
            ),
        )
        capacity_options = self.builder.cluster_capacity_options
        return NeuronClusterConfig(
            x_axis_total_neurons=capacity_options.x_axis_total_neurons,
            y_axis_total_neurons=capacity_options.y_axis_total_neurons,
            z_axis_total_neurons=capacity_options.z_axis_total_neurons,
            initial_x_axis_total_neurons=(
                capacity_options.initial_x_axis_total_neurons
            ),
            initial_y_axis_total_neurons=(
                capacity_options.initial_y_axis_total_neurons
            ),
            initial_z_axis_total_neurons=(
                capacity_options.initial_z_axis_total_neurons
            ),
            entry_sampler_config=None,
            max_steps=capacity_options.max_steps,
            growth_threshold=capacity_options.growth_threshold,
            halting_config=self._build_cluster_halting_config(hidden_dim),
            neuron_config=neuron_config,
        )

    def _build_terminal_sampler_config(self, hidden_dim: int) -> SamplerConfig:
        num_experts = self._terminal_num_experts()
        terminal_options = self.builder.terminal_options
        sampler_options = self.builder.terminal_sampler_options
        top_k = min(max(1, terminal_options.top_k), num_experts)
        return SamplerConfig(
            top_k=top_k,
            threshold=sampler_options.threshold,
            filter_above_threshold=sampler_options.filter_above_threshold,
            num_topk_samples=min(sampler_options.num_topk_samples, top_k),
            normalize_probabilities_flag=sampler_options.normalize_probabilities_flag,
            noisy_topk_flag=sampler_options.noisy_topk_flag,
            num_experts=num_experts,
            coefficient_of_variation_loss_weight=(
                sampler_options.coefficient_of_variation_loss_weight
            ),
            switch_loss_weight=sampler_options.switch_loss_weight,
            zero_centred_loss_weight=sampler_options.zero_centred_loss_weight,
            mutual_information_loss_weight=sampler_options.mutual_information_loss_weight,
            router_config=self._build_router_config(hidden_dim, num_experts),
        )

    def _build_router_config(
        self,
        hidden_dim: int,
        num_experts: int,
    ) -> RouterConfig:
        router_options = self.builder.terminal_router_options
        router_hidden_dim = router_options.hidden_dim or max(
            hidden_dim,
            num_experts,
        )
        return RouterConfig(
            input_dim=hidden_dim,
            num_experts=num_experts,
            noisy_topk_flag=self.builder.terminal_sampler_options.noisy_topk_flag,
            model_config=LayerStackConfig(
                input_dim=hidden_dim,
                hidden_dim=router_hidden_dim,
                output_dim=num_experts,
                num_layers=router_options.num_layers,
                last_layer_bias_option=router_options.last_layer_bias_option,
                apply_output_pipeline_flag=router_options.apply_output_pipeline_flag,
                layer_config=LayerConfig(
                    activation=router_options.activation,
                    residual_connection_option=(
                        router_options.residual_connection_option
                    ),
                    dropout_probability=router_options.dropout_probability,
                    layer_norm_position=router_options.layer_norm_position,
                    gate_config=None,
                    halting_config=None,
                    memory_config=None,
                    layer_model_config=LinearLayerConfig(
                        bias_flag=router_options.bias_flag
                    ),
                ),
            ),
        )

    def _build_cluster_halting_config(
        self,
        hidden_dim: int,
    ) -> StickBreakingConfig | None:
        halting_options = self.builder.cluster_halting_options
        if not halting_options.enabled:
            return None
        stack_options = halting_options.stack_options
        return StickBreakingConfig(
            input_dim=hidden_dim,
            threshold=halting_options.threshold,
            halting_dropout=halting_options.dropout,
            hidden_state_mode=halting_options.hidden_state_mode,
            halting_gate_config=LayerStackConfig(
                input_dim=hidden_dim,
                hidden_dim=stack_options.hidden_dim,
                output_dim=halting_options.output_dim,
                num_layers=stack_options.num_layers,
                last_layer_bias_option=stack_options.last_layer_bias_option,
                apply_output_pipeline_flag=stack_options.apply_output_pipeline_flag,
                layer_config=LayerConfig(
                    activation=stack_options.activation,
                    residual_connection_option=(
                        stack_options.residual_connection_option
                    ),
                    dropout_probability=stack_options.dropout_probability,
                    layer_norm_position=stack_options.layer_norm_position,
                    gate_config=None,
                    halting_config=None,
                    memory_config=None,
                    layer_model_config=LinearLayerConfig(
                        bias_flag=stack_options.bias_flag
                    ),
                ),
            ),
        )

    def _terminal_num_experts(self) -> int:
        terminal_options = self.builder.terminal_options
        xy_range = self._enum_or_int_value(terminal_options.xy_axis_range)
        z_range = self._enum_or_int_value(terminal_options.z_axis_range)
        return (xy_range * 2 + 1) ** 2 * (z_range + 1)

    def _enum_or_int_value(self, value) -> int:
        return int(value.value if hasattr(value, "value") else value)

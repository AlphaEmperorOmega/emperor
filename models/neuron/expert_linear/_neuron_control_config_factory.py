import copy
from dataclasses import dataclass

from emperor.base.utils import ConfigBase
from emperor.halting.config import StickBreakingConfig
from emperor.neuron.core.config import (
    AxonsConfig,
    NeuronClusterConfig,
    NeuronConfig,
    NucleusConfig,
    TerminalConfig,
)
from emperor.sampler.core.config import RouterConfig, SamplerConfig

from models.neuron.expert_linear._hidden_block import HiddenBlockConfig
from models.neuron.expert_linear._neuron_controller_stack_config_factory import (
    NeuronControllerStackConfigFactory,
)
from models.neuron.expert_linear.runtime_options import (
    ClusterRouteHaltingOptions,
    NeuronClusterCapacityOptions,
    NeuronSubmoduleStackOptions,
    NeuronTerminalOptions,
    NeuronTerminalSamplerOptions,
)


@dataclass(frozen=True)
class NeuronControlConfigDependencies:
    cluster_capacity_options: NeuronClusterCapacityOptions
    terminal_options: NeuronTerminalOptions
    terminal_router_options: NeuronSubmoduleStackOptions
    terminal_sampler_options: NeuronTerminalSamplerOptions
    cluster_halting_options: ClusterRouteHaltingOptions


class NeuronControlConfigFactory:
    def __init__(self, dependencies: NeuronControlConfigDependencies) -> None:
        self.cluster_capacity_options = dependencies.cluster_capacity_options
        self.terminal_options = dependencies.terminal_options
        self.terminal_router_options = dependencies.terminal_router_options
        self.terminal_sampler_options = dependencies.terminal_sampler_options
        self.cluster_halting_options = dependencies.cluster_halting_options
        self.controller_stack_config_factory = NeuronControllerStackConfigFactory()

    def build(
        self,
        hidden_model_config: ConfigBase,
        hidden_dim: int,
    ) -> NeuronClusterConfig:
        hidden_block_config = HiddenBlockConfig(
            input_dim=hidden_dim,
            output_dim=hidden_dim,
            model_config=copy.deepcopy(hidden_model_config),
        )
        terminal_sampler_config = self.__build_terminal_sampler_config(hidden_dim)
        neuron_config = self.__build_neuron_config(
            hidden_dim,
            hidden_block_config,
            terminal_sampler_config,
        )
        return self.__build_neuron_cluster_config(
            hidden_dim,
            neuron_config,
        )

    def __build_neuron_config(
        self,
        hidden_dim: int,
        hidden_block_config: HiddenBlockConfig,
        terminal_sampler_config: SamplerConfig,
    ) -> NeuronConfig:
        terminal_options = self.terminal_options
        return NeuronConfig(
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

    def __build_neuron_cluster_config(
        self,
        hidden_dim: int,
        neuron_config: NeuronConfig,
    ) -> NeuronClusterConfig:
        capacity_options = self.cluster_capacity_options
        halting_config = self.__build_cluster_halting_config(hidden_dim)
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
            halting_config=halting_config,
            neuron_config=neuron_config,
        )

    def __build_terminal_sampler_config(self, hidden_dim: int) -> SamplerConfig:
        terminal_options = self.terminal_options
        sampler_options = self.terminal_sampler_options
        num_experts = self.__terminal_num_experts()
        top_k = min(max(1, terminal_options.top_k), num_experts)
        router_config = self.__build_router_config(hidden_dim, num_experts)
        return SamplerConfig(
            top_k=top_k,
            threshold=sampler_options.threshold,
            filter_above_threshold=sampler_options.filter_above_threshold,
            num_topk_samples=min(sampler_options.num_topk_samples, top_k),
            normalize_probabilities_flag=(sampler_options.normalize_probabilities_flag),
            noisy_topk_flag=sampler_options.noisy_topk_flag,
            num_experts=num_experts,
            coefficient_of_variation_loss_weight=(
                sampler_options.coefficient_of_variation_loss_weight
            ),
            switch_loss_weight=sampler_options.switch_loss_weight,
            zero_centred_loss_weight=sampler_options.zero_centred_loss_weight,
            mutual_information_loss_weight=(
                sampler_options.mutual_information_loss_weight
            ),
            router_config=router_config,
        )

    def __build_router_config(
        self,
        hidden_dim: int,
        num_experts: int,
    ) -> RouterConfig:
        router_options = self.terminal_router_options
        router_hidden_dim = router_options.hidden_dim or max(
            hidden_dim,
            num_experts,
        )
        model_config = self.controller_stack_config_factory.build_config(
            router_options,
            input_dim=hidden_dim,
            hidden_dim=router_hidden_dim,
            output_dim=num_experts,
        )
        return RouterConfig(
            input_dim=hidden_dim,
            num_experts=num_experts,
            noisy_topk_flag=self.terminal_sampler_options.noisy_topk_flag,
            model_config=model_config,
        )

    def __build_cluster_halting_config(
        self,
        hidden_dim: int,
    ) -> StickBreakingConfig | None:
        halting_options = self.cluster_halting_options
        if not halting_options.enabled:
            return None
        stack_options = halting_options.stack_options
        halting_gate_config = self.controller_stack_config_factory.build_config(
            stack_options,
            input_dim=hidden_dim,
            hidden_dim=stack_options.hidden_dim,
            output_dim=halting_options.output_dim,
        )
        return StickBreakingConfig(
            input_dim=hidden_dim,
            threshold=halting_options.threshold,
            halting_dropout=halting_options.dropout,
            hidden_state_mode=halting_options.hidden_state_mode,
            halting_gate_config=halting_gate_config,
        )

    def __terminal_num_experts(self) -> int:
        terminal_options = self.terminal_options
        xy_range = self.__enum_or_int_value(terminal_options.xy_axis_range)
        z_range = self.__enum_or_int_value(terminal_options.z_axis_range)
        return (xy_range * 2 + 1) ** 2 * (z_range + 1)

    @staticmethod
    def __enum_or_int_value(value) -> int:
        return int(value.value if hasattr(value, "value") else value)

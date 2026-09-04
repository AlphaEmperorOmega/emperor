from __future__ import annotations

from typing import TYPE_CHECKING

from torch import Tensor

from emperor.neuron._terminal.routing.node import RoutingTreeNode
from emperor.neuron._terminal.routing_tree_topology import RoutingTreeCompiler
from emperor.neuron._terminal.validation import RoutingTreeDelegateValidator
from emperor.nn import Module
from emperor.sampler import SamplerConfig, SamplerModel

if TYPE_CHECKING:
    from emperor.neuron._config import TerminalConfig, TerminalRoutingTreeConfig
    from emperor.neuron._terminal.routing_tree_topology import RoutingTreePlan


class RoutingTreeDelegate(Module):
    """Own and execute an independently parameterized spatial routing tree."""

    VALIDATOR = RoutingTreeDelegateValidator

    def __init__(
        self,
        cfg: TerminalConfig,
        neuron_connections: Tensor,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.input_dim: int = self.cfg.input_dim
        self.leaf_sampler_config: SamplerConfig = self.cfg.sampler_config
        self.routing_tree_config: TerminalRoutingTreeConfig = (
            self.cfg.routing_tree_config
        )
        self.VALIDATOR.validate_routing_tree_config(self.routing_tree_config)
        self.direction_sampler_config: SamplerConfig = (
            self.__resolve_direction_sampler_config()
        )
        self.neuron_connections = neuron_connections
        self.plan = self.compile_routing_tree_plan()
        self.output_width = self.plan.output_width
        self.root = self.__build_routing_tree_root_node()

    def __resolve_direction_sampler_config(self) -> SamplerConfig:
        return (
            self.routing_tree_config.direction_sampler_config
            or self.leaf_sampler_config
        )

    def compile_routing_tree_plan(self) -> RoutingTreePlan:
        routing_tree_compiler = RoutingTreeCompiler(
            neuron_connections=self.neuron_connections,
            routing_tree_config=self.routing_tree_config,
            leaf_top_k=self.leaf_sampler_config.top_k,
        )
        routing_tree_plan = routing_tree_compiler.compile()
        self.VALIDATOR.validate_routing_tree_plan(self, routing_tree_plan)
        return routing_tree_plan

    def __build_routing_tree_root_node(self) -> RoutingTreeNode:
        return RoutingTreeNode(
            input_dim=self.input_dim,
            plan=self.plan,
            node_plan=self.plan.root,
            leaf_sampler_config=self.leaf_sampler_config,
            direction_sampler_config=self.direction_sampler_config,
        )

    def sample_probabilities_and_indices(
        self,
        input_matrix: Tensor,
        skip_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, None, Tensor]:
        self.VALIDATOR.validate_forward_inputs(self, input_matrix, skip_mask)

        self.__reset_runtime_observations()
        probabilities, connection_indices, auxiliary_loss = self.root.route(
            input_matrix
        )
        sorted_probabilities, sorted_path_indices = probabilities.sort(
            dim=1,
            descending=True,
            stable=True,
        )
        sorted_connection_indices = connection_indices.gather(
            1,
            sorted_path_indices,
        )
        return sorted_probabilities, sorted_connection_indices, None, auxiliary_loss

    def __reset_runtime_observations(self) -> None:
        for module in self.modules():
            if isinstance(module, SamplerModel):
                module.reset_runtime_observation()

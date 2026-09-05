from __future__ import annotations

import copy
from collections.abc import Iterator
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from emperor.nn import Module
from emperor.sampler import SamplerConfig

if TYPE_CHECKING:
    from emperor.neuron._terminal.routing_tree_topology import (
        RoutingTreeNodePlan,
        RoutingTreePlan,
    )


class RoutingTreeNode(Module):
    """Own one direction or leaf sampler and route its selected input rows."""

    def __init__(
        self,
        *,
        input_dim: int,
        plan: RoutingTreePlan,
        node_plan: RoutingTreeNodePlan,
        leaf_sampler_config: SamplerConfig,
        direction_sampler_config: SamplerConfig,
    ) -> None:
        super().__init__()
        self.input_dim: int = input_dim
        self.plan: RoutingTreePlan = plan
        self.node_plan: RoutingTreeNodePlan = node_plan
        self.leaf_sampler_config: SamplerConfig = leaf_sampler_config
        self.direction_sampler_config: SamplerConfig = direction_sampler_config
        self.is_leaf: bool = self.node_plan.is_leaf
        self.sampler = self.__build_sampler()

    def __build_sampler(self):
        if self.is_leaf:
            node_sampler_config = self.__initialize_leaf_node()
        else:
            node_sampler_config = self.__initialize_direction_node()
        return node_sampler_config.build()

    def __initialize_leaf_node(self) -> SamplerConfig:
        node_sampler_config = RoutingTreeNode.derive_sampler_config(
            self.leaf_sampler_config,
            input_dim=self.input_dim,
            num_experts=len(self.node_plan.connection_indices),
            top_k=self.plan.leaf_top_k,
        )
        self.output_width = self.plan.leaf_top_k
        self.__initialize_global_connection_indices()
        self.branches = nn.ModuleList()
        return node_sampler_config

    @staticmethod
    def derive_sampler_config(
        template: SamplerConfig,
        *,
        input_dim: int,
        num_experts: int,
        top_k: int,
    ) -> SamplerConfig:
        """Derive an independent node config without constructing a sampler."""

        derived_config = copy.deepcopy(template)
        derived_config.num_experts = num_experts
        derived_config.top_k = top_k
        if derived_config.num_topk_samples is not None:
            derived_config.num_topk_samples = min(
                derived_config.num_topk_samples,
                top_k,
            )

        if derived_config.router_config is None:
            raise ValueError(
                "Terminal routing trees require learned router_config values for "
                "both direction and connection sampler templates."
            )
        derived_config.router_config.input_dim = input_dim
        derived_config.router_config.num_experts = num_experts
        derived_config.router_config.noisy_topk_flag = derived_config.noisy_topk_flag
        return derived_config

    def __initialize_global_connection_indices(self) -> None:
        global_connection_indices = torch.tensor(
            self.node_plan.connection_indices,
            dtype=torch.long,
        )
        self.register_buffer(
            "global_connection_indices",
            global_connection_indices,
            persistent=False,
        )

    def __initialize_direction_node(self) -> SamplerConfig:
        direction_top_k = self.plan.direction_top_k[self.node_plan.level]
        node_sampler_config = RoutingTreeNode.derive_sampler_config(
            self.direction_sampler_config,
            input_dim=self.input_dim,
            num_experts=len(self.node_plan.children),
            top_k=direction_top_k,
        )
        self.branches = self.__build_branches()
        child_output_width = self.branches[0].output_width
        self.output_width = direction_top_k * child_output_width
        return node_sampler_config

    def __build_branches(self) -> nn.ModuleList:
        return nn.ModuleList(
            RoutingTreeNode(
                input_dim=self.input_dim,
                plan=self.plan,
                node_plan=child_plan,
                leaf_sampler_config=self.leaf_sampler_config,
                direction_sampler_config=self.direction_sampler_config,
            )
            for child_plan in self.node_plan.children
        )

    def route(self, input_matrix: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        probabilities, selected_indices, _, auxiliary_loss = (
            self.sampler.sample_probabilities_and_indices(input_matrix)
        )
        probability_matrix = self.__ensure_matrix(probabilities)
        index_matrix = self.__resolve_index_matrix(
            selected_indices,
            batch_size=input_matrix.shape[0],
            num_experts=self.sampler.num_experts,
            device=input_matrix.device,
        )
        if self.is_leaf:
            global_indices = self.__select_global_connection_indices(index_matrix)
            return probability_matrix, global_indices, auxiliary_loss

        return self.__route_selected_children(
            input_matrix,
            probability_matrix,
            index_matrix,
            auxiliary_loss,
        )

    def __ensure_matrix(self, values: Tensor) -> Tensor:
        if values.dim() == 1:
            return values.unsqueeze(-1)
        return values

    def __resolve_index_matrix(
        self,
        indices: Tensor | None,
        *,
        batch_size: int,
        num_experts: int,
        device: torch.device,
    ) -> Tensor:
        if indices is not None:
            return self.__ensure_matrix(indices)
        all_connection_indices = torch.arange(
            num_experts,
            device=device,
            dtype=torch.long,
        )
        batched_connection_indices = all_connection_indices.expand(batch_size, -1)
        return batched_connection_indices

    def __select_global_connection_indices(self, index_matrix: Tensor) -> Tensor:
        device_aligned_global_indices = self.global_connection_indices.to(
            index_matrix.device
        )
        return device_aligned_global_indices[index_matrix]

    def __route_selected_children(
        self,
        input_matrix: Tensor,
        direction_probabilities: Tensor,
        selected_child_indices: Tensor,
        auxiliary_loss: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch_size, selected_direction_count = selected_child_indices.shape
        flattened_child_indices = selected_child_indices.reshape(-1)
        flattened_direction_probabilities = direction_probabilities.reshape(-1)
        flattened_original_inputs = self.__expand_inputs_for_selected_directions(
            input_matrix,
            selected_direction_count,
        )
        probability_paths, connection_paths, accumulated_auxiliary_loss = (
            self.__route_flattened_children(
                flattened_original_inputs,
                flattened_direction_probabilities,
                flattened_child_indices,
                auxiliary_loss,
            )
        )
        probability_paths, connection_paths = self.__restore_batch_path_layout(
            probability_paths,
            connection_paths,
            batch_size,
        )
        return probability_paths, connection_paths, accumulated_auxiliary_loss

    def __expand_inputs_for_selected_directions(
        self,
        input_matrix: Tensor,
        selected_direction_count: int,
    ) -> Tensor:
        input_direction_axis = input_matrix.unsqueeze(1)
        inputs_per_selected_direction = input_direction_axis.expand(
            -1,
            selected_direction_count,
            -1,
        )
        flattened_original_inputs = inputs_per_selected_direction.reshape(
            -1,
            self.input_dim,
        )
        return flattened_original_inputs

    def __route_flattened_children(
        self,
        input_matrix: Tensor,
        direction_probabilities: Tensor,
        selected_child_indices: Tensor,
        auxiliary_loss: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        probability_paths, connection_paths = self.__initialize_route_paths(
            direction_probabilities,
            selected_child_indices,
        )
        accumulated_auxiliary_loss = auxiliary_loss
        for child, selected_positions in self.__iter_selected_children(
            selected_child_indices
        ):
            joint_probabilities, child_connections, child_auxiliary_loss = (
                self.__route_selected_child(
                    child,
                    input_matrix,
                    direction_probabilities,
                    selected_positions,
                )
            )
            probability_paths, connection_paths = self.__merge_child_paths(
                probability_paths,
                connection_paths,
                selected_positions,
                joint_probabilities,
                child_connections,
            )
            accumulated_auxiliary_loss = (
                accumulated_auxiliary_loss + child_auxiliary_loss
            )
        return probability_paths, connection_paths, accumulated_auxiliary_loss

    def __initialize_route_paths(
        self,
        direction_probabilities: Tensor,
        selected_child_indices: Tensor,
    ) -> tuple[Tensor, Tensor]:
        selected_path_count = selected_child_indices.shape[0]
        child_output_width = self.branches[0].output_width
        probability_paths = direction_probabilities.new_zeros(
            selected_path_count,
            child_output_width,
        )
        if selected_path_count == 0:
            probability_paths = probability_paths + direction_probabilities.sum() * 0
        connection_paths = selected_child_indices.new_zeros(
            selected_path_count,
            child_output_width,
        )
        return probability_paths, connection_paths

    def __iter_selected_children(
        self,
        selected_child_indices: Tensor,
    ) -> Iterator[tuple[RoutingTreeNode, Tensor]]:
        for child_index, child in enumerate(self.branches):
            selected_positions = torch.nonzero(
                selected_child_indices == child_index,
                as_tuple=False,
            ).flatten()
            if selected_positions.numel() > 0:
                yield child, selected_positions

    def __route_selected_child(
        self,
        child: RoutingTreeNode,
        input_matrix: Tensor,
        direction_probabilities: Tensor,
        selected_positions: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        child_inputs = input_matrix.index_select(0, selected_positions)
        child_probabilities, child_connections, auxiliary_loss = child.route(
            child_inputs
        )
        selected_direction_probabilities = direction_probabilities.index_select(
            0,
            selected_positions,
        ).unsqueeze(1)
        joint_probabilities = selected_direction_probabilities * child_probabilities
        return joint_probabilities, child_connections, auxiliary_loss

    def __merge_child_paths(
        self,
        probability_paths: Tensor,
        connection_paths: Tensor,
        selected_positions: Tensor,
        joint_probabilities: Tensor,
        child_connections: Tensor,
    ) -> tuple[Tensor, Tensor]:
        common_dtype = torch.promote_types(
            probability_paths.dtype, joint_probabilities.dtype
        )
        probability_paths = probability_paths.to(dtype=common_dtype)
        joint_probabilities = joint_probabilities.to(dtype=common_dtype)
        probability_paths = probability_paths.index_copy(
            0,
            selected_positions,
            joint_probabilities,
        )
        connection_paths = connection_paths.index_copy(
            0,
            selected_positions,
            child_connections,
        )
        return probability_paths, connection_paths

    def __restore_batch_path_layout(
        self,
        probability_paths: Tensor,
        connection_paths: Tensor,
        batch_size: int,
    ) -> tuple[Tensor, Tensor]:
        probability_paths = probability_paths.reshape(
            batch_size,
            self.output_width,
        )
        connection_paths = connection_paths.reshape(
            batch_size,
            self.output_width,
        )
        return probability_paths, connection_paths

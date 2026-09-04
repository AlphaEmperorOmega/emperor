from __future__ import annotations

import copy
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
        self.input_dim = input_dim
        self.is_leaf = node_plan.is_leaf

        if self.is_leaf:
            node_sampler_config = RoutingTreeNode.derive_sampler_config(
                leaf_sampler_config,
                input_dim=input_dim,
                num_experts=len(node_plan.connection_indices),
                top_k=plan.leaf_top_k,
            )
            self.output_width = plan.leaf_top_k
            self.register_buffer(
                "global_connection_indices",
                torch.tensor(node_plan.connection_indices, dtype=torch.long),
                persistent=False,
            )
            self.branches = nn.ModuleList()
        else:
            direction_top_k = plan.direction_top_k[node_plan.level]
            node_sampler_config = RoutingTreeNode.derive_sampler_config(
                direction_sampler_config,
                input_dim=input_dim,
                num_experts=len(node_plan.children),
                top_k=direction_top_k,
            )
            self.branches = nn.ModuleList(
                RoutingTreeNode(
                    input_dim=input_dim,
                    plan=plan,
                    node_plan=child_plan,
                    leaf_sampler_config=leaf_sampler_config,
                    direction_sampler_config=direction_sampler_config,
                )
                for child_plan in node_plan.children
            )
            child_output_width = self.branches[0].output_width
            self.output_width = direction_top_k * child_output_width

        self.sampler = node_sampler_config.build()

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

    def route(self, input_matrix: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        probabilities, selected_indices, _, auxiliary_loss = (
            self.sampler.sample_probabilities_and_indices(input_matrix)
        )
        probability_matrix = _ensure_matrix(probabilities)
        index_matrix = _resolve_index_matrix(
            selected_indices,
            batch_size=input_matrix.shape[0],
            num_experts=self.sampler.num_experts,
            device=input_matrix.device,
        )
        if self.is_leaf:
            global_indices = self.global_connection_indices.to(index_matrix.device)[
                index_matrix
            ]
            return probability_matrix, global_indices, auxiliary_loss

        return self.__route_selected_children(
            input_matrix,
            probability_matrix,
            index_matrix,
            auxiliary_loss,
        )

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
        flattened_original_inputs = (
            input_matrix.unsqueeze(1)
            .expand(-1, selected_direction_count, -1)
            .reshape(-1, self.input_dim)
        )
        child_output_width = self.branches[0].output_width
        flattened_probability_paths = direction_probabilities.new_zeros(
            flattened_child_indices.shape[0],
            child_output_width,
        )
        flattened_connection_paths = selected_child_indices.new_zeros(
            flattened_child_indices.shape[0],
            child_output_width,
        )

        accumulated_auxiliary_loss = auxiliary_loss
        for child_index, child in enumerate(self.branches):
            selected_positions = torch.nonzero(
                flattened_child_indices == child_index,
                as_tuple=False,
            ).flatten()
            if selected_positions.numel() == 0:
                continue

            child_inputs = flattened_original_inputs.index_select(
                0,
                selected_positions,
            )
            child_probabilities, child_connections, child_auxiliary_loss = child.route(
                child_inputs
            )
            selected_direction_probabilities = (
                flattened_direction_probabilities.index_select(
                    0,
                    selected_positions,
                ).unsqueeze(1)
            )
            joint_probabilities = selected_direction_probabilities * child_probabilities
            flattened_probability_paths = flattened_probability_paths.index_copy(
                0,
                selected_positions,
                joint_probabilities,
            )
            flattened_connection_paths = flattened_connection_paths.index_copy(
                0,
                selected_positions,
                child_connections,
            )
            accumulated_auxiliary_loss = (
                accumulated_auxiliary_loss + child_auxiliary_loss
            )

        return (
            flattened_probability_paths.reshape(batch_size, self.output_width),
            flattened_connection_paths.reshape(batch_size, self.output_width),
            accumulated_auxiliary_loss,
        )


def _ensure_matrix(values: Tensor) -> Tensor:
    return values.unsqueeze(-1) if values.dim() == 1 else values


def _resolve_index_matrix(
    indices: Tensor | None,
    *,
    batch_size: int,
    num_experts: int,
    device: torch.device,
) -> Tensor:
    if indices is None:
        return torch.arange(
            num_experts,
            device=device,
            dtype=torch.long,
        ).expand(batch_size, -1)
    return _ensure_matrix(indices)

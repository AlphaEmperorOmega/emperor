from typing import TYPE_CHECKING

import torch
from torch import Tensor

from emperor.neuron._options import TerminalConnectionShapeOptions
from emperor.neuron._terminal.routing import TerminalRoutingTreeDelegate
from emperor.neuron._terminal.topology import initialize_terminal_connections
from emperor.neuron._terminal.validation import TerminalValidator
from emperor.nn import Module

if TYPE_CHECKING:
    from emperor.neuron._config import TerminalConfig


class Terminal(Module):
    VALIDATOR = TerminalValidator

    def __init__(
        self,
        cfg: "TerminalConfig",
        overrides: "TerminalConfig | None" = None,
    ):
        super().__init__()
        self.cfg: TerminalConfig = self._override_config(cfg, overrides)
        self.__initialize_from_config()
        self.VALIDATOR.validate(self)
        self.sampler = self.__build_sampler()

    def __initialize_from_config(self) -> None:
        self.VALIDATOR.validate_config_fields(self.cfg)

        self.input_dim: int = self.cfg.input_dim
        self.x_axis_position: int = self.cfg.x_axis_position
        self.y_axis_position: int = self.cfg.y_axis_position
        self.z_axis_position: int = self.cfg.z_axis_position
        self.xy_axis_range: int = self.cfg.xy_axis_range.value
        self.z_axis_range: int = self.cfg.z_axis_range.value
        self.connection_shape: TerminalConnectionShapeOptions = (
            self.cfg.connection_shape
        )
        self.sampler_config = self.cfg.sampler_config
        self.routing_tree_config = self.cfg.routing_tree_config
        neuron_connections = initialize_terminal_connections(self.cfg)
        self.total_neuron_connections = int(neuron_connections.shape[0])
        self.register_buffer("neuron_connections", neuron_connections, persistent=False)

    def __build_sampler(self):
        if self.routing_tree_config is not None:
            return TerminalRoutingTreeDelegate(
                cfg=self.cfg,
                neuron_connections=self.neuron_connections,
            )
        if self.sampler_config.router_config is None:
            return self.sampler_config.build()
        return self.sampler_config.build_with_router_input_dim(self.input_dim)

    def forward(self, input: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        self.VALIDATOR.validate_forward_input(self, input)
        probabilities, selected_connection_indices, _, auxiliary_loss = (
            self.sampler.sample_probabilities_and_indices(input)
        )
        probabilities = self.__ensure_probability_matrix(probabilities)
        selected_connection_indices = self.__resolve_selected_indices(
            input, selected_connection_indices
        )
        selected_connection_indices = self.__ensure_index_matrix(
            selected_connection_indices
        )
        device_aligned_neuron_connections = self.neuron_connections.to(
            selected_connection_indices.device
        )
        selected_neurons = device_aligned_neuron_connections[
            selected_connection_indices
        ]
        return (input, probabilities, selected_neurons, auxiliary_loss)

    def __ensure_probability_matrix(self, probabilities: Tensor) -> Tensor:
        if probabilities.dim() == 1:
            return probabilities.unsqueeze(-1)
        return probabilities

    def __resolve_selected_indices(
        self, input: Tensor, indices: Tensor | None
    ) -> Tensor:
        if indices is not None:
            return indices
        all_connection_indices = torch.arange(
            self.total_neuron_connections,
            device=input.device,
            dtype=torch.long,
        )
        batch_size = input.shape[0]
        batched_connection_indices = all_connection_indices.expand(batch_size, -1)
        return batched_connection_indices

    def __ensure_index_matrix(self, indices: Tensor) -> Tensor:
        if indices.dim() == 1:
            return indices.unsqueeze(-1)
        return indices

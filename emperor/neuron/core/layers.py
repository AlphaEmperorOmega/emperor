import torch

from torch import Tensor

from emperor.base.utils import Module
from emperor.neuron.core._validator import (
    AxonsValidator,
    NeuronValidator,
    NucleusValidator,
    TerminalValidator,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.neuron.core.config import (
        AxonsConfig,
        NeuronConfig,
        NucleusConfig,
        TerminalConfig,
    )


class Nucleus(Module):
    def __init__(
        self,
        cfg: "NucleusConfig",
        overrides: "NucleusConfig | None" = None,
    ):
        super().__init__()
        config = getattr(cfg, "nucleus_config", cfg)
        self.cfg: "NucleusConfig" = self._override_config(config, overrides)
        self.model_config = self.cfg.model_config
        NucleusValidator.validate(self)
        self.model = self.model_config.build()

    def forward(self, input: Tensor) -> Tensor:
        NucleusValidator.validate_forward_input(input)
        return self.model(input)


class Axons(Module):
    def __init__(
        self,
        cfg: "AxonsConfig",
        overrides: "AxonsConfig | None" = None,
    ):
        super().__init__()
        config = getattr(cfg, "axons_config", cfg)
        self.cfg: "AxonsConfig" = self._override_config(config, overrides)
        self.memory_config = self.cfg.memory_config
        AxonsValidator.validate(self)
        self.memory_model = self.__maybe_build_memory_model()

    def __maybe_build_memory_model(self) -> Module | None:
        if self.memory_config is None:
            return None
        return self.memory_config.build()

    def forward(self, input: Tensor) -> Tensor:
        AxonsValidator.validate_forward_input(input)
        if self.memory_model is None:
            return input
        return self.memory_model(input)


class Terminal(Module):
    def __init__(
        self,
        cfg: "TerminalConfig",
        overrides: "TerminalConfig | None" = None,
    ):
        super().__init__()
        config = getattr(cfg, "terminal_config", cfg)
        self.cfg: "TerminalConfig" = self._override_config(config, overrides)
        TerminalValidator.validate_config_fields(self.cfg)

        self.input_dim: int = self.cfg.input_dim
        self.x_axis_position: int = self.cfg.x_axis_position
        self.y_axis_position: int = self.cfg.y_axis_position
        self.z_axis_position: int = self.cfg.z_axis_position
        self.xy_axis_range: int = self.cfg.xy_axis_range.value
        self.z_axis_range: int = self.cfg.z_axis_range.value
        self.z_axis_offset: int = self.cfg.z_axis_offset.value
        self.sampler_config = self.cfg.sampler_config
        self.total_neuron_connections = self.__compute_total_neuron_connections()

        TerminalValidator.validate(self)
        self.sampler = self.__build_sampler()
        self.register_buffer(
            "neuron_connections",
            self.__initialize_connections(),
            persistent=False,
        )

    def __build_sampler(self):
        if self.sampler_config.router_config is None:
            return self.sampler_config.build()
        return self.sampler_config.build_with_router_input_dim(self.input_dim)

    def __compute_total_neuron_connections(self) -> int:
        single_axis_range = self.xy_axis_range * 2 + 1
        return single_axis_range**2 * (self.z_axis_range + 1)

    def __initialize_connections(self) -> Tensor:
        x_axis_range_indices = self.__compute_xy_axis_range()
        y_axis_range_indices = self.__compute_xy_axis_range(is_y_axis_flag=True)
        z_axis_range_indices = self.__compute_z_axis_range()
        return torch.cartesian_prod(
            x_axis_range_indices,
            y_axis_range_indices,
            z_axis_range_indices,
        )

    def __compute_xy_axis_range(self, is_y_axis_flag: bool = False) -> Tensor:
        position = self.y_axis_position if is_y_axis_flag else self.x_axis_position
        range_start = position - self.xy_axis_range
        range_end = position + self.xy_axis_range + 1
        return torch.arange(range_start, range_end)

    def __compute_z_axis_range(self) -> Tensor:
        range_start = self.z_axis_position - self.z_axis_offset
        range_end = self.z_axis_position + self.z_axis_range - self.z_axis_offset + 1
        return torch.arange(range_start, range_end)

    def forward(self, input: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        TerminalValidator.validate_forward_input(self, input)
        probabilities, indices, _, auxiliary_loss = (
            self.sampler.sample_probabilities_and_indices(input)
        )
        probabilities = self.__ensure_probability_matrix(probabilities)
        indices = self.__resolve_selected_indices(input, indices)
        indices = self.__ensure_index_matrix(indices)
        selected_neurons = self.neuron_connections.to(indices.device)[indices]
        return input, probabilities, selected_neurons, auxiliary_loss

    def __ensure_probability_matrix(self, probabilities: Tensor) -> Tensor:
        if probabilities.dim() == 1:
            return probabilities.unsqueeze(-1)
        return probabilities

    def __resolve_selected_indices(self, input: Tensor, indices: Tensor | None) -> Tensor:
        if indices is not None:
            return indices
        return torch.arange(
            self.total_neuron_connections,
            device=input.device,
            dtype=torch.long,
        ).expand(input.shape[0], -1)

    def __ensure_index_matrix(self, indices: Tensor) -> Tensor:
        if indices.dim() == 1:
            return indices.unsqueeze(-1)
        return indices


class Neuron(Module):
    def __init__(
        self,
        cfg: "NeuronConfig",
        overrides: "NeuronConfig | None" = None,
    ):
        super().__init__()
        config = getattr(cfg, "neuron_config", cfg)
        self.cfg: "NeuronConfig" = self._override_config(config, overrides)
        NeuronValidator.validate(self.cfg)
        self.nucleus = self.cfg.nucleus_config.build()
        self.axons = self.cfg.axons_config.build()
        self.terminal = self.cfg.terminal_config.build()
        self.register_buffer(
            "batch_counter",
            torch.tensor(0, dtype=torch.int64),
            persistent=True,
        )

    def process_signal(self, input: Tensor) -> Tensor:
        NeuronValidator.validate_forward_input(input)
        if self.training:
            self.batch_counter += 1
        processed_signal = self.nucleus(input)
        return self.axons(processed_signal)

    def route_signal(self, processed_signal: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        NeuronValidator.validate_forward_input(processed_signal)
        _, probabilities, selected_neurons, auxiliary_loss = self.terminal(
            processed_signal
        )
        return probabilities, selected_neurons, auxiliary_loss

    def forward(self, input: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        processed_signal = self.process_signal(input)
        probabilities, selected_neurons, auxiliary_loss = self.route_signal(
            processed_signal
        )
        return processed_signal, probabilities, selected_neurons, auxiliary_loss

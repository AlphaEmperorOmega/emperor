from typing import TYPE_CHECKING

import torch
from torch import Tensor

from emperor.neuron._options import TerminalConnectionShapeOptions
from emperor.neuron._terminal_capture import (
    TerminalRoute,
    publish_terminal_route,
    run_scored_terminal_forward,
)
from emperor.neuron._terminal_topology import initialize_terminal_connections
from emperor.neuron._validation import (
    AxonsValidator,
    NeuronValidator,
    NucleusValidator,
    TerminalValidator,
)
from emperor.nn import Module

if TYPE_CHECKING:
    from emperor.neuron._config import (
        AxonsConfig,
        NeuronConfig,
        NucleusConfig,
        TerminalConfig,
    )


class Nucleus(Module):
    VALIDATOR = NucleusValidator

    def __init__(
        self,
        cfg: "NucleusConfig",
        overrides: "NucleusConfig | None" = None,
    ):
        super().__init__()
        nucleus_config = getattr(cfg, "nucleus_config", cfg)
        self.cfg: NucleusConfig = self._override_config(nucleus_config, overrides)
        self.model_config = self.cfg.model_config
        self.VALIDATOR.validate(self)
        self.model = self.model_config.build()

    def forward(self, input: Tensor) -> Tensor:
        self.VALIDATOR.validate_forward_input(input)
        return self.model(input)


class Axons(Module):
    VALIDATOR = AxonsValidator

    def __init__(
        self,
        cfg: "AxonsConfig",
        overrides: "AxonsConfig | None" = None,
    ):
        super().__init__()
        axons_config = getattr(cfg, "axons_config", cfg)
        self.cfg: AxonsConfig = self._override_config(axons_config, overrides)
        self.memory_config = self.cfg.memory_config
        self.VALIDATOR.validate(self)
        self.memory_model = self.__maybe_build_memory_model()

    def __maybe_build_memory_model(self) -> Module | None:
        if self.memory_config is None:
            return None
        return self._build_from_config(
            self.memory_config,
            input_dim=self.memory_config.input_dim,
            output_dim=self.memory_config.input_dim,
        )

    def forward(self, input: Tensor) -> Tensor:
        self.VALIDATOR.validate_forward_input(input)
        if self.memory_model is None:
            return input
        return self.memory_model(input)


class Terminal(Module):
    VALIDATOR = TerminalValidator

    def __init__(
        self,
        cfg: "TerminalConfig",
        overrides: "TerminalConfig | None" = None,
    ):
        super().__init__()
        terminal_config = getattr(cfg, "terminal_config", cfg)
        self.cfg: TerminalConfig = self._override_config(terminal_config, overrides)
        neuron_connections = self.__initialize_configuration()
        self.VALIDATOR.validate(self)
        self.sampler = self.__build_sampler()
        self.register_buffer(
            "neuron_connections",
            neuron_connections,
            persistent=False,
        )

    def __initialize_configuration(self) -> Tensor:
        self.VALIDATOR.validate_config_fields(self.cfg)

        self.input_dim: int = self.cfg.input_dim
        self.x_axis_position: int = self.cfg.x_axis_position
        self.y_axis_position: int = self.cfg.y_axis_position
        self.z_axis_position: int = self.cfg.z_axis_position
        self.xy_axis_range: int = self.cfg.xy_axis_range.value
        self.z_axis_range: int = self.cfg.z_axis_range.value
        self.z_axis_offset: int = self.cfg.z_axis_offset.value
        self.connection_shape: TerminalConnectionShapeOptions = (
            TerminalConnectionShapeOptions.BOX
            if self.cfg.connection_shape is None
            else self.cfg.connection_shape
        )
        self.sampler_config = self.cfg.sampler_config
        neuron_connections = initialize_terminal_connections(self.cfg)
        self.total_neuron_connections = int(neuron_connections.shape[0])
        return neuron_connections

    def __build_sampler(self):
        if self.sampler_config.router_config is None:
            return self.sampler_config.build()
        return self.sampler_config.build_with_router_input_dim(self.input_dim)

    def forward(self, input: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        routed_signal = self.__compute_route(input)
        publish_terminal_route(self, routed_signal)
        routed_input, probabilities, _, _, selected_neurons, auxiliary_loss = (
            routed_signal
        )
        return routed_input, probabilities, selected_neurons, auxiliary_loss

    def _forward_with_log_probabilities(
        self,
        input: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        (
            routed_input,
            probabilities,
            log_probabilities,
            _,
            selected_neurons,
            auxiliary_loss,
        ) = self._forward_with_router_scores(input)
        return (
            routed_input,
            probabilities,
            log_probabilities,
            selected_neurons,
            auxiliary_loss,
        )

    def _forward_with_router_scores(
        self,
        input: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        return run_scored_terminal_forward(self, input)

    def __compute_route(self, input: Tensor) -> TerminalRoute:
        self.VALIDATOR.validate_forward_input(self, input)
        if hasattr(
            self.sampler,
            "sample_probabilities_log_scores_router_scores_and_indices",
        ):
            (
                probabilities,
                log_probabilities,
                router_scores,
                selected_connection_indices,
                _,
                auxiliary_loss,
            ) = self.sampler.sample_probabilities_log_scores_router_scores_and_indices(
                input
            )
        else:
            (
                probabilities,
                log_probabilities,
                selected_connection_indices,
                _,
                auxiliary_loss,
            ) = self.sampler.sample_probabilities_log_scores_and_indices(input)
            router_scores = log_probabilities
        probabilities = self.__ensure_probability_matrix(probabilities)
        log_probabilities = self.__ensure_probability_matrix(log_probabilities)
        router_scores = self.__ensure_probability_matrix(router_scores)
        selected_connection_indices = self.__resolve_selected_indices(
            input,
            selected_connection_indices,
        )
        selected_connection_indices = self.__ensure_index_matrix(
            selected_connection_indices
        )
        selected_neurons = self.neuron_connections.to(
            selected_connection_indices.device
        )[selected_connection_indices]
        return (
            input,
            probabilities,
            log_probabilities,
            router_scores,
            selected_neurons,
            auxiliary_loss,
        )

    def __ensure_probability_matrix(self, probabilities: Tensor) -> Tensor:
        if probabilities.dim() == 1:
            return probabilities.unsqueeze(-1)
        return probabilities

    def __resolve_selected_indices(
        self, input: Tensor, indices: Tensor | None
    ) -> Tensor:
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
    COORDINATE_EMBEDDING_FREQUENCY_BASE = 10000.0
    VALIDATOR = NeuronValidator

    def __init__(
        self,
        cfg: "NeuronConfig",
        overrides: "NeuronConfig | None" = None,
    ):
        super().__init__()
        neuron_config = getattr(cfg, "neuron_config", cfg)
        self.cfg: NeuronConfig = self._override_config(neuron_config, overrides)
        self.VALIDATOR.validate(self.cfg)
        self.coordinate_embedding_flag: bool = bool(self.cfg.coordinate_embedding_flag)
        self.nucleus = self.cfg.nucleus_config.build()
        self.axons = self.cfg.axons_config.build()
        self.terminal = self.cfg.terminal_config.build()
        self.register_buffer(
            "batch_counter",
            torch.tensor(0, dtype=torch.int64),
            persistent=True,
        )
        self.register_buffer(
            "atrophy_counter",
            torch.tensor(0, dtype=torch.int64),
            persistent=True,
        )
        if self.coordinate_embedding_flag:
            self.register_buffer(
                "coordinate_embedding",
                self.__initialize_coordinate_embedding(),
                persistent=False,
            )
        else:
            self.coordinate_embedding = None

    def __initialize_coordinate_embedding(self) -> Tensor:
        axis_positions = (
            self.terminal.x_axis_position,
            self.terminal.y_axis_position,
            self.terminal.z_axis_position,
        )
        axis_channel_counts = self.__split_channels_across_axes(self.terminal.input_dim)
        axis_encodings = [
            self.__sinusoidal_axis_encoding(axis_position, axis_channel_count)
            for axis_position, axis_channel_count in zip(
                axis_positions,
                axis_channel_counts,
                strict=True,
            )
        ]
        return torch.cat(axis_encodings)

    def __split_channels_across_axes(self, input_dim: int) -> tuple[int, int, int]:
        base_channel_count = input_dim // 3
        remainder_channel_count = input_dim % 3
        return tuple(
            base_channel_count + (1 if axis_index < remainder_channel_count else 0)
            for axis_index in range(3)
        )

    def __sinusoidal_axis_encoding(
        self,
        axis_position: int,
        axis_channel_count: int,
    ) -> Tensor:
        channel_indices = torch.arange(axis_channel_count, dtype=torch.float32)
        frequency_exponents = (channel_indices - channel_indices % 2) / float(
            axis_channel_count
        )
        sinusoidal_angles = float(axis_position) / torch.pow(
            torch.tensor(self.COORDINATE_EMBEDDING_FREQUENCY_BASE),
            frequency_exponents,
        )
        return torch.where(
            channel_indices % 2 == 0,
            torch.sin(sinusoidal_angles),
            torch.cos(sinusoidal_angles),
        )

    def __inject_coordinate_embedding(self, input: Tensor) -> Tensor:
        if self.coordinate_embedding is None:
            return input
        return input + self.coordinate_embedding.to(
            device=input.device,
            dtype=input.dtype,
        )

    def process_signal(self, input: Tensor) -> Tensor:
        self.VALIDATOR.validate_forward_input(input)
        if self.training:
            self.batch_counter += 1
        processed_signal = self.nucleus(self.__inject_coordinate_embedding(input))
        return self.axons(processed_signal)

    def route_signal(self, processed_signal: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        probabilities, _, selected_neurons, auxiliary_loss = (
            self._route_signal_with_log_probabilities(processed_signal)
        )
        return probabilities, selected_neurons, auxiliary_loss

    def _route_signal_with_log_probabilities(
        self,
        processed_signal: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        (
            probabilities,
            log_probabilities,
            _,
            selected_neurons,
            auxiliary_loss,
        ) = self._route_signal_with_router_scores(processed_signal)
        return probabilities, log_probabilities, selected_neurons, auxiliary_loss

    def _route_signal_with_router_scores(
        self,
        processed_signal: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        self.VALIDATOR.validate_forward_input(processed_signal)
        (
            _,
            probabilities,
            log_probabilities,
            router_scores,
            selected_neurons,
            auxiliary_loss,
        ) = self.terminal._forward_with_router_scores(
            self.__inject_coordinate_embedding(processed_signal)
        )
        return (
            probabilities,
            log_probabilities,
            router_scores,
            selected_neurons,
            auxiliary_loss,
        )

    def forward(self, input: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        processed_signal = self.process_signal(input)
        probabilities, selected_neurons, auxiliary_loss = self.route_signal(
            processed_signal
        )
        return processed_signal, probabilities, selected_neurons, auxiliary_loss

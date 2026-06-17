import torch

from torch import Tensor

from emperor.base.utils import Module
from emperor.neuron.core.options import TerminalConnectionShapeOptions
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
        return self._build_from_config(
            self.memory_config,
            input_dim=self.memory_config.input_dim,
            output_dim=self.memory_config.input_dim,
        )

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
        self.connection_shape: TerminalConnectionShapeOptions = (
            TerminalConnectionShapeOptions.BOX
            if self.cfg.connection_shape is None
            else self.cfg.connection_shape
        )
        self.sampler_config = self.cfg.sampler_config
        neuron_connections = self.__initialize_connections()
        self.total_neuron_connections = int(neuron_connections.shape[0])

        TerminalValidator.validate(self)
        self.sampler = self.__build_sampler()
        self.register_buffer(
            "neuron_connections",
            neuron_connections,
            persistent=False,
        )

    def __build_sampler(self):
        if self.sampler_config.router_config is None:
            return self.sampler_config.build()
        return self.sampler_config.build_with_router_input_dim(self.input_dim)

    def __initialize_connections(self) -> Tensor:
        if self.connection_shape is TerminalConnectionShapeOptions.BOX:
            return torch.cartesian_prod(
                self.__compute_xy_axis_range(),
                self.__compute_xy_axis_range(is_y_axis_flag=True),
                self.__compute_z_axis_range(),
            )
        return self.__initialize_shaped_connections()

    def __compute_xy_axis_range(self, is_y_axis_flag: bool = False) -> Tensor:
        position = self.y_axis_position if is_y_axis_flag else self.x_axis_position
        range_start = position - self.xy_axis_range
        range_end = position + self.xy_axis_range + 1
        return torch.arange(range_start, range_end)

    def __compute_z_axis_range(self) -> Tensor:
        range_start = self.z_axis_position - self.z_axis_offset
        range_end = self.z_axis_position + self.z_axis_range - self.z_axis_offset + 1
        return torch.arange(range_start, range_end)

    def __initialize_shaped_connections(self) -> Tensor:
        deduplicated_offsets = list(
            dict.fromkeys(self.__connection_offsets_for_shape())
        )
        position = torch.tensor(
            [self.x_axis_position, self.y_axis_position, self.z_axis_position],
            dtype=torch.long,
        )
        return position + torch.tensor(deduplicated_offsets, dtype=torch.long)

    def __connection_offsets_for_shape(self) -> list[tuple[int, int, int]]:
        shape = self.connection_shape
        if shape is TerminalConnectionShapeOptions.CROSS:
            return (
                self.__x_axis_line_offsets()
                + self.__y_axis_line_offsets()
                + self.__z_axis_line_offsets()
            )
        if shape is TerminalConnectionShapeOptions.SPHERE:
            return self.__ellipsoid_offsets()
        if shape is TerminalConnectionShapeOptions.DIAGONAL_X:
            return self.__xy_diagonal_offsets()
        if shape is TerminalConnectionShapeOptions.LINE_LEFT_RIGHT:
            return self.__x_axis_line_offsets()
        if shape is TerminalConnectionShapeOptions.LINE_UP_DOWN:
            return self.__y_axis_line_offsets()
        if shape is TerminalConnectionShapeOptions.LINE_FRONT_BACK:
            return self.__z_axis_line_offsets()
        raise ValueError(f"Unsupported connection_shape {shape!r} for Terminal.")

    def __x_axis_line_offsets(self) -> list[tuple[int, int, int]]:
        return [
            (delta, 0, 0)
            for delta in range(-self.xy_axis_range, self.xy_axis_range + 1)
        ]

    def __y_axis_line_offsets(self) -> list[tuple[int, int, int]]:
        return [
            (0, delta, 0)
            for delta in range(-self.xy_axis_range, self.xy_axis_range + 1)
        ]

    def __z_axis_line_offsets(self) -> list[tuple[int, int, int]]:
        return [
            (0, 0, delta)
            for delta in range(
                -self.z_axis_offset,
                self.z_axis_range - self.z_axis_offset + 1,
            )
        ]

    def __xy_diagonal_offsets(self) -> list[tuple[int, int, int]]:
        offsets = []
        for delta in range(-self.xy_axis_range, self.xy_axis_range + 1):
            offsets.append((delta, delta, 0))
            offsets.append((delta, -delta, 0))
        return offsets

    def __ellipsoid_offsets(self) -> list[tuple[int, int, int]]:
        # The z window is directional (offset back, range forward), so the
        # ellipsoid is centered on the window rather than on the neuron.
        z_window_center = (self.z_axis_range - 2 * self.z_axis_offset) / 2
        z_half_extent = self.z_axis_range / 2
        offsets = []
        for x_delta in range(-self.xy_axis_range, self.xy_axis_range + 1):
            for y_delta in range(-self.xy_axis_range, self.xy_axis_range + 1):
                for z_delta in range(
                    -self.z_axis_offset,
                    self.z_axis_range - self.z_axis_offset + 1,
                ):
                    normalized_distance = (
                        (x_delta / self.xy_axis_range) ** 2
                        + (y_delta / self.xy_axis_range) ** 2
                        + ((z_delta - z_window_center) / z_half_extent) ** 2
                    )
                    if normalized_distance <= 1.0 + 1e-9:
                        offsets.append((x_delta, y_delta, z_delta))
        return offsets

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
    COORDINATE_EMBEDDING_FREQUENCY_BASE = 10000.0

    def __init__(
        self,
        cfg: "NeuronConfig",
        overrides: "NeuronConfig | None" = None,
    ):
        super().__init__()
        config = getattr(cfg, "neuron_config", cfg)
        self.cfg: "NeuronConfig" = self._override_config(config, overrides)
        NeuronValidator.validate(self.cfg)
        self.coordinate_embedding_flag: bool = bool(
            self.cfg.coordinate_embedding_flag
        )
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
        axis_channel_counts = self.__split_channels_across_axes(
            self.terminal.input_dim
        )
        axis_encodings = [
            self.__sinusoidal_axis_encoding(axis_position, axis_channel_count)
            for axis_position, axis_channel_count in zip(
                axis_positions,
                axis_channel_counts,
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
        angles = float(axis_position) / torch.pow(
            torch.tensor(self.COORDINATE_EMBEDDING_FREQUENCY_BASE),
            frequency_exponents,
        )
        return torch.where(
            channel_indices % 2 == 0,
            torch.sin(angles),
            torch.cos(angles),
        )

    def __inject_coordinate_embedding(self, input: Tensor) -> Tensor:
        if self.coordinate_embedding is None:
            return input
        return input + self.coordinate_embedding.to(
            device=input.device,
            dtype=input.dtype,
        )

    def process_signal(self, input: Tensor) -> Tensor:
        NeuronValidator.validate_forward_input(input)
        if self.training:
            self.batch_counter += 1
        processed_signal = self.nucleus(self.__inject_coordinate_embedding(input))
        return self.axons(processed_signal)

    def route_signal(self, processed_signal: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        NeuronValidator.validate_forward_input(processed_signal)
        _, probabilities, selected_neurons, auxiliary_loss = self.terminal(
            self.__inject_coordinate_embedding(processed_signal)
        )
        return probabilities, selected_neurons, auxiliary_loss

    def forward(self, input: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        processed_signal = self.process_signal(input)
        probabilities, selected_neurons, auxiliary_loss = self.route_signal(
            processed_signal
        )
        return processed_signal, probabilities, selected_neurons, auxiliary_loss

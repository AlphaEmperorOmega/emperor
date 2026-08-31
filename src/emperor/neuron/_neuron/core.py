from typing import TYPE_CHECKING

import torch
from torch import Tensor

from emperor.neuron._neuron.validation import NeuronValidator
from emperor.nn import Module

if TYPE_CHECKING:
    from emperor.neuron._config import NeuronConfig


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
        self.VALIDATOR.validate_forward_input(processed_signal)
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

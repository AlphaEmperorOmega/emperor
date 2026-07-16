import copy
from dataclasses import fields
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch.nn import ModuleDict

from emperor.config import ConfigBase
from emperor.neuron._cluster.beam_routes import _NeuronClusterBeamRoutesMixin
from emperor.neuron._cluster.checkpointing import _NeuronClusterCheckpointingMixin
from emperor.neuron._cluster.plasticity import _NeuronClusterPlasticityMixin
from emperor.neuron._cluster.recurrent_routes import (
    _NeuronClusterRecurrentRoutesMixin,
)
from emperor.neuron._cluster.state import _NeuronClusterStateMixin
from emperor.neuron._cluster.topology import _NeuronClusterTopologyMixin
from emperor.neuron._config import NeuronClusterConfig, TerminalConfig
from emperor.neuron._trace import NeuronClusterTrace
from emperor.neuron._validation import NeuronClusterValidator
from emperor.nn import Module

if TYPE_CHECKING:
    from emperor.halting import HaltingBase


class NeuronCluster(
    _NeuronClusterTopologyMixin,
    _NeuronClusterStateMixin,
    _NeuronClusterRecurrentRoutesMixin,
    _NeuronClusterBeamRoutesMixin,
    _NeuronClusterPlasticityMixin,
    _NeuronClusterCheckpointingMixin,
    Module,
):
    VALIDATOR = NeuronClusterValidator

    def __init__(
        self,
        cfg: NeuronClusterConfig,
        overrides: NeuronClusterConfig | None = None,
    ):
        super().__init__()
        self.cfg: NeuronClusterConfig = self._override_config(cfg, overrides)
        self.VALIDATOR.validate(self)

        self.x_axis_total_neurons: int = self.cfg.x_axis_total_neurons
        self.y_axis_total_neurons: int = self.cfg.y_axis_total_neurons
        self.z_axis_total_neurons: int = self.cfg.z_axis_total_neurons
        self.initial_x_axis_total_neurons: int = self.__resolve_initial_dimension(
            self.cfg.initial_x_axis_total_neurons,
            self.x_axis_total_neurons,
        )
        self.initial_y_axis_total_neurons: int = self.__resolve_initial_dimension(
            self.cfg.initial_y_axis_total_neurons,
            self.y_axis_total_neurons,
        )
        self.initial_z_axis_total_neurons: int = self.__resolve_initial_dimension(
            self.cfg.initial_z_axis_total_neurons,
            self.z_axis_total_neurons,
        )
        self.initial_x_axis_start: int = self.__resolve_initial_axis_start(
            self.initial_x_axis_total_neurons,
            self.x_axis_total_neurons,
        )
        self.initial_y_axis_start: int = self.__resolve_initial_axis_start(
            self.initial_y_axis_total_neurons,
            self.y_axis_total_neurons,
        )
        self.initial_z_axis_start: int = self.__resolve_initial_axis_start(
            self.initial_z_axis_total_neurons,
            self.z_axis_total_neurons,
        )
        self.max_steps: int = self.cfg.max_steps
        self.beam_width: int = 1 if self.cfg.beam_width is None else self.cfg.beam_width
        self.growth_threshold: int | None = self.cfg.growth_threshold
        self.growth_cooldown_steps: int | None = self.cfg.growth_cooldown_steps
        self.max_total_growths: int | None = self.cfg.max_total_growths
        self.growth_warmup_steps: int | None = self.cfg.growth_warmup_steps
        self.pruning_threshold: int | None = self.cfg.pruning_threshold
        self.escape_driven_growth_flag: bool = bool(self.cfg.escape_driven_growth_flag)
        self.mitosis_initialization_flag: bool = bool(
            self.cfg.mitosis_initialization_flag
        )
        self.halting_config = self.cfg.halting_config
        self.input_dim: int = self.cfg.neuron_config.terminal_config.input_dim

        self.register_buffer(
            "entry_coordinates",
            self.__initialize_entry_coordinates(),
            persistent=False,
        )
        if self.escape_driven_growth_flag:
            self.register_buffer(
                "escape_counts",
                torch.zeros(
                    self.x_axis_total_neurons,
                    self.y_axis_total_neurons,
                    self.z_axis_total_neurons,
                    dtype=torch.long,
                ),
                persistent=True,
            )
        else:
            self.escape_counts = None
        if self.growth_cooldown_steps is not None:
            self.register_buffer(
                "forwards_since_last_growth",
                torch.zeros((), dtype=torch.long),
                persistent=True,
            )
        else:
            self.forwards_since_last_growth = None
        if self.max_total_growths is not None:
            self.register_buffer(
                "total_growth_count",
                torch.zeros((), dtype=torch.long),
                persistent=True,
            )
        else:
            self.total_growth_count = None
        self.entry_sampler_config = self.__resolve_entry_sampler_config()
        self.cluster = self.__initialize_cluster()
        self.entry_sampler = self.__build_entry_sampler()
        self.halting_model = self.__build_halting_model()
        self.register_load_state_dict_pre_hook(self._reconcile_cluster_with_state_dict)

    def __resolve_initial_dimension(
        self,
        configured_value: int | None,
        maximum_value: int,
    ) -> int:
        return maximum_value if configured_value is None else configured_value

    def __resolve_initial_axis_start(
        self,
        initial_value: int,
        maximum_value: int,
    ) -> int:
        return ((maximum_value - initial_value) // 2) + 1

    def __initialize_entry_coordinates(self) -> Tensor:
        x_indices = torch.arange(
            self.initial_x_axis_start,
            self.initial_x_axis_start + self.initial_x_axis_total_neurons,
            dtype=torch.long,
        )
        y_indices = torch.arange(
            self.initial_y_axis_start,
            self.initial_y_axis_start + self.initial_y_axis_total_neurons,
            dtype=torch.long,
        )
        z_indices = torch.tensor([self.initial_z_axis_start], dtype=torch.long)
        return torch.cartesian_prod(x_indices, y_indices, z_indices)

    def __resolve_entry_sampler_config(self):
        if self.cfg.entry_sampler_config is not None:
            return copy.deepcopy(self.cfg.entry_sampler_config)

        entry_count = int(self.entry_coordinates.shape[0])
        sampler_config = copy.deepcopy(
            self.cfg.neuron_config.terminal_config.sampler_config
        )
        sampler_config.num_experts = entry_count
        sampler_config.top_k = min(sampler_config.top_k, entry_count)
        sampler_config.num_topk_samples = min(
            sampler_config.num_topk_samples,
            sampler_config.top_k,
        )
        if sampler_config.top_k == 1:
            sampler_config.normalize_probabilities_flag = False
        if sampler_config.router_config is not None:
            sampler_config.router_config.num_experts = entry_count
        return sampler_config

    def __build_entry_sampler(self):
        if self.entry_sampler_config.router_config is None:
            return self.entry_sampler_config.build()
        return self.entry_sampler_config.build_with_router_input_dim(self.input_dim)

    def __build_halting_model(self) -> "HaltingBase | None":
        return self.__build_from_config(self.halting_config, input_dim=self.input_dim)

    def __build_from_config(
        self,
        config: "ConfigBase | None",
        **kwargs,
    ) -> "Module | None":
        if config is None:
            return None
        declared_fields = {field.name for field in fields(config)}
        overrides = type(config)(
            **{name: value for name, value in kwargs.items() if name in declared_fields}
        )
        return config.build(overrides=overrides)

    def __initialize_cluster(self) -> ModuleDict:
        cluster = ModuleDict()
        for x_coordinate in range(
            self.initial_x_axis_start,
            self.initial_x_axis_start + self.initial_x_axis_total_neurons,
        ):
            for y_coordinate in range(
                self.initial_y_axis_start,
                self.initial_y_axis_start + self.initial_y_axis_total_neurons,
            ):
                for z_coordinate in range(
                    self.initial_z_axis_start,
                    self.initial_z_axis_start + self.initial_z_axis_total_neurons,
                ):
                    name = self._neuron_name(
                        x_coordinate,
                        y_coordinate,
                        z_coordinate,
                    )
                    self._add_neuron(
                        cluster,
                        name,
                        self._initialize_neuron(
                            x_coordinate,
                            y_coordinate,
                            z_coordinate,
                        ),
                    )
        return cluster

    def _initialize_neuron(self, x: int, y: int, z: int) -> Module:
        neuron_config = copy.deepcopy(self.cfg.neuron_config)
        terminal_config = neuron_config.terminal_config
        terminal_overrides = TerminalConfig(
            x_axis_position=x,
            y_axis_position=y,
            z_axis_position=z,
        )
        neuron_config.terminal_config = self._override_config(
            terminal_config,
            terminal_overrides,
        )
        return self.__move_to_current_context(neuron_config.build())

    def __move_to_current_context(self, module: Module) -> Module:
        device, dtype = self.__current_device_and_dtype()
        if dtype is None:
            return module.to(device=device)
        return module.to(device=device, dtype=dtype)

    def __current_device_and_dtype(
        self,
    ) -> tuple[torch.device, torch.dtype | None]:
        fallback_device = None
        for parameter in self.parameters():
            fallback_device = parameter.device
            if parameter.is_floating_point() or parameter.is_complex():
                return parameter.device, parameter.dtype

        for buffer in self.buffers():
            if fallback_device is None:
                fallback_device = buffer.device
            if buffer.is_floating_point() or buffer.is_complex():
                return buffer.device, buffer.dtype

        return fallback_device or torch.device("cpu"), None

    def forward(
        self,
        input: Tensor,
        return_trace: bool = False,
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, NeuronClusterTrace]:
        self.VALIDATOR.validate_forward_input(input)
        self.__validate_feature_dimension(input)
        if return_trace and self.beam_width > 1:
            raise NotImplementedError(
                "return_trace is not supported when beam_width > 1; route "
                "traces describe a single chosen branch per sample and beams "
                "have no such branch."
            )
        self._neurons_called_this_forward: set[str] = set()

        flat_input = input.reshape(-1, input.shape[-1])
        output, auxiliary_loss, trace = self._propagate_signal_through_recurrent_routes(
            flat_input,
            tuple(input.shape),
            return_trace,
        )
        output = output.reshape(*input.shape[:-1], output.shape[-1])

        if self.training:
            # Warmup advances before growth so a neuron grown this forward
            # keeps its full countdown for its first routable forward.
            self._advance_grown_neuron_warmup()
            self._check_neuron_growth()
            self._check_neuron_atrophy()
        if return_trace:
            return output, auxiliary_loss, trace
        return output, auxiliary_loss

    def __validate_feature_dimension(self, input: Tensor) -> None:
        if input.shape[-1] != self.input_dim:
            raise ValueError(
                "NeuronCluster input feature dimension must match "
                "neuron_config.terminal_config.input_dim, received "
                f"input_dim={self.input_dim} and input shape {tuple(input.shape)}."
            )

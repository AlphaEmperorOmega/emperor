from types import SimpleNamespace
from typing import TYPE_CHECKING

from torch import Tensor

from emperor._validation import ValidatorBase
from emperor.neuron._validation.common import NeuronValidationMixin

if TYPE_CHECKING:
    from emperor.neuron._terminal.core import Terminal


class TerminalValidator(ValidatorBase, NeuronValidationMixin):
    OPTIONAL_FIELDS = set()

    @classmethod
    def validate_config_fields(cls, cfg) -> None:
        cls.validate_required_fields(cfg)
        cls.validate_field_types(cfg)
        cls.validate_connection_shape(cfg)

    @staticmethod
    def validate_connection_shape(cfg) -> None:
        from emperor.neuron._options import TerminalConnectionShapeOptions

        if not isinstance(
            cfg.connection_shape,
            TerminalConnectionShapeOptions,
        ):
            raise TypeError(
                "connection_shape must be a TerminalConnectionShapeOptions "
                f"for TerminalConfig, got {type(cfg.connection_shape).__name__}."
            )

    @classmethod
    def validate(cls, model: "Terminal") -> None:
        cls.validate_config_fields(model.cfg)
        cls.validate_positive_integer("input_dim", model.input_dim)
        cls.validate_integer("x_axis_position", model.x_axis_position)
        cls.validate_integer("y_axis_position", model.y_axis_position)
        cls.validate_integer("z_axis_position", model.z_axis_position)
        cls.validate_axis_ranges(model)
        cls.validate_sampler_config(model)

    @classmethod
    def validate_config_composition(cls, cfg) -> None:
        """Validate Terminal composition without constructing trainable modules."""

        from emperor.neuron._terminal.topology import initialize_terminal_connections

        cls.validate_config_fields(cfg)
        neuron_connections = initialize_terminal_connections(cfg)
        terminal_validation_target = SimpleNamespace(
            cfg=cfg,
            input_dim=cfg.input_dim,
            x_axis_position=cfg.x_axis_position,
            y_axis_position=cfg.y_axis_position,
            z_axis_position=cfg.z_axis_position,
            xy_axis_range=cfg.xy_axis_range.value,
            z_axis_range=cfg.z_axis_range.value,
            z_axis_offset=cfg.z_axis_offset.value,
            sampler_config=cfg.sampler_config,
            total_neuron_connections=int(neuron_connections.shape[0]),
        )
        cls.validate(terminal_validation_target)
        cfg.sampler_config.validate_for_router_input_dim(cfg.input_dim)

    @staticmethod
    def validate_axis_ranges(model: "Terminal") -> None:
        if model.z_axis_offset >= model.z_axis_range:
            raise ValueError(
                "z_axis_offset must be smaller than z_axis_range for Terminal, "
                f"received z_axis_offset={model.z_axis_offset} and "
                f"z_axis_range={model.z_axis_range}."
            )

    @classmethod
    def validate_sampler_config(cls, model: "Terminal") -> None:
        from emperor.sampler import RouterConfig

        sampler_config = model.sampler_config
        cls.validate_positive_integer(
            "sampler_config.num_experts",
            sampler_config.num_experts,
        )
        if sampler_config.num_experts != model.total_neuron_connections:
            raise ValueError(
                "sampler_config.num_experts must equal Terminal "
                "total_neuron_connections, received "
                f"num_experts={sampler_config.num_experts} and "
                f"total_neuron_connections={model.total_neuron_connections}."
            )

        router_config = sampler_config.router_config
        if router_config is None:
            cls.validate_logits_only_input_dim(model)
            return
        if not isinstance(router_config, RouterConfig):
            raise TypeError(
                "sampler_config.router_config must be a RouterConfig for Terminal, "
                f"got {type(router_config).__name__}."
            )
        cls.validate_positive_integer(
            "sampler_config.router_config.num_experts",
            router_config.num_experts,
        )
        if router_config.num_experts != model.total_neuron_connections:
            raise ValueError(
                "sampler_config.router_config.num_experts must equal Terminal "
                "total_neuron_connections, received "
                f"num_experts={router_config.num_experts} and "
                f"total_neuron_connections={model.total_neuron_connections}."
            )

    @staticmethod
    def validate_logits_only_input_dim(model: "Terminal") -> None:
        if model.input_dim == model.total_neuron_connections:
            return
        raise ValueError(
            "sampler_config.router_config is required when Terminal input_dim "
            "does not equal total_neuron_connections, received "
            f"input_dim={model.input_dim} and "
            f"total_neuron_connections={model.total_neuron_connections}."
        )

    @classmethod
    def validate_forward_input(cls, model: "Terminal", input: Tensor) -> None:
        cls.validate_tensor_rank("Terminal input", input, 2)
        if input.shape[-1] != model.input_dim:
            raise ValueError(
                "Terminal input feature dimension must match input_dim, "
                f"received input_dim={model.input_dim} and input shape "
                f"{tuple(input.shape)}."
            )

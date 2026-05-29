from torch import Tensor

from emperor.base.utils import ConfigBase
from emperor.base.validator import ValidatorBase

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.neuron.config import NeuronClusterConfig, NeuronConfig
    from emperor.neuron.core.layers import Axons, Nucleus, Terminal


class NeuronValidationMixin:
    @staticmethod
    def validate_config_base(name: str, value) -> None:
        if not isinstance(value, ConfigBase):
            raise TypeError(
                f"{name} must be a ConfigBase instance, "
                f"got {type(value).__name__}."
            )

    @staticmethod
    def validate_integer(name: str, value: int) -> None:
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(
                f"{name} must be an integer, received {type(value).__name__}."
            )

    @staticmethod
    def validate_positive_integer(name: str, value: int) -> None:
        NeuronValidationMixin.validate_integer(name, value)
        if value <= 0:
            raise ValueError(f"{name} must be a positive integer, received {value!r}.")

    @staticmethod
    def validate_tensor_rank(name: str, value, rank: int) -> None:
        if not isinstance(value, Tensor):
            raise TypeError(
                f"{name} must be a Tensor, received {type(value).__name__}."
            )
        if value.dim() != rank:
            raise ValueError(
                f"{name} must be a {rank}D tensor, received a "
                f"{value.dim()}D tensor with shape {tuple(value.shape)}."
            )


class NucleusValidator(ValidatorBase, NeuronValidationMixin):
    @staticmethod
    def validate(model: "Nucleus") -> None:
        NucleusValidator.validate_required_fields(model.cfg)
        NucleusValidator.validate_field_types(model.cfg)
        NucleusValidator.validate_config_base("model_config", model.cfg.model_config)

    @staticmethod
    def validate_forward_input(input: Tensor) -> None:
        NucleusValidator.validate_tensor_rank("Nucleus input", input, 2)


class AxonsValidator(ValidatorBase, NeuronValidationMixin):
    OPTIONAL_FIELDS = {"memory_config"}

    @staticmethod
    def validate(model: "Axons") -> None:
        AxonsValidator.validate_required_fields(model.cfg)
        AxonsValidator.validate_field_types(model.cfg)
        AxonsValidator.validate_memory_config(model.cfg.memory_config)

    @staticmethod
    def validate_memory_config(memory_config: ConfigBase | None) -> None:
        if memory_config is None:
            return
        AxonsValidator.validate_config_base("memory_config", memory_config)

    @staticmethod
    def validate_forward_input(input: Tensor) -> None:
        AxonsValidator.validate_tensor_rank("Axons input", input, 2)


class TerminalValidator(ValidatorBase, NeuronValidationMixin):
    @staticmethod
    def validate_config_fields(cfg) -> None:
        TerminalValidator.validate_required_fields(cfg)
        TerminalValidator.validate_field_types(cfg)

    @staticmethod
    def validate(model: "Terminal") -> None:
        TerminalValidator.validate_config_fields(model.cfg)
        TerminalValidator.validate_positive_integer("input_dim", model.input_dim)
        TerminalValidator.validate_integer("x_axis_position", model.x_axis_position)
        TerminalValidator.validate_integer("y_axis_position", model.y_axis_position)
        TerminalValidator.validate_integer("z_axis_position", model.z_axis_position)
        TerminalValidator.validate_axis_ranges(model)
        TerminalValidator.validate_sampler_config(model)

    @staticmethod
    def validate_axis_ranges(model: "Terminal") -> None:
        if model.z_axis_offset >= model.z_axis_range:
            raise ValueError(
                "z_axis_offset must be smaller than z_axis_range for Terminal, "
                f"received z_axis_offset={model.z_axis_offset} and "
                f"z_axis_range={model.z_axis_range}."
            )

    @staticmethod
    def validate_sampler_config(model: "Terminal") -> None:
        from emperor.sampler.core.config import RouterConfig, SamplerConfig

        sampler_config = model.sampler_config
        if not isinstance(sampler_config, SamplerConfig):
            raise TypeError(
                "sampler_config must be a SamplerConfig for Terminal, "
                f"got {type(sampler_config).__name__}."
            )
        TerminalValidator.validate_positive_integer(
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
            TerminalValidator.validate_logits_only_input_dim(model)
            return
        if not isinstance(router_config, RouterConfig):
            raise TypeError(
                "sampler_config.router_config must be a RouterConfig for Terminal, "
                f"got {type(router_config).__name__}."
            )
        TerminalValidator.validate_positive_integer(
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

    @staticmethod
    def validate_forward_input(model: "Terminal", input: Tensor) -> None:
        TerminalValidator.validate_tensor_rank("Terminal input", input, 2)
        if input.shape[-1] != model.input_dim:
            raise ValueError(
                "Terminal input feature dimension must match input_dim, "
                f"received input_dim={model.input_dim} and input shape "
                f"{tuple(input.shape)}."
            )


class NeuronValidator(ValidatorBase, NeuronValidationMixin):
    @staticmethod
    def validate(cfg: "NeuronConfig") -> None:
        NeuronValidator.validate_required_fields(cfg)
        NeuronValidator.validate_field_types(cfg)

    @staticmethod
    def validate_forward_input(input: Tensor) -> None:
        NeuronValidator.validate_tensor_rank("Neuron input", input, 2)


class NeuronClusterValidator(ValidatorBase, NeuronValidationMixin):
    OPTIONAL_FIELDS = {"growth_threshold"}

    @staticmethod
    def validate(model) -> None:
        NeuronClusterValidator.validate_required_fields(model.cfg)
        NeuronClusterValidator.validate_field_types(model.cfg)
        NeuronValidator.validate(model.cfg.neuron_config)
        NeuronClusterValidator.validate_positive_integer(
            "x_axis_total_neurons",
            model.cfg.x_axis_total_neurons,
        )
        NeuronClusterValidator.validate_positive_integer(
            "y_axis_total_neurons",
            model.cfg.y_axis_total_neurons,
        )
        NeuronClusterValidator.validate_positive_integer(
            "z_axis_total_neurons",
            model.cfg.z_axis_total_neurons,
        )
        NeuronClusterValidator.validate_growth_threshold(
            model.cfg.growth_threshold,
        )

    @staticmethod
    def validate_growth_threshold(growth_threshold: int | None) -> None:
        if growth_threshold is None:
            return
        NeuronClusterValidator.validate_positive_integer(
            "growth_threshold",
            growth_threshold,
        )

    @staticmethod
    def validate_forward_input(input: Tensor) -> None:
        NeuronClusterValidator.validate_tensor_rank("NeuronCluster input", input, 2)

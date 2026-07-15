from typing import TYPE_CHECKING

from torch import Tensor

from emperor.base.config import ConfigBase
from emperor.base.validator import ValidatorBase

if TYPE_CHECKING:
    from emperor.memory.config import DynamicMemoryConfig
    from emperor.neuron.core.config import NeuronClusterConfig, NeuronConfig
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
    def validate_memory_config(memory_config: "DynamicMemoryConfig | None") -> None:
        if memory_config is None:
            return
        from emperor.memory.config import DynamicMemoryConfig

        if not isinstance(memory_config, DynamicMemoryConfig):
            raise TypeError(
                "memory_config must be an instance of DynamicMemoryConfig for "
                f"AxonsConfig, got {type(memory_config).__name__}."
            )

    @staticmethod
    def validate_forward_input(input: Tensor) -> None:
        AxonsValidator.validate_tensor_rank("Axons input", input, 2)


class TerminalValidator(ValidatorBase, NeuronValidationMixin):
    OPTIONAL_FIELDS = {"connection_shape"}

    @staticmethod
    def validate_config_fields(cfg) -> None:
        TerminalValidator.validate_required_fields(cfg)
        TerminalValidator.validate_field_types(cfg)
        TerminalValidator.validate_connection_shape(cfg)

    @staticmethod
    def validate_connection_shape(cfg) -> None:
        from emperor.neuron.core.options import TerminalConnectionShapeOptions

        if cfg.connection_shape is None:
            return
        if not isinstance(cfg.connection_shape, TerminalConnectionShapeOptions):
            raise TypeError(
                "connection_shape must be a TerminalConnectionShapeOptions "
                f"for TerminalConfig, got {type(cfg.connection_shape).__name__}."
            )

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
    OPTIONAL_FIELDS = {"coordinate_embedding_flag"}

    @staticmethod
    def validate(cfg: "NeuronConfig") -> None:
        NeuronValidator.validate_required_fields(cfg)
        NeuronValidator.validate_field_types(cfg)
        NeuronValidator.validate_coordinate_embedding_options(cfg)

    @staticmethod
    def validate_coordinate_embedding_options(cfg: "NeuronConfig") -> None:
        flag_value = cfg.coordinate_embedding_flag
        if flag_value is None:
            return
        if not isinstance(flag_value, bool):
            raise TypeError(
                "coordinate_embedding_flag must be a bool for NeuronConfig, "
                f"got {type(flag_value).__name__}."
            )
        if not flag_value:
            return
        input_dim = cfg.terminal_config.input_dim
        if input_dim < 3:
            raise ValueError(
                "coordinate_embedding_flag requires terminal_config.input_dim "
                "of at least 3 so every coordinate axis receives at least one "
                f"encoding channel, received input_dim={input_dim}."
            )

    @staticmethod
    def validate_forward_input(input: Tensor) -> None:
        NeuronValidator.validate_tensor_rank("Neuron input", input, 2)


class NeuronClusterValidator(ValidatorBase, NeuronValidationMixin):
    OPTIONAL_FIELDS = {
        "beam_width",
        "entry_sampler_config",
        "escape_driven_growth_flag",
        "growth_cooldown_steps",
        "growth_threshold",
        "growth_warmup_steps",
        "max_total_growths",
        "halting_config",
        "initial_x_axis_total_neurons",
        "initial_y_axis_total_neurons",
        "initial_z_axis_total_neurons",
        "mitosis_initialization_flag",
        "pruning_threshold",
    }

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
        NeuronClusterValidator.validate_initial_grid_dimensions(model.cfg)
        NeuronClusterValidator.validate_positive_integer(
            "max_steps",
            model.cfg.max_steps,
        )
        NeuronClusterValidator.validate_beam_width(model.cfg.beam_width)
        NeuronClusterValidator.validate_entry_sampler_config(model.cfg)
        NeuronClusterValidator.validate_derived_entry_sampler_config(model.cfg)
        NeuronClusterValidator.validate_growth_threshold(
            model.cfg.growth_threshold,
        )
        NeuronClusterValidator.validate_pruning_threshold(
            model.cfg.pruning_threshold,
        )
        NeuronClusterValidator.validate_growth_placement_options(model.cfg)
        NeuronClusterValidator.validate_growth_budget_options(model.cfg)
        NeuronClusterValidator.validate_growth_warmup_steps(model.cfg)
        NeuronClusterValidator.validate_halting_config(model.cfg)
        NeuronClusterValidator.validate_nucleus_model_dimensions(model.cfg)

    @staticmethod
    def validate_initial_grid_dimensions(cfg: "NeuronClusterConfig") -> None:
        axis_pairs = (
            (
                "initial_x_axis_total_neurons",
                cfg.initial_x_axis_total_neurons,
                "x_axis_total_neurons",
                cfg.x_axis_total_neurons,
            ),
            (
                "initial_y_axis_total_neurons",
                cfg.initial_y_axis_total_neurons,
                "y_axis_total_neurons",
                cfg.y_axis_total_neurons,
            ),
            (
                "initial_z_axis_total_neurons",
                cfg.initial_z_axis_total_neurons,
                "z_axis_total_neurons",
                cfg.z_axis_total_neurons,
            ),
        )
        for initial_name, initial_value, max_name, max_value in axis_pairs:
            if initial_value is None:
                continue
            NeuronClusterValidator.validate_positive_integer(
                initial_name,
                initial_value,
            )
            if initial_value > max_value:
                raise ValueError(
                    f"{initial_name} cannot exceed {max_name}, received "
                    f"{initial_name}={initial_value} and {max_name}={max_value}."
                )

    @staticmethod
    def validate_entry_sampler_config(cfg: "NeuronClusterConfig") -> None:
        sampler_config = cfg.entry_sampler_config
        if sampler_config is None:
            return

        from emperor.sampler.core.config import RouterConfig, SamplerConfig

        if not isinstance(sampler_config, SamplerConfig):
            raise TypeError(
                "entry_sampler_config must be a SamplerConfig for "
                "NeuronClusterConfig, got "
                f"{type(sampler_config).__name__}."
            )

        entry_count = (
            (cfg.initial_x_axis_total_neurons or cfg.x_axis_total_neurons)
            * (cfg.initial_y_axis_total_neurons or cfg.y_axis_total_neurons)
        )
        NeuronClusterValidator.validate_positive_integer(
            "entry_sampler_config.num_experts",
            sampler_config.num_experts,
        )
        if sampler_config.num_experts != entry_count:
            raise ValueError(
                "entry_sampler_config.num_experts must equal the initialized "
                "entry coordinate count, received "
                f"num_experts={sampler_config.num_experts} and "
                f"entry_coordinate_count={entry_count}."
            )
        NeuronClusterValidator.validate_positive_integer(
            "entry_sampler_config.top_k",
            sampler_config.top_k,
        )
        if sampler_config.top_k > entry_count:
            raise ValueError(
                "entry_sampler_config.top_k cannot exceed the initialized entry "
                "coordinate count, received "
                f"top_k={sampler_config.top_k} and "
                f"entry_coordinate_count={entry_count}."
            )

        router_config = sampler_config.router_config
        if router_config is None:
            return
        if not isinstance(router_config, RouterConfig):
            raise TypeError(
                "entry_sampler_config.router_config must be a RouterConfig for "
                "NeuronClusterConfig, got "
                f"{type(router_config).__name__}."
            )
        NeuronClusterValidator.validate_positive_integer(
            "entry_sampler_config.router_config.num_experts",
            router_config.num_experts,
        )
        if router_config.num_experts != entry_count:
            raise ValueError(
                "entry_sampler_config.router_config.num_experts must equal the "
                "initialized entry coordinate count, received "
                f"num_experts={router_config.num_experts} and "
                f"entry_coordinate_count={entry_count}."
            )

    @staticmethod
    def validate_derived_entry_sampler_config(cfg: "NeuronClusterConfig") -> None:
        if cfg.entry_sampler_config is not None:
            return

        terminal_config = cfg.neuron_config.terminal_config
        if terminal_config.sampler_config.router_config is not None:
            return

        entry_count = (
            (cfg.initial_x_axis_total_neurons or cfg.x_axis_total_neurons)
            * (cfg.initial_y_axis_total_neurons or cfg.y_axis_total_neurons)
        )
        if terminal_config.input_dim == entry_count:
            return
        raise ValueError(
            "entry_sampler_config is required when the terminal sampler has no "
            "router_config and input_dim does not equal the initialized entry "
            "coordinate count, received "
            f"input_dim={terminal_config.input_dim} and "
            f"entry_coordinate_count={entry_count}."
        )

    @staticmethod
    def validate_beam_width(beam_width: int | None) -> None:
        if beam_width is None:
            return
        NeuronClusterValidator.validate_positive_integer("beam_width", beam_width)

    @staticmethod
    def validate_growth_threshold(growth_threshold: int | None) -> None:
        if growth_threshold is None:
            return
        NeuronClusterValidator.validate_positive_integer(
            "growth_threshold",
            growth_threshold,
        )

    @staticmethod
    def validate_pruning_threshold(pruning_threshold: int | None) -> None:
        if pruning_threshold is None:
            return
        NeuronClusterValidator.validate_positive_integer(
            "pruning_threshold",
            pruning_threshold,
        )

    @staticmethod
    def validate_growth_placement_options(cfg: "NeuronClusterConfig") -> None:
        flag_fields = (
            ("escape_driven_growth_flag", cfg.escape_driven_growth_flag),
            ("mitosis_initialization_flag", cfg.mitosis_initialization_flag),
        )
        for flag_name, flag_value in flag_fields:
            if flag_value is None:
                continue
            if not isinstance(flag_value, bool):
                raise TypeError(
                    f"{flag_name} must be a bool for NeuronClusterConfig, "
                    f"got {type(flag_value).__name__}."
                )
            if flag_value and cfg.growth_threshold is None:
                raise ValueError(
                    f"{flag_name} requires growth_threshold to be set for "
                    "NeuronClusterConfig; growth options have no effect when "
                    "growth is disabled."
                )

    @staticmethod
    def validate_growth_budget_options(cfg: "NeuronClusterConfig") -> None:
        budget_fields = (
            ("growth_cooldown_steps", cfg.growth_cooldown_steps),
            ("max_total_growths", cfg.max_total_growths),
        )
        for budget_name, budget_value in budget_fields:
            if budget_value is None:
                continue
            NeuronClusterValidator.validate_positive_integer(
                budget_name,
                budget_value,
            )
            if cfg.growth_threshold is None:
                raise ValueError(
                    f"{budget_name} requires growth_threshold to be set for "
                    "NeuronClusterConfig; growth options have no effect when "
                    "growth is disabled."
                )

    @staticmethod
    def validate_growth_warmup_steps(cfg: "NeuronClusterConfig") -> None:
        if cfg.growth_warmup_steps is None:
            return
        NeuronClusterValidator.validate_positive_integer(
            "growth_warmup_steps",
            cfg.growth_warmup_steps,
        )
        if cfg.growth_threshold is None:
            raise ValueError(
                "growth_warmup_steps requires growth_threshold to be set for "
                "NeuronClusterConfig; growth options have no effect when "
                "growth is disabled."
            )

    @staticmethod
    def validate_halting_config(cfg: "NeuronClusterConfig") -> None:
        halting_config = cfg.halting_config
        if halting_config is None:
            return

        from emperor.halting.config import HaltingConfig

        if not isinstance(halting_config, HaltingConfig):
            raise TypeError(
                "halting_config must be an instance of HaltingConfig for "
                f"NeuronClusterConfig, got {type(halting_config).__name__}"
            )

        try:
            owner = halting_config._registry_owner()
        except NotImplementedError as exc:
            raise ValueError(
                "halting_config must be a concrete halting config for "
                "NeuronClusterConfig"
            ) from exc

        if not hasattr(owner, "update_halting_state") or not hasattr(
            owner,
            "finalize_weighted_accumulation",
        ):
            raise ValueError(
                f"halting_config {type(halting_config).__name__} builds "
                f"{owner.__name__}, which does not expose update_halting_state "
                "and finalize_weighted_accumulation required by NeuronCluster"
            )

        terminal_input_dim = cfg.neuron_config.terminal_config.input_dim
        if (
            halting_config.input_dim is not None
            and halting_config.input_dim != terminal_input_dim
        ):
            raise ValueError(
                "halting_config.input_dim must match "
                "neuron_config.terminal_config.input_dim for NeuronClusterConfig, "
                f"got halting_config.input_dim={halting_config.input_dim} and "
                f"terminal input_dim={terminal_input_dim}."
            )

    @staticmethod
    def validate_nucleus_model_dimensions(cfg: "NeuronClusterConfig") -> None:
        model_config = cfg.neuron_config.nucleus_config.model_config
        terminal_input_dim = cfg.neuron_config.terminal_config.input_dim
        for dimension_name in ("input_dim", "output_dim"):
            dimension_value = getattr(model_config, dimension_name, None)
            if dimension_value is None:
                continue
            if dimension_value != terminal_input_dim:
                raise ValueError(
                    "nucleus_config.model_config must preserve the terminal "
                    "feature dimension for NeuronClusterConfig, received "
                    f"{dimension_name}={dimension_value} and terminal "
                    f"input_dim={terminal_input_dim}."
                )

    @staticmethod
    def validate_forward_input(input: Tensor) -> None:
        if not isinstance(input, Tensor):
            raise TypeError(
                "NeuronCluster input must be a Tensor, "
                f"received {type(input).__name__}."
            )
        if input.dim() < 2:
            raise ValueError(
                "NeuronCluster input must be a feature-last tensor with at least "
                f"2 dimensions, received shape {tuple(input.shape)}."
            )

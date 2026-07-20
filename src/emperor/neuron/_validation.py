import copy
from typing import TYPE_CHECKING

from torch import Tensor

from emperor._validation import ValidatorBase

if TYPE_CHECKING:
    from emperor.memory import DynamicMemoryConfig
    from emperor.neuron._config import NeuronClusterConfig, NeuronConfig
    from emperor.neuron._parts import Axons, Nucleus, Terminal


class NeuronValidationMixin:
    @staticmethod
    def validate_integer(name: str, value: int) -> None:
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(
                f"{name} must be an integer, received {type(value).__name__}."
            )

    @classmethod
    def validate_positive_integer(cls, name: str, value: int) -> None:
        cls.validate_integer(name, value)
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
    @classmethod
    def validate(cls, model: "Nucleus") -> None:
        cls.validate_required_fields(model.cfg)
        cls.validate_field_types(model.cfg)

    @classmethod
    def validate_forward_input(cls, input: Tensor) -> None:
        cls.validate_tensor_rank("Nucleus input", input, 2)


class AxonsValidator(ValidatorBase, NeuronValidationMixin):
    OPTIONAL_FIELDS = {"memory_config"}

    @classmethod
    def validate(cls, model: "Axons") -> None:
        cls.validate_required_fields(model.cfg)
        cls.validate_field_types(model.cfg)
        cls.validate_memory_config(model.cfg.memory_config)

    @staticmethod
    def validate_memory_config(memory_config: "DynamicMemoryConfig | None") -> None:
        if memory_config is None:
            return
        from emperor.memory import DynamicMemoryConfig

        if not isinstance(memory_config, DynamicMemoryConfig):
            raise TypeError(
                "memory_config must be an instance of DynamicMemoryConfig for "
                f"AxonsConfig, got {type(memory_config).__name__}."
            )

    @classmethod
    def validate_forward_input(cls, input: Tensor) -> None:
        cls.validate_tensor_rank("Axons input", input, 2)


class TerminalValidator(ValidatorBase, NeuronValidationMixin):
    OPTIONAL_FIELDS = {"connection_shape"}

    @classmethod
    def validate_config_fields(cls, cfg) -> None:
        cls.validate_required_fields(cfg)
        cls.validate_field_types(cfg)
        cls.validate_connection_shape(cfg)

    @staticmethod
    def validate_connection_shape(cfg) -> None:
        from emperor.neuron._options import TerminalConnectionShapeOptions

        if cfg.connection_shape is None:
            return
        if not isinstance(cfg.connection_shape, TerminalConnectionShapeOptions):
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


class NeuronValidator(ValidatorBase, NeuronValidationMixin):
    OPTIONAL_FIELDS = {"coordinate_embedding_flag"}

    @classmethod
    def validate(cls, cfg: "NeuronConfig") -> None:
        cls.validate_required_fields(cfg)
        cls.validate_field_types(cfg)
        cls.validate_coordinate_embedding_options(cfg)

    @staticmethod
    def validate_coordinate_embedding_options(cfg: "NeuronConfig") -> None:
        coordinate_embedding_flag = cfg.coordinate_embedding_flag
        if coordinate_embedding_flag is None:
            return
        if not isinstance(coordinate_embedding_flag, bool):
            raise TypeError(
                "coordinate_embedding_flag must be a bool for NeuronConfig, "
                f"got {type(coordinate_embedding_flag).__name__}."
            )
        if not coordinate_embedding_flag:
            return
        terminal_input_dim = cfg.terminal_config.input_dim
        if terminal_input_dim < 3:
            raise ValueError(
                "coordinate_embedding_flag requires terminal_config.input_dim "
                "of at least 3 so every coordinate axis receives at least one "
                f"encoding channel, received input_dim={terminal_input_dim}."
            )

    @classmethod
    def validate_forward_input(cls, input: Tensor) -> None:
        cls.validate_tensor_rank("Neuron input", input, 2)


class NeuronClusterValidator(ValidatorBase, NeuronValidationMixin):
    NEURON_VALIDATOR = NeuronValidator

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

    @classmethod
    def validate(cls, model) -> None:
        cls.validate_required_fields(model.cfg)
        cls.validate_field_types(model.cfg)
        cls.NEURON_VALIDATOR.validate(model.cfg.neuron_config)
        cls.validate_positive_integer(
            "x_axis_total_neurons",
            model.cfg.x_axis_total_neurons,
        )
        cls.validate_positive_integer(
            "y_axis_total_neurons",
            model.cfg.y_axis_total_neurons,
        )
        cls.validate_positive_integer(
            "z_axis_total_neurons",
            model.cfg.z_axis_total_neurons,
        )
        cls.validate_initial_grid_dimensions(model.cfg)
        cls.validate_positive_integer(
            "max_steps",
            model.cfg.max_steps,
        )
        cls.validate_beam_width(model.cfg.beam_width)
        cls.validate_entry_sampler_config(model.cfg)
        cls.validate_derived_entry_sampler_config(model.cfg)
        cls.validate_growth_threshold(model.cfg.growth_threshold)
        cls.validate_pruning_threshold(model.cfg.pruning_threshold)
        cls.validate_growth_placement_options(model.cfg)
        cls.validate_growth_budget_options(model.cfg)
        cls.validate_growth_warmup_steps(model.cfg)
        cls.validate_halting_config(model.cfg)
        cls.validate_nucleus_model_dimensions(model.cfg)

    @classmethod
    def validate_initial_grid_dimensions(cls, cfg: "NeuronClusterConfig") -> None:
        initial_capacity_pairs = (
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
        for (
            initial_dimension_name,
            initial_dimension,
            capacity_name,
            capacity,
        ) in initial_capacity_pairs:
            if initial_dimension is None:
                continue
            cls.validate_positive_integer(initial_dimension_name, initial_dimension)
            if initial_dimension > capacity:
                raise ValueError(
                    f"{initial_dimension_name} cannot exceed {capacity_name}, "
                    f"received {initial_dimension_name}={initial_dimension} and "
                    f"{capacity_name}={capacity}."
                )

    @classmethod
    def validate_entry_sampler_config(cls, cfg: "NeuronClusterConfig") -> None:
        sampler_config = cfg.entry_sampler_config
        if sampler_config is None:
            return

        from emperor.sampler import RouterConfig, SamplerConfig

        if not isinstance(sampler_config, SamplerConfig):
            raise TypeError(
                "entry_sampler_config must be a SamplerConfig for "
                "NeuronClusterConfig, got "
                f"{type(sampler_config).__name__}."
            )

        initialized_entry_count = (
            cfg.initial_x_axis_total_neurons or cfg.x_axis_total_neurons
        ) * (cfg.initial_y_axis_total_neurons or cfg.y_axis_total_neurons)
        cls.validate_positive_integer(
            "entry_sampler_config.num_experts",
            sampler_config.num_experts,
        )
        if sampler_config.num_experts != initialized_entry_count:
            raise ValueError(
                "entry_sampler_config.num_experts must equal the initialized "
                "entry coordinate count, received "
                f"num_experts={sampler_config.num_experts} and "
                f"entry_coordinate_count={initialized_entry_count}."
            )
        cls.validate_positive_integer(
            "entry_sampler_config.top_k",
            sampler_config.top_k,
        )
        if sampler_config.top_k > initialized_entry_count:
            raise ValueError(
                "entry_sampler_config.top_k cannot exceed the initialized entry "
                "coordinate count, received "
                f"top_k={sampler_config.top_k} and "
                f"entry_coordinate_count={initialized_entry_count}."
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
        cls.validate_positive_integer(
            "entry_sampler_config.router_config.num_experts",
            router_config.num_experts,
        )
        if router_config.num_experts != initialized_entry_count:
            raise ValueError(
                "entry_sampler_config.router_config.num_experts must equal the "
                "initialized entry coordinate count, received "
                f"num_experts={router_config.num_experts} and "
                f"entry_coordinate_count={initialized_entry_count}."
            )

    @staticmethod
    def validate_derived_entry_sampler_config(cfg: "NeuronClusterConfig") -> None:
        if cfg.entry_sampler_config is not None:
            return

        terminal_config = cfg.neuron_config.terminal_config
        if terminal_config.sampler_config.router_config is not None:
            return

        initialized_entry_count = (
            cfg.initial_x_axis_total_neurons or cfg.x_axis_total_neurons
        ) * (cfg.initial_y_axis_total_neurons or cfg.y_axis_total_neurons)
        if terminal_config.input_dim == initialized_entry_count:
            return
        raise ValueError(
            "entry_sampler_config is required when the terminal sampler has no "
            "router_config and input_dim does not equal the initialized entry "
            "coordinate count, received "
            f"input_dim={terminal_config.input_dim} and "
            f"entry_coordinate_count={initialized_entry_count}."
        )

    @classmethod
    def validate_beam_width(cls, beam_width: int | None) -> None:
        if beam_width is None:
            return
        cls.validate_positive_integer("beam_width", beam_width)

    @classmethod
    def validate_growth_threshold(cls, growth_threshold: int | None) -> None:
        if growth_threshold is None:
            return
        cls.validate_positive_integer("growth_threshold", growth_threshold)

    @classmethod
    def validate_pruning_threshold(cls, pruning_threshold: int | None) -> None:
        if pruning_threshold is None:
            return
        cls.validate_positive_integer("pruning_threshold", pruning_threshold)

    @staticmethod
    def validate_growth_placement_options(cfg: "NeuronClusterConfig") -> None:
        growth_flag_fields = (
            ("escape_driven_growth_flag", cfg.escape_driven_growth_flag),
            ("mitosis_initialization_flag", cfg.mitosis_initialization_flag),
        )
        for flag_name, flag_value in growth_flag_fields:
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

    @classmethod
    def validate_growth_budget_options(cls, cfg: "NeuronClusterConfig") -> None:
        growth_budget_fields = (
            ("growth_cooldown_steps", cfg.growth_cooldown_steps),
            ("max_total_growths", cfg.max_total_growths),
        )
        for budget_name, budget_value in growth_budget_fields:
            if budget_value is None:
                continue
            cls.validate_positive_integer(budget_name, budget_value)
            if cfg.growth_threshold is None:
                raise ValueError(
                    f"{budget_name} requires growth_threshold to be set for "
                    "NeuronClusterConfig; growth options have no effect when "
                    "growth is disabled."
                )

    @classmethod
    def validate_growth_warmup_steps(cls, cfg: "NeuronClusterConfig") -> None:
        if cfg.growth_warmup_steps is None:
            return
        cls.validate_positive_integer("growth_warmup_steps", cfg.growth_warmup_steps)
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

        from emperor.halting import HaltingConfig

        if not isinstance(halting_config, HaltingConfig):
            raise TypeError(
                "halting_config must be an instance of HaltingConfig for "
                f"NeuronClusterConfig, got {type(halting_config).__name__}"
            )

        try:
            halting_model_type = halting_config._registry_owner()
        except NotImplementedError as registry_error:
            raise ValueError(
                "halting_config must be a concrete halting config for "
                "NeuronClusterConfig"
            ) from registry_error

        from emperor.halting import HaltingBase

        implements_halting_interface = (
            isinstance(halting_model_type, type)
            and issubclass(halting_model_type, HaltingBase)
            and halting_model_type.implements_halting_interface()
        )
        if not implements_halting_interface:
            halting_model_name = (
                halting_model_type.__name__
                if isinstance(halting_model_type, type)
                else type(halting_model_type).__name__
            )
            raise ValueError(
                f"halting_config {type(halting_config).__name__} builds "
                f"{halting_model_name}, which does not implement the HaltingBase "
                "lifecycle required by NeuronCluster"
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

        resolved_halting_config = copy.deepcopy(halting_config)
        resolved_halting_config.input_dim = terminal_input_dim
        if resolved_halting_config.threshold is None:
            resolved_halting_config.threshold = halting_config.DEFAULT_THRESHOLD
        halting_model_type.validate_resolved_config(resolved_halting_config)

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

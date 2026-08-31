import copy
from typing import TYPE_CHECKING

from torch import Tensor

from emperor._validation import ValidatorBase
from emperor.neuron._neuron.validation import NeuronValidator
from emperor.neuron._validation.common import NeuronValidationMixin

if TYPE_CHECKING:
    from emperor.neuron._config import NeuronClusterConfig, NeuronConfig


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
        cls.NEURON_VALIDATOR.validate(
            cls.__neuron_config_with_validation_positions(model.cfg.neuron_config)
        )
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

    @staticmethod
    def __neuron_config_with_validation_positions(
        neuron_config: "NeuronConfig",
    ) -> "NeuronConfig":
        terminal_config = neuron_config.terminal_config
        if terminal_config is None:
            return neuron_config
        if all(
            getattr(terminal_config, field_name) is not None
            for field_name in (
                "x_axis_position",
                "y_axis_position",
                "z_axis_position",
            )
        ):
            return neuron_config
        resolved_neuron_config = copy.deepcopy(neuron_config)
        for field_name in (
            "x_axis_position",
            "y_axis_position",
            "z_axis_position",
        ):
            if getattr(resolved_neuron_config.terminal_config, field_name) is None:
                setattr(
                    resolved_neuron_config.terminal_config,
                    field_name,
                    0,
                )
        return resolved_neuron_config

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
            sampler_config.validate_for_router_input_dim()
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
        sampler_config.validate_for_router_input_dim(
            cfg.neuron_config.terminal_config.input_dim
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

        validator = getattr(halting_model_type, "VALIDATOR", None)
        validate_owner_step_contract = getattr(
            validator,
            "validate_owner_step_contract",
            None,
        )
        if callable(validate_owner_step_contract):
            validate_owner_step_contract(
                halting_config,
                owner_step_limit=None,
                owner_name="NeuronClusterConfig",
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

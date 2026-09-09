import math
from dataclasses import replace
from numbers import Real

from emperor.config import ConfigBase
from emperor.halting import HaltingConfig
from emperor.layers import (
    GateConfig,
    LastLayerBiasOptions,
    LayerStackConfig,
    RecurrentLayerConfig,
)
from emperor.memory import DynamicMemoryConfig
from models.gpt.expert_linear_adaptive.runtime_options import (
    ExpertsDynamicMemoryOptions,
    ExpertsLayerControllerOptions,
    ExpertsRecurrentControllerOptions,
    ExpertsStackOptions,
    ExpertsSubmoduleStackOptions,
    resolve_experts_controller_stack_options,
)

from ._controller_stack_config import build_controller_stack_config
from ._residual import build_residual_config


class ExpertsGateConfigFactory:
    def __init__(
        self,
        *,
        layer_controller_options: ExpertsLayerControllerOptions,
        recurrent_controller_options: ExpertsRecurrentControllerOptions,
        submodule_stack_options: ExpertsSubmoduleStackOptions,
        recurrent_stack_inherits_gate_stack: bool = True,
    ) -> None:
        self.layer_controller_options = layer_controller_options
        self.recurrent_controller_options = recurrent_controller_options
        self.submodule_stack_options = submodule_stack_options
        self.recurrent_stack_inherits_gate_stack = recurrent_stack_inherits_gate_stack

    def build_gate_config(self) -> GateConfig | None:
        if not self.layer_controller_options.stack_gate_flag:
            return None
        return GateConfig(
            model_config=self.__build_gate_model_config(),
            option=self.layer_controller_options.gate_option,
            activation=self.layer_controller_options.gate_activation,
        )

    def build_recurrent_gate_config(self) -> GateConfig | None:
        if not self.recurrent_controller_options.recurrent_stack_gate_flag:
            return None
        options = resolve_experts_controller_stack_options(
            self.recurrent_controller_options.recurrent_gate_stack_source,
            self.__recurrent_gate_stack_defaults(),
        )
        return GateConfig(
            model_config=build_controller_stack_config(options),
            option=self.recurrent_controller_options.recurrent_gate_option,
            activation=self.recurrent_controller_options.recurrent_gate_activation,
        )

    def __build_gate_model_config(self) -> LayerStackConfig:
        options = resolve_experts_controller_stack_options(
            self.layer_controller_options.gate_stack_source,
            self.submodule_stack_options,
        )
        return build_controller_stack_config(options)

    def __recurrent_gate_stack_defaults(self) -> ExpertsSubmoduleStackOptions:
        if not self.recurrent_stack_inherits_gate_stack:
            return self.submodule_stack_options
        return resolve_experts_controller_stack_options(
            self.layer_controller_options.gate_stack_source,
            self.submodule_stack_options,
        )


class ExpertsHaltingConfigFactory:
    def __init__(
        self,
        *,
        layer_controller_options: ExpertsLayerControllerOptions,
        recurrent_controller_options: ExpertsRecurrentControllerOptions,
        submodule_stack_options: ExpertsSubmoduleStackOptions,
        output_dim: int,
        halting_stack_defaults: ExpertsSubmoduleStackOptions | None = None,
        recurrent_stack_inherits_halting_stack: bool = True,
    ) -> None:
        self.layer_controller_options = layer_controller_options
        self.recurrent_controller_options = recurrent_controller_options
        self.submodule_stack_options = submodule_stack_options
        self.output_dim = output_dim
        self.halting_stack_defaults = halting_stack_defaults
        self.recurrent_stack_inherits_halting_stack = (
            recurrent_stack_inherits_halting_stack
        )

    def build_halting_config(self) -> HaltingConfig | None:
        if not self.layer_controller_options.stack_halting_flag:
            return None
        controller = self.layer_controller_options
        options = resolve_experts_controller_stack_options(
            controller.halting_stack_source,
            self.__halting_stack_defaults(),
        )
        return controller.halting_option(
            threshold=controller.halting_threshold,
            min_steps=1,
            ponder_cost_weight=1.0,
            dropout_probability=controller.halting_dropout,
            hidden_state_mode=controller.halting_hidden_state_mode,
            halting_gate_config=self.__build_halting_gate_stack(options),
        )

    def build_recurrent_halting_config(self) -> HaltingConfig | None:
        if not self.recurrent_controller_options.recurrent_stack_halting_flag:
            return None
        controller = self.recurrent_controller_options
        options = resolve_experts_controller_stack_options(
            controller.recurrent_halting_stack_source,
            self.__recurrent_halting_stack_defaults(),
        )
        return controller.recurrent_halting_option(
            threshold=controller.recurrent_halting_threshold,
            min_steps=controller.recurrent_min_steps,
            ponder_cost_weight=controller.recurrent_ponder_cost_weight,
            dropout_probability=controller.recurrent_halting_dropout,
            hidden_state_mode=controller.recurrent_halting_hidden_state_mode,
            halting_gate_config=self.__build_halting_gate_stack(options),
        )

    def __build_halting_gate_stack(
        self, options: ExpertsSubmoduleStackOptions
    ) -> LayerStackConfig:
        return build_controller_stack_config(
            options,
            hidden_dim=options.hidden_dim or self.output_dim,
            output_dim=self.layer_controller_options.halting_output_dim,
        )

    def __halting_stack_defaults(self) -> ExpertsSubmoduleStackOptions:
        if self.halting_stack_defaults is not None:
            return self.halting_stack_defaults
        return replace(
            self.submodule_stack_options,
            last_layer_bias_option=LastLayerBiasOptions.DISABLED,
        )

    def __recurrent_halting_stack_defaults(self) -> ExpertsSubmoduleStackOptions:
        if not self.recurrent_stack_inherits_halting_stack:
            return self.__halting_stack_defaults()
        return resolve_experts_controller_stack_options(
            self.layer_controller_options.halting_stack_source,
            self.__halting_stack_defaults(),
        )


class ExpertsMemoryConfigFactory:
    def __init__(
        self,
        *,
        stack_options: ExpertsStackOptions | ExpertsSubmoduleStackOptions,
        dynamic_memory_options: ExpertsDynamicMemoryOptions,
        submodule_stack_options: ExpertsSubmoduleStackOptions,
    ) -> None:
        self.stack_options = stack_options
        self.dynamic_memory_options = dynamic_memory_options
        self.submodule_stack_options = submodule_stack_options

    def build_memory_config(self) -> DynamicMemoryConfig | None:
        if not self.dynamic_memory_options.memory_flag:
            return None
        options = resolve_experts_controller_stack_options(
            self.dynamic_memory_options.memory_stack_source,
            self.submodule_stack_options,
        )
        return self.dynamic_memory_options.memory_option(
            input_dim=self.stack_options.hidden_dim,
            output_dim=self.stack_options.hidden_dim,
            memory_position_option=self.dynamic_memory_options.memory_position_option,
            test_time_training_learning_rate=(
                self.dynamic_memory_options.memory_test_time_training_learning_rate
            ),
            test_time_training_num_inner_steps=(
                self.dynamic_memory_options.memory_test_time_training_num_inner_steps
            ),
            model_config=build_controller_stack_config(options),
        )


class ExpertsRecurrentConfigFactory:
    def __init__(
        self,
        *,
        recurrent_controller_options: ExpertsRecurrentControllerOptions,
        gate_config_factory: ExpertsGateConfigFactory,
        halting_config_factory: ExpertsHaltingConfigFactory,
    ) -> None:
        self.recurrent_controller_options = recurrent_controller_options
        self.gate_config_factory = gate_config_factory
        self.halting_config_factory = halting_config_factory

    def build_config(
        self, block_config: ConfigBase
    ) -> ConfigBase | RecurrentLayerConfig:
        self.__validate_recurrent_halting_configuration()
        if not self.recurrent_controller_options.recurrent_flag:
            return block_config
        return RecurrentLayerConfig(
            max_steps=self.recurrent_controller_options.recurrent_max_steps,
            gradient_transition_count=(
                self.recurrent_controller_options.recurrent_gradient_transition_count
            ),
            no_gradient_transition_count=(
                self.recurrent_controller_options.recurrent_no_gradient_transition_count
            ),
            initial_iterations=self.recurrent_controller_options.recurrent_initial_iterations,
            iteration_increment=self.recurrent_controller_options.recurrent_iteration_increment,
            forward_calls_before_iteration_increment=(
                self.recurrent_controller_options.recurrent_forward_calls_before_iteration_increment
            ),
            smooth_iteration_growth_flag=(
                self.recurrent_controller_options.recurrent_smooth_iteration_growth_flag
            ),
            recurrent_layer_norm_position=(
                self.recurrent_controller_options.recurrent_layer_norm_position
            ),
            recurrent_normalization=(
                self.recurrent_controller_options.recurrent_normalization
            ),
            block_config=block_config,
            gate_config=self.gate_config_factory.build_recurrent_gate_config(),
            residual_config=build_residual_config(
                self.recurrent_controller_options.recurrent_residual_connection_option,
                self.recurrent_controller_options.recurrent_residual_model_flag,
                self.recurrent_controller_options.residual_stack_options,
            ),
            halting_config=self.halting_config_factory.build_recurrent_halting_config(),
        )

    def __validate_recurrent_halting_configuration(self) -> None:
        controller = self.recurrent_controller_options
        min_steps = controller.recurrent_min_steps
        if isinstance(min_steps, bool) or not isinstance(min_steps, int):
            raise TypeError(
                "recurrent_min_steps must be an integer, "
                f"received {type(min_steps).__name__}."
            )
        if min_steps < 1:
            raise ValueError(
                "recurrent_min_steps must be greater than or equal to 1, "
                f"received {min_steps}."
            )
        max_steps = controller.recurrent_max_steps
        if (
            isinstance(max_steps, int)
            and not isinstance(max_steps, bool)
            and min_steps > max_steps
        ):
            raise ValueError(
                "recurrent_min_steps must be less than or equal to "
                f"recurrent_max_steps, received {min_steps} and {max_steps}."
            )

        ponder_cost_weight = controller.recurrent_ponder_cost_weight
        if isinstance(ponder_cost_weight, bool) or not isinstance(
            ponder_cost_weight,
            Real,
        ):
            raise TypeError(
                "recurrent_ponder_cost_weight must be a number, "
                f"received {type(ponder_cost_weight).__name__}."
            )
        if not math.isfinite(float(ponder_cost_weight)) or ponder_cost_weight < 0:
            raise ValueError(
                "recurrent_ponder_cost_weight must be finite and greater than or "
                f"equal to 0, received {ponder_cost_weight}."
            )

        minimum_is_default = min_steps == 1
        ponder_weight_is_default = ponder_cost_weight == 1.0
        if minimum_is_default and ponder_weight_is_default:
            return
        if not controller.recurrent_flag:
            configured_field = (
                "recurrent_min_steps"
                if not minimum_is_default
                else "recurrent_ponder_cost_weight"
            )
            raise ValueError(
                f"{configured_field} requires recurrent_flag=True when configured "
                "away from its default."
            )
        if not controller.recurrent_stack_halting_flag:
            configured_field = (
                "recurrent_min_steps"
                if not minimum_is_default
                else "recurrent_ponder_cost_weight"
            )
            raise ValueError(
                f"{configured_field} requires recurrent_stack_halting_flag=True "
                "when configured away from its default."
            )

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum, auto
from types import MappingProxyType, ModuleType

from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
    ResidualConfig,
)
from models.vit.expert_linear_adaptive.runtime_options import (
    ExpertsDynamicMemoryOptions,
    ExpertsLayerControllerOptions,
    ExpertsMixtureOptions,
    ExpertsRecurrentControllerOptions,
    ExpertsRouterOptions,
    ExpertsSamplerOptions,
    ExpertsSubmoduleStackOptions,
    ExpertsSubmoduleStackSource,
)


class ExpertStackRole(Enum):
    MAIN = auto()
    EXPERT = auto()
    ROUTER = auto()


class ExpertControlRole(Enum):
    MAIN = auto()
    EXPERT = auto()
    ROUTER = auto()


class ControllerStackRole(Enum):
    GATE = auto()
    HALTING = auto()
    MEMORY = auto()
    RECURRENT_GATE = auto()
    RECURRENT_HALTING = auto()


@dataclass(frozen=True, slots=True)
class MainControllerStackDefaults:
    independent_flag: bool
    hidden_dim: int | None
    num_layers: int | None
    last_layer_bias_option: LastLayerBiasOptions | None
    apply_output_postprocessing_flag: bool | None
    activation: ActivationOptions | None
    layer_norm_position: LayerNormPositionOptions | None
    residual_connection_option: type[ResidualConfig] | None
    residual_model_flag: bool
    dropout_probability: float | None
    bias_flag: bool | None


_StackOptionsFactory = Callable[[ModuleType], ExpertsSubmoduleStackOptions]
_StackSourceFactory = Callable[[ModuleType], ExpertsSubmoduleStackSource]
_LayerControllerFactory = Callable[[ModuleType], ExpertsLayerControllerOptions]
_DynamicMemoryFactory = Callable[[ModuleType], ExpertsDynamicMemoryOptions]
_RecurrentControllerFactory = Callable[[ModuleType], ExpertsRecurrentControllerOptions]


_STACK_OPTIONS_FACTORIES: Mapping[ExpertStackRole, _StackOptionsFactory] = (
    MappingProxyType(
        {
            ExpertStackRole.MAIN: lambda config: ExpertsSubmoduleStackOptions(
                hidden_dim=config.SUBMODULE_STACK_HIDDEN_DIM,
                num_layers=config.SUBMODULE_STACK_NUM_LAYERS,
                last_layer_bias_option=config.SUBMODULE_STACK_LAST_LAYER_BIAS_OPTION,
                apply_output_postprocessing_flag=config.SUBMODULE_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG,
                activation=config.SUBMODULE_STACK_ACTIVATION,
                layer_norm_position=config.SUBMODULE_STACK_LAYER_NORM_POSITION,
                residual_connection_option=config.SUBMODULE_STACK_RESIDUAL_CONNECTION_OPTION,
                residual_model_flag=config.SUBMODULE_STACK_RESIDUAL_MODEL_FLAG,
                dropout_probability=config.SUBMODULE_STACK_DROPOUT_PROBABILITY,
                bias_flag=config.SUBMODULE_STACK_BIAS_FLAG,
            ),
            ExpertStackRole.EXPERT: lambda config: ExpertsSubmoduleStackOptions(
                hidden_dim=config.EXPERT_STACK_HIDDEN_DIM,
                num_layers=config.EXPERT_STACK_NUM_LAYERS,
                last_layer_bias_option=config.EXPERT_STACK_LAST_LAYER_BIAS_OPTION,
                apply_output_postprocessing_flag=config.EXPERT_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG,
                activation=config.EXPERT_STACK_ACTIVATION,
                layer_norm_position=config.EXPERT_STACK_LAYER_NORM_POSITION,
                residual_connection_option=config.EXPERT_STACK_RESIDUAL_CONNECTION_OPTION,
                residual_model_flag=config.EXPERT_STACK_RESIDUAL_MODEL_FLAG,
                dropout_probability=config.EXPERT_STACK_DROPOUT_PROBABILITY,
                bias_flag=config.EXPERT_BIAS_FLAG,
            ),
            ExpertStackRole.ROUTER: lambda config: ExpertsSubmoduleStackOptions(
                hidden_dim=config.ROUTER_STACK_HIDDEN_DIM,
                num_layers=config.ROUTER_STACK_NUM_LAYERS,
                last_layer_bias_option=config.ROUTER_STACK_LAST_LAYER_BIAS_OPTION,
                apply_output_postprocessing_flag=config.ROUTER_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG,
                activation=config.ROUTER_STACK_ACTIVATION,
                layer_norm_position=config.ROUTER_STACK_LAYER_NORM_POSITION,
                residual_connection_option=config.ROUTER_STACK_RESIDUAL_CONNECTION_OPTION,
                residual_model_flag=config.ROUTER_STACK_RESIDUAL_MODEL_FLAG,
                dropout_probability=config.ROUTER_STACK_DROPOUT_PROBABILITY,
                bias_flag=config.ROUTER_BIAS_FLAG,
            ),
        }
    )
)


def experts_submodule_stack_options(
    config: ModuleType, role: ExpertStackRole
) -> ExpertsSubmoduleStackOptions:
    return _STACK_OPTIONS_FACTORIES[role](config)


def main_controller_stack_defaults(
    config: ModuleType,
    stack_role: ControllerStackRole,
) -> MainControllerStackDefaults:
    if stack_role is ControllerStackRole.GATE:
        return MainControllerStackDefaults(
            independent_flag=config.GATE_STACK_INDEPENDENT_FLAG,
            hidden_dim=config.GATE_STACK_HIDDEN_DIM,
            num_layers=config.GATE_STACK_NUM_LAYERS,
            last_layer_bias_option=config.GATE_STACK_LAST_LAYER_BIAS_OPTION,
            apply_output_postprocessing_flag=config.GATE_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG,
            activation=config.GATE_STACK_ACTIVATION,
            layer_norm_position=config.GATE_STACK_LAYER_NORM_POSITION,
            residual_connection_option=config.GATE_STACK_RESIDUAL_CONNECTION_OPTION,
            residual_model_flag=config.GATE_STACK_RESIDUAL_MODEL_FLAG,
            dropout_probability=config.GATE_STACK_DROPOUT_PROBABILITY,
            bias_flag=config.GATE_STACK_BIAS_FLAG,
        )
    if stack_role is ControllerStackRole.HALTING:
        return MainControllerStackDefaults(
            independent_flag=config.HALTING_STACK_INDEPENDENT_FLAG,
            hidden_dim=config.HALTING_STACK_HIDDEN_DIM,
            num_layers=config.HALTING_STACK_NUM_LAYERS,
            last_layer_bias_option=config.HALTING_STACK_LAST_LAYER_BIAS_OPTION,
            apply_output_postprocessing_flag=config.HALTING_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG,
            activation=config.HALTING_STACK_ACTIVATION,
            layer_norm_position=config.HALTING_STACK_LAYER_NORM_POSITION,
            residual_connection_option=config.HALTING_STACK_RESIDUAL_CONNECTION_OPTION,
            residual_model_flag=config.HALTING_STACK_RESIDUAL_MODEL_FLAG,
            dropout_probability=config.HALTING_STACK_DROPOUT_PROBABILITY,
            bias_flag=config.HALTING_STACK_BIAS_FLAG,
        )
    if stack_role is ControllerStackRole.MEMORY:
        return MainControllerStackDefaults(
            independent_flag=config.MEMORY_STACK_INDEPENDENT_FLAG,
            hidden_dim=config.MEMORY_STACK_HIDDEN_DIM,
            num_layers=config.MEMORY_STACK_NUM_LAYERS,
            last_layer_bias_option=config.MEMORY_STACK_LAST_LAYER_BIAS_OPTION,
            apply_output_postprocessing_flag=config.MEMORY_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG,
            activation=config.MEMORY_STACK_ACTIVATION,
            layer_norm_position=config.MEMORY_STACK_LAYER_NORM_POSITION,
            residual_connection_option=config.MEMORY_STACK_RESIDUAL_CONNECTION_OPTION,
            residual_model_flag=config.MEMORY_STACK_RESIDUAL_MODEL_FLAG,
            dropout_probability=config.MEMORY_STACK_DROPOUT_PROBABILITY,
            bias_flag=config.MEMORY_STACK_BIAS_FLAG,
        )
    if stack_role is ControllerStackRole.RECURRENT_GATE:
        return MainControllerStackDefaults(
            independent_flag=config.RECURRENT_GATE_STACK_INDEPENDENT_FLAG,
            hidden_dim=config.RECURRENT_GATE_STACK_HIDDEN_DIM,
            num_layers=config.RECURRENT_GATE_STACK_NUM_LAYERS,
            last_layer_bias_option=config.RECURRENT_GATE_STACK_LAST_LAYER_BIAS_OPTION,
            apply_output_postprocessing_flag=config.RECURRENT_GATE_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG,
            activation=config.RECURRENT_GATE_STACK_ACTIVATION,
            layer_norm_position=config.RECURRENT_GATE_STACK_LAYER_NORM_POSITION,
            residual_connection_option=config.RECURRENT_GATE_STACK_RESIDUAL_CONNECTION_OPTION,
            residual_model_flag=config.RECURRENT_GATE_STACK_RESIDUAL_MODEL_FLAG,
            dropout_probability=config.RECURRENT_GATE_STACK_DROPOUT_PROBABILITY,
            bias_flag=config.RECURRENT_GATE_STACK_BIAS_FLAG,
        )
    return MainControllerStackDefaults(
        independent_flag=config.RECURRENT_HALTING_STACK_INDEPENDENT_FLAG,
        hidden_dim=config.RECURRENT_HALTING_STACK_HIDDEN_DIM,
        num_layers=config.RECURRENT_HALTING_STACK_NUM_LAYERS,
        last_layer_bias_option=config.RECURRENT_HALTING_STACK_LAST_LAYER_BIAS_OPTION,
        apply_output_postprocessing_flag=config.RECURRENT_HALTING_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG,
        activation=config.RECURRENT_HALTING_STACK_ACTIVATION,
        layer_norm_position=config.RECURRENT_HALTING_STACK_LAYER_NORM_POSITION,
        residual_connection_option=config.RECURRENT_HALTING_STACK_RESIDUAL_CONNECTION_OPTION,
        residual_model_flag=config.RECURRENT_HALTING_STACK_RESIDUAL_MODEL_FLAG,
        dropout_probability=config.RECURRENT_HALTING_STACK_DROPOUT_PROBABILITY,
        bias_flag=config.RECURRENT_HALTING_STACK_BIAS_FLAG,
    )


def _experts_submodule_stack_source(
    defaults: MainControllerStackDefaults,
) -> ExpertsSubmoduleStackSource:
    return ExpertsSubmoduleStackSource(
        independent_flag=defaults.independent_flag,
        hidden_dim=defaults.hidden_dim,
        num_layers=defaults.num_layers,
        last_layer_bias_option=defaults.last_layer_bias_option,
        apply_output_postprocessing_flag=defaults.apply_output_postprocessing_flag,
        activation=defaults.activation,
        layer_norm_position=defaults.layer_norm_position,
        residual_connection_option=defaults.residual_connection_option,
        residual_model_flag=defaults.residual_model_flag,
        dropout_probability=defaults.dropout_probability,
        bias_flag=defaults.bias_flag,
    )


_STACK_SOURCE_FACTORIES: Mapping[
    tuple[ExpertControlRole, ControllerStackRole], _StackSourceFactory
] = MappingProxyType(
    {
        (
            ExpertControlRole.EXPERT,
            ControllerStackRole.GATE,
        ): lambda config: ExpertsSubmoduleStackSource(
            independent_flag=config.EXPERT_GATE_STACK_INDEPENDENT_FLAG,
            hidden_dim=config.EXPERT_GATE_STACK_HIDDEN_DIM,
            num_layers=config.EXPERT_GATE_STACK_NUM_LAYERS,
            last_layer_bias_option=config.EXPERT_GATE_STACK_LAST_LAYER_BIAS_OPTION,
            apply_output_postprocessing_flag=config.EXPERT_GATE_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG,
            activation=config.EXPERT_GATE_STACK_ACTIVATION,
            layer_norm_position=config.EXPERT_GATE_STACK_LAYER_NORM_POSITION,
            residual_connection_option=config.EXPERT_GATE_STACK_RESIDUAL_CONNECTION_OPTION,
            residual_model_flag=config.EXPERT_GATE_STACK_RESIDUAL_MODEL_FLAG,
            dropout_probability=config.EXPERT_GATE_STACK_DROPOUT_PROBABILITY,
            bias_flag=config.EXPERT_GATE_STACK_BIAS_FLAG,
        ),
        (
            ExpertControlRole.EXPERT,
            ControllerStackRole.HALTING,
        ): lambda config: ExpertsSubmoduleStackSource(
            independent_flag=config.EXPERT_HALTING_STACK_INDEPENDENT_FLAG,
            hidden_dim=config.EXPERT_HALTING_STACK_HIDDEN_DIM,
            num_layers=config.EXPERT_HALTING_STACK_NUM_LAYERS,
            last_layer_bias_option=config.EXPERT_HALTING_STACK_LAST_LAYER_BIAS_OPTION,
            apply_output_postprocessing_flag=config.EXPERT_HALTING_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG,
            activation=config.EXPERT_HALTING_STACK_ACTIVATION,
            layer_norm_position=config.EXPERT_HALTING_STACK_LAYER_NORM_POSITION,
            residual_connection_option=config.EXPERT_HALTING_STACK_RESIDUAL_CONNECTION_OPTION,
            residual_model_flag=config.EXPERT_HALTING_STACK_RESIDUAL_MODEL_FLAG,
            dropout_probability=config.EXPERT_HALTING_STACK_DROPOUT_PROBABILITY,
            bias_flag=config.EXPERT_HALTING_STACK_BIAS_FLAG,
        ),
        (
            ExpertControlRole.EXPERT,
            ControllerStackRole.MEMORY,
        ): lambda config: ExpertsSubmoduleStackSource(
            independent_flag=config.EXPERT_MEMORY_STACK_INDEPENDENT_FLAG,
            hidden_dim=config.EXPERT_MEMORY_STACK_HIDDEN_DIM,
            num_layers=config.EXPERT_MEMORY_STACK_NUM_LAYERS,
            last_layer_bias_option=config.EXPERT_MEMORY_STACK_LAST_LAYER_BIAS_OPTION,
            apply_output_postprocessing_flag=config.EXPERT_MEMORY_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG,
            activation=config.EXPERT_MEMORY_STACK_ACTIVATION,
            layer_norm_position=config.EXPERT_MEMORY_STACK_LAYER_NORM_POSITION,
            residual_connection_option=config.EXPERT_MEMORY_STACK_RESIDUAL_CONNECTION_OPTION,
            residual_model_flag=config.EXPERT_MEMORY_STACK_RESIDUAL_MODEL_FLAG,
            dropout_probability=config.EXPERT_MEMORY_STACK_DROPOUT_PROBABILITY,
            bias_flag=config.EXPERT_MEMORY_STACK_BIAS_FLAG,
        ),
        (
            ExpertControlRole.EXPERT,
            ControllerStackRole.RECURRENT_GATE,
        ): lambda config: ExpertsSubmoduleStackSource(
            independent_flag=config.EXPERT_RECURRENT_GATE_STACK_INDEPENDENT_FLAG,
            hidden_dim=config.EXPERT_RECURRENT_GATE_STACK_HIDDEN_DIM,
            num_layers=config.EXPERT_RECURRENT_GATE_STACK_NUM_LAYERS,
            last_layer_bias_option=config.EXPERT_RECURRENT_GATE_STACK_LAST_LAYER_BIAS_OPTION,
            apply_output_postprocessing_flag=config.EXPERT_RECURRENT_GATE_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG,
            activation=config.EXPERT_RECURRENT_GATE_STACK_ACTIVATION,
            layer_norm_position=config.EXPERT_RECURRENT_GATE_STACK_LAYER_NORM_POSITION,
            residual_connection_option=config.EXPERT_RECURRENT_GATE_STACK_RESIDUAL_CONNECTION_OPTION,
            residual_model_flag=config.EXPERT_RECURRENT_GATE_STACK_RESIDUAL_MODEL_FLAG,
            dropout_probability=config.EXPERT_RECURRENT_GATE_STACK_DROPOUT_PROBABILITY,
            bias_flag=config.EXPERT_RECURRENT_GATE_STACK_BIAS_FLAG,
        ),
        (
            ExpertControlRole.EXPERT,
            ControllerStackRole.RECURRENT_HALTING,
        ): lambda config: ExpertsSubmoduleStackSource(
            independent_flag=config.EXPERT_RECURRENT_HALTING_STACK_INDEPENDENT_FLAG,
            hidden_dim=config.EXPERT_RECURRENT_HALTING_STACK_HIDDEN_DIM,
            num_layers=config.EXPERT_RECURRENT_HALTING_STACK_NUM_LAYERS,
            last_layer_bias_option=config.EXPERT_RECURRENT_HALTING_STACK_LAST_LAYER_BIAS_OPTION,
            apply_output_postprocessing_flag=config.EXPERT_RECURRENT_HALTING_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG,
            activation=config.EXPERT_RECURRENT_HALTING_STACK_ACTIVATION,
            layer_norm_position=config.EXPERT_RECURRENT_HALTING_STACK_LAYER_NORM_POSITION,
            residual_connection_option=config.EXPERT_RECURRENT_HALTING_STACK_RESIDUAL_CONNECTION_OPTION,
            residual_model_flag=config.EXPERT_RECURRENT_HALTING_STACK_RESIDUAL_MODEL_FLAG,
            dropout_probability=config.EXPERT_RECURRENT_HALTING_STACK_DROPOUT_PROBABILITY,
            bias_flag=config.EXPERT_RECURRENT_HALTING_STACK_BIAS_FLAG,
        ),
        (
            ExpertControlRole.ROUTER,
            ControllerStackRole.GATE,
        ): lambda config: ExpertsSubmoduleStackSource(
            independent_flag=config.ROUTER_GATE_STACK_INDEPENDENT_FLAG,
            hidden_dim=config.ROUTER_GATE_STACK_HIDDEN_DIM,
            num_layers=config.ROUTER_GATE_STACK_NUM_LAYERS,
            last_layer_bias_option=config.ROUTER_GATE_STACK_LAST_LAYER_BIAS_OPTION,
            apply_output_postprocessing_flag=config.ROUTER_GATE_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG,
            activation=config.ROUTER_GATE_STACK_ACTIVATION,
            layer_norm_position=config.ROUTER_GATE_STACK_LAYER_NORM_POSITION,
            residual_connection_option=config.ROUTER_GATE_STACK_RESIDUAL_CONNECTION_OPTION,
            residual_model_flag=config.ROUTER_GATE_STACK_RESIDUAL_MODEL_FLAG,
            dropout_probability=config.ROUTER_GATE_STACK_DROPOUT_PROBABILITY,
            bias_flag=config.ROUTER_GATE_STACK_BIAS_FLAG,
        ),
        (
            ExpertControlRole.ROUTER,
            ControllerStackRole.HALTING,
        ): lambda config: ExpertsSubmoduleStackSource(
            independent_flag=config.ROUTER_HALTING_STACK_INDEPENDENT_FLAG,
            hidden_dim=config.ROUTER_HALTING_STACK_HIDDEN_DIM,
            num_layers=config.ROUTER_HALTING_STACK_NUM_LAYERS,
            last_layer_bias_option=config.ROUTER_HALTING_STACK_LAST_LAYER_BIAS_OPTION,
            apply_output_postprocessing_flag=config.ROUTER_HALTING_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG,
            activation=config.ROUTER_HALTING_STACK_ACTIVATION,
            layer_norm_position=config.ROUTER_HALTING_STACK_LAYER_NORM_POSITION,
            residual_connection_option=config.ROUTER_HALTING_STACK_RESIDUAL_CONNECTION_OPTION,
            residual_model_flag=config.ROUTER_HALTING_STACK_RESIDUAL_MODEL_FLAG,
            dropout_probability=config.ROUTER_HALTING_STACK_DROPOUT_PROBABILITY,
            bias_flag=config.ROUTER_HALTING_STACK_BIAS_FLAG,
        ),
        (
            ExpertControlRole.ROUTER,
            ControllerStackRole.MEMORY,
        ): lambda config: ExpertsSubmoduleStackSource(
            independent_flag=config.ROUTER_MEMORY_STACK_INDEPENDENT_FLAG,
            hidden_dim=config.ROUTER_MEMORY_STACK_HIDDEN_DIM,
            num_layers=config.ROUTER_MEMORY_STACK_NUM_LAYERS,
            last_layer_bias_option=config.ROUTER_MEMORY_STACK_LAST_LAYER_BIAS_OPTION,
            apply_output_postprocessing_flag=config.ROUTER_MEMORY_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG,
            activation=config.ROUTER_MEMORY_STACK_ACTIVATION,
            layer_norm_position=config.ROUTER_MEMORY_STACK_LAYER_NORM_POSITION,
            residual_connection_option=config.ROUTER_MEMORY_STACK_RESIDUAL_CONNECTION_OPTION,
            residual_model_flag=config.ROUTER_MEMORY_STACK_RESIDUAL_MODEL_FLAG,
            dropout_probability=config.ROUTER_MEMORY_STACK_DROPOUT_PROBABILITY,
            bias_flag=config.ROUTER_MEMORY_STACK_BIAS_FLAG,
        ),
        (
            ExpertControlRole.ROUTER,
            ControllerStackRole.RECURRENT_GATE,
        ): lambda config: ExpertsSubmoduleStackSource(
            independent_flag=config.ROUTER_RECURRENT_GATE_STACK_INDEPENDENT_FLAG,
            hidden_dim=config.ROUTER_RECURRENT_GATE_STACK_HIDDEN_DIM,
            num_layers=config.ROUTER_RECURRENT_GATE_STACK_NUM_LAYERS,
            last_layer_bias_option=config.ROUTER_RECURRENT_GATE_STACK_LAST_LAYER_BIAS_OPTION,
            apply_output_postprocessing_flag=config.ROUTER_RECURRENT_GATE_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG,
            activation=config.ROUTER_RECURRENT_GATE_STACK_ACTIVATION,
            layer_norm_position=config.ROUTER_RECURRENT_GATE_STACK_LAYER_NORM_POSITION,
            residual_connection_option=config.ROUTER_RECURRENT_GATE_STACK_RESIDUAL_CONNECTION_OPTION,
            residual_model_flag=config.ROUTER_RECURRENT_GATE_STACK_RESIDUAL_MODEL_FLAG,
            dropout_probability=config.ROUTER_RECURRENT_GATE_STACK_DROPOUT_PROBABILITY,
            bias_flag=config.ROUTER_RECURRENT_GATE_STACK_BIAS_FLAG,
        ),
        (
            ExpertControlRole.ROUTER,
            ControllerStackRole.RECURRENT_HALTING,
        ): lambda config: ExpertsSubmoduleStackSource(
            independent_flag=config.ROUTER_RECURRENT_HALTING_STACK_INDEPENDENT_FLAG,
            hidden_dim=config.ROUTER_RECURRENT_HALTING_STACK_HIDDEN_DIM,
            num_layers=config.ROUTER_RECURRENT_HALTING_STACK_NUM_LAYERS,
            last_layer_bias_option=config.ROUTER_RECURRENT_HALTING_STACK_LAST_LAYER_BIAS_OPTION,
            apply_output_postprocessing_flag=config.ROUTER_RECURRENT_HALTING_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG,
            activation=config.ROUTER_RECURRENT_HALTING_STACK_ACTIVATION,
            layer_norm_position=config.ROUTER_RECURRENT_HALTING_STACK_LAYER_NORM_POSITION,
            residual_connection_option=config.ROUTER_RECURRENT_HALTING_STACK_RESIDUAL_CONNECTION_OPTION,
            residual_model_flag=config.ROUTER_RECURRENT_HALTING_STACK_RESIDUAL_MODEL_FLAG,
            dropout_probability=config.ROUTER_RECURRENT_HALTING_STACK_DROPOUT_PROBABILITY,
            bias_flag=config.ROUTER_RECURRENT_HALTING_STACK_BIAS_FLAG,
        ),
    }
)


def experts_submodule_stack_source(
    config: ModuleType,
    role: ExpertControlRole,
    stack_role: ControllerStackRole,
) -> ExpertsSubmoduleStackSource:
    if role is ExpertControlRole.MAIN:
        if not isinstance(stack_role, ControllerStackRole):
            raise KeyError((role, stack_role))
        return _experts_submodule_stack_source(
            main_controller_stack_defaults(config, stack_role)
        )
    return _STACK_SOURCE_FACTORIES[role, stack_role](config)


_LAYER_CONTROLLER_FACTORIES: Mapping[ExpertControlRole, _LayerControllerFactory] = (
    MappingProxyType(
        {
            ExpertControlRole.MAIN: lambda config: ExpertsLayerControllerOptions(
                stack_gate_flag=config.STACK_GATE_FLAG,
                gate_option=config.GATE_OPTION,
                gate_activation=config.GATE_ACTIVATION,
                gate_stack_source=experts_submodule_stack_source(
                    config, ExpertControlRole.MAIN, ControllerStackRole.GATE
                ),
                stack_halting_flag=config.STACK_HALTING_FLAG,
                halting_option=config.HALTING_OPTION,
                halting_threshold=config.HALTING_THRESHOLD,
                halting_dropout=config.HALTING_DROPOUT,
                halting_hidden_state_mode=config.HALTING_HIDDEN_STATE_MODE,
                halting_stack_source=experts_submodule_stack_source(
                    config, ExpertControlRole.MAIN, ControllerStackRole.HALTING
                ),
                halting_output_dim=config.HALTING_OUTPUT_DIM,
            ),
            ExpertControlRole.EXPERT: lambda config: ExpertsLayerControllerOptions(
                stack_gate_flag=config.EXPERT_STACK_GATE_FLAG,
                gate_option=config.EXPERT_GATE_OPTION,
                gate_activation=config.EXPERT_GATE_ACTIVATION,
                gate_stack_source=experts_submodule_stack_source(
                    config, ExpertControlRole.EXPERT, ControllerStackRole.GATE
                ),
                stack_halting_flag=config.EXPERT_STACK_HALTING_FLAG,
                halting_option=config.EXPERT_HALTING_OPTION,
                halting_threshold=config.EXPERT_HALTING_THRESHOLD,
                halting_dropout=config.EXPERT_HALTING_DROPOUT,
                halting_hidden_state_mode=config.EXPERT_HALTING_HIDDEN_STATE_MODE,
                halting_stack_source=experts_submodule_stack_source(
                    config, ExpertControlRole.EXPERT, ControllerStackRole.HALTING
                ),
                halting_output_dim=config.EXPERT_HALTING_OUTPUT_DIM,
            ),
            ExpertControlRole.ROUTER: lambda config: ExpertsLayerControllerOptions(
                stack_gate_flag=config.ROUTER_STACK_GATE_FLAG,
                gate_option=config.ROUTER_GATE_OPTION,
                gate_activation=config.ROUTER_GATE_ACTIVATION,
                gate_stack_source=experts_submodule_stack_source(
                    config, ExpertControlRole.ROUTER, ControllerStackRole.GATE
                ),
                stack_halting_flag=config.ROUTER_STACK_HALTING_FLAG,
                halting_option=config.ROUTER_HALTING_OPTION,
                halting_threshold=config.ROUTER_HALTING_THRESHOLD,
                halting_dropout=config.ROUTER_HALTING_DROPOUT,
                halting_hidden_state_mode=config.ROUTER_HALTING_HIDDEN_STATE_MODE,
                halting_stack_source=experts_submodule_stack_source(
                    config, ExpertControlRole.ROUTER, ControllerStackRole.HALTING
                ),
                halting_output_dim=config.ROUTER_HALTING_OUTPUT_DIM,
            ),
        }
    )
)


def experts_layer_controller_options(
    config: ModuleType, role: ExpertControlRole
) -> ExpertsLayerControllerOptions:
    return _LAYER_CONTROLLER_FACTORIES[role](config)


_DYNAMIC_MEMORY_FACTORIES: Mapping[ExpertControlRole, _DynamicMemoryFactory] = (
    MappingProxyType(
        {
            ExpertControlRole.MAIN: lambda config: ExpertsDynamicMemoryOptions(
                memory_flag=config.MEMORY_FLAG,
                memory_option=config.MEMORY_OPTION,
                memory_position_option=config.MEMORY_POSITION_OPTION,
                memory_test_time_training_learning_rate=config.MEMORY_TEST_TIME_TRAINING_LEARNING_RATE,
                memory_test_time_training_num_inner_steps=config.MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS,
                memory_stack_source=experts_submodule_stack_source(
                    config, ExpertControlRole.MAIN, ControllerStackRole.MEMORY
                ),
            ),
            ExpertControlRole.EXPERT: lambda config: ExpertsDynamicMemoryOptions(
                memory_flag=config.EXPERT_MEMORY_FLAG,
                memory_option=config.EXPERT_MEMORY_OPTION,
                memory_position_option=config.EXPERT_MEMORY_POSITION_OPTION,
                memory_test_time_training_learning_rate=config.EXPERT_MEMORY_TEST_TIME_TRAINING_LEARNING_RATE,
                memory_test_time_training_num_inner_steps=config.EXPERT_MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS,
                memory_stack_source=experts_submodule_stack_source(
                    config, ExpertControlRole.EXPERT, ControllerStackRole.MEMORY
                ),
            ),
            ExpertControlRole.ROUTER: lambda config: ExpertsDynamicMemoryOptions(
                memory_flag=config.ROUTER_MEMORY_FLAG,
                memory_option=config.ROUTER_MEMORY_OPTION,
                memory_position_option=config.ROUTER_MEMORY_POSITION_OPTION,
                memory_test_time_training_learning_rate=config.ROUTER_MEMORY_TEST_TIME_TRAINING_LEARNING_RATE,
                memory_test_time_training_num_inner_steps=config.ROUTER_MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS,
                memory_stack_source=experts_submodule_stack_source(
                    config, ExpertControlRole.ROUTER, ControllerStackRole.MEMORY
                ),
            ),
        }
    )
)


def experts_dynamic_memory_options(
    config: ModuleType, role: ExpertControlRole
) -> ExpertsDynamicMemoryOptions:
    return _DYNAMIC_MEMORY_FACTORIES[role](config)


_RECURRENT_CONTROLLER_FACTORIES: Mapping[
    ExpertControlRole, _RecurrentControllerFactory
] = MappingProxyType(
    {
        ExpertControlRole.MAIN: lambda config: ExpertsRecurrentControllerOptions(
            recurrent_flag=config.RECURRENT_FLAG,
            recurrent_max_steps=config.RECURRENT_MAX_STEPS,
            recurrent_initial_iterations=config.RECURRENT_INITIAL_ITERATIONS,
            recurrent_gradient_transition_count=config.RECURRENT_GRADIENT_TRANSITION_COUNT,
            recurrent_no_gradient_transition_count=config.RECURRENT_NO_GRADIENT_TRANSITION_COUNT,
            recurrent_iteration_increment=config.RECURRENT_ITERATION_INCREMENT,
            recurrent_forward_calls_before_iteration_increment=config.RECURRENT_FORWARD_CALLS_BEFORE_ITERATION_INCREMENT,
            recurrent_layer_norm_position=config.RECURRENT_LAYER_NORM_POSITION,
            recurrent_stack_gate_flag=config.RECURRENT_STACK_GATE_FLAG,
            recurrent_gate_option=config.RECURRENT_GATE_OPTION,
            recurrent_gate_activation=config.RECURRENT_GATE_ACTIVATION,
            recurrent_gate_stack_source=experts_submodule_stack_source(
                config, ExpertControlRole.MAIN, ControllerStackRole.RECURRENT_GATE
            ),
            recurrent_stack_halting_flag=config.RECURRENT_STACK_HALTING_FLAG,
            recurrent_halting_option=config.RECURRENT_HALTING_OPTION,
            recurrent_halting_threshold=config.RECURRENT_HALTING_THRESHOLD,
            recurrent_halting_dropout=config.RECURRENT_HALTING_DROPOUT,
            recurrent_halting_hidden_state_mode=config.RECURRENT_HALTING_HIDDEN_STATE_MODE,
            recurrent_halting_stack_source=experts_submodule_stack_source(
                config,
                ExpertControlRole.MAIN,
                ControllerStackRole.RECURRENT_HALTING,
            ),
        ),
        ExpertControlRole.EXPERT: lambda config: ExpertsRecurrentControllerOptions(
            recurrent_flag=config.EXPERT_RECURRENT_FLAG,
            recurrent_max_steps=config.EXPERT_RECURRENT_MAX_STEPS,
            recurrent_initial_iterations=2,
            recurrent_gradient_transition_count=None,
            recurrent_no_gradient_transition_count=None,
            recurrent_iteration_increment=1,
            recurrent_forward_calls_before_iteration_increment=1,
            recurrent_layer_norm_position=config.EXPERT_RECURRENT_LAYER_NORM_POSITION,
            recurrent_stack_gate_flag=config.EXPERT_RECURRENT_STACK_GATE_FLAG,
            recurrent_gate_option=config.EXPERT_RECURRENT_GATE_OPTION,
            recurrent_gate_activation=config.EXPERT_RECURRENT_GATE_ACTIVATION,
            recurrent_gate_stack_source=experts_submodule_stack_source(
                config, ExpertControlRole.EXPERT, ControllerStackRole.RECURRENT_GATE
            ),
            recurrent_stack_halting_flag=config.EXPERT_RECURRENT_STACK_HALTING_FLAG,
            recurrent_halting_option=config.EXPERT_RECURRENT_HALTING_OPTION,
            recurrent_halting_threshold=config.EXPERT_RECURRENT_HALTING_THRESHOLD,
            recurrent_halting_dropout=config.EXPERT_RECURRENT_HALTING_DROPOUT,
            recurrent_halting_hidden_state_mode=config.EXPERT_RECURRENT_HALTING_HIDDEN_STATE_MODE,
            recurrent_halting_stack_source=experts_submodule_stack_source(
                config,
                ExpertControlRole.EXPERT,
                ControllerStackRole.RECURRENT_HALTING,
            ),
        ),
        ExpertControlRole.ROUTER: lambda config: ExpertsRecurrentControllerOptions(
            recurrent_flag=config.ROUTER_RECURRENT_FLAG,
            recurrent_max_steps=config.ROUTER_RECURRENT_MAX_STEPS,
            recurrent_initial_iterations=2,
            recurrent_gradient_transition_count=None,
            recurrent_no_gradient_transition_count=None,
            recurrent_iteration_increment=1,
            recurrent_forward_calls_before_iteration_increment=1,
            recurrent_layer_norm_position=config.ROUTER_RECURRENT_LAYER_NORM_POSITION,
            recurrent_stack_gate_flag=config.ROUTER_RECURRENT_STACK_GATE_FLAG,
            recurrent_gate_option=config.ROUTER_RECURRENT_GATE_OPTION,
            recurrent_gate_activation=config.ROUTER_RECURRENT_GATE_ACTIVATION,
            recurrent_gate_stack_source=experts_submodule_stack_source(
                config, ExpertControlRole.ROUTER, ControllerStackRole.RECURRENT_GATE
            ),
            recurrent_stack_halting_flag=config.ROUTER_RECURRENT_STACK_HALTING_FLAG,
            recurrent_halting_option=config.ROUTER_RECURRENT_HALTING_OPTION,
            recurrent_halting_threshold=config.ROUTER_RECURRENT_HALTING_THRESHOLD,
            recurrent_halting_dropout=config.ROUTER_RECURRENT_HALTING_DROPOUT,
            recurrent_halting_hidden_state_mode=config.ROUTER_RECURRENT_HALTING_HIDDEN_STATE_MODE,
            recurrent_halting_stack_source=experts_submodule_stack_source(
                config,
                ExpertControlRole.ROUTER,
                ControllerStackRole.RECURRENT_HALTING,
            ),
        ),
    }
)


def experts_recurrent_controller_options(
    config: ModuleType, role: ExpertControlRole
) -> ExpertsRecurrentControllerOptions:
    return _RECURRENT_CONTROLLER_FACTORIES[role](config)


def experts_mixture_options(config: ModuleType) -> ExpertsMixtureOptions:
    return ExpertsMixtureOptions(
        top_k=config.TOP_K,
        num_experts=config.NUM_EXPERTS,
        capacity_factor=config.CAPACITY_FACTOR,
        dropped_token_behavior=config.DROPPED_TOKEN_BEHAVIOR,
        compute_expert_mixture_flag=config.COMPUTE_EXPERT_MIXTURE_FLAG,
        weighted_parameters_flag=config.WEIGHTED_PARAMETERS_FLAG,
        weighting_position_option=config.WEIGHTING_POSITION_OPTION,
        routing_initialization_mode=config.ROUTING_INITIALIZATION_MODE,
    )


def experts_sampler_options(config: ModuleType) -> ExpertsSamplerOptions:
    return ExpertsSamplerOptions(
        threshold=config.SAMPLER_THRESHOLD,
        filter_above_threshold=config.SAMPLER_FILTER_ABOVE_THRESHOLD,
        num_topk_samples=config.SAMPLER_NUM_TOPK_SAMPLES,
        normalize_probabilities_flag=config.SAMPLER_NORMALIZE_PROBABILITIES_FLAG,
        noisy_topk_flag=config.SAMPLER_NOISY_TOPK_FLAG,
        coefficient_of_variation_loss_weight=(
            config.SAMPLER_COEFFICIENT_OF_VARIATION_LOSS_WEIGHT
        ),
        switch_loss_weight=config.SAMPLER_SWITCH_LOSS_WEIGHT,
        zero_centred_loss_weight=config.SAMPLER_ZERO_CENTRED_LOSS_WEIGHT,
        mutual_information_loss_weight=(config.SAMPLER_MUTUAL_INFORMATION_LOSS_WEIGHT),
    )


def experts_router_options(config: ModuleType) -> ExpertsRouterOptions:
    return ExpertsRouterOptions(noisy_topk_flag=config.ROUTER_NOISY_TOPK_FLAG)

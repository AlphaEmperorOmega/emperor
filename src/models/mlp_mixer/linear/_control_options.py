from __future__ import annotations

from dataclasses import dataclass, field

from emperor.halting import HaltingConfig, HaltingHiddenStateModeOptions
from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerGateOptions,
    LayerNormPositionOptions,
    NormalizationOptions,
    ResidualConfig,
)
from emperor.memory import DynamicMemoryConfig, MemoryPositionOptions

from .runtime_options import RuntimeOptions


@dataclass(frozen=True, slots=True)
class StackOptions:
    hidden_dim: int
    num_layers: int
    activation: ActivationOptions
    dropout_probability: float
    layer_norm_position: LayerNormPositionOptions
    normalization: NormalizationOptions = field(
        default=NormalizationOptions.RMS_NORM, kw_only=True
    )
    residual_connection_option: type[ResidualConfig] | None
    residual_model_flag: bool
    last_layer_bias_option: LastLayerBiasOptions
    apply_output_postprocessing_flag: bool
    bias_flag: bool


@dataclass(frozen=True, slots=True)
class ControllerStackSource:
    independent_flag: bool
    hidden_dim: int | None
    num_layers: int | None
    activation: ActivationOptions | None
    dropout_probability: float | None
    layer_norm_position: LayerNormPositionOptions | None
    normalization: NormalizationOptions | None = field(default=None, kw_only=True)
    residual_connection_option: type[ResidualConfig] | None
    residual_model_flag: bool
    last_layer_bias_option: LastLayerBiasOptions | None
    apply_output_postprocessing_flag: bool | None
    bias_flag: bool | None

    def resolve(self, defaults: StackOptions) -> StackOptions:
        if not self.independent_flag:
            return defaults
        return StackOptions(
            hidden_dim=(
                defaults.hidden_dim if self.hidden_dim is None else self.hidden_dim
            ),
            num_layers=(
                defaults.num_layers if self.num_layers is None else self.num_layers
            ),
            activation=(
                defaults.activation if self.activation is None else self.activation
            ),
            dropout_probability=(
                defaults.dropout_probability
                if self.dropout_probability is None
                else self.dropout_probability
            ),
            layer_norm_position=(
                defaults.layer_norm_position
                if self.layer_norm_position is None
                else self.layer_norm_position
            ),
            normalization=(
                defaults.normalization
                if self.normalization is None
                else self.normalization
            ),
            residual_connection_option=(
                defaults.residual_connection_option
                if self.residual_connection_option is None
                else self.residual_connection_option
            ),
            residual_model_flag=self.residual_model_flag,
            last_layer_bias_option=(
                defaults.last_layer_bias_option
                if self.last_layer_bias_option is None
                else self.last_layer_bias_option
            ),
            apply_output_postprocessing_flag=(
                defaults.apply_output_postprocessing_flag
                if self.apply_output_postprocessing_flag is None
                else self.apply_output_postprocessing_flag
            ),
            bias_flag=(
                defaults.bias_flag if self.bias_flag is None else self.bias_flag
            ),
        )


@dataclass(frozen=True, slots=True)
class GateOptions:
    enabled: bool
    option: LayerGateOptions | None
    activation: ActivationOptions | None
    stack: ControllerStackSource


@dataclass(frozen=True, slots=True)
class HaltingOptions:
    enabled: bool
    implementation: type[HaltingConfig]
    threshold: float
    dropout_probability: float
    hidden_state_mode: HaltingHiddenStateModeOptions
    stack: ControllerStackSource


@dataclass(frozen=True, slots=True)
class MemoryOptions:
    enabled: bool
    implementation: type[DynamicMemoryConfig]
    position: MemoryPositionOptions
    test_time_training_learning_rate: float | None
    test_time_training_num_inner_steps: int | None
    stack: ControllerStackSource


@dataclass(frozen=True, slots=True)
class RecurrentOptions:
    enabled: bool
    max_steps: int
    gradient_transition_count: int | None
    no_gradient_transition_count: int | None
    initial_iterations: int
    iteration_increment: int
    forward_calls_before_iteration_increment: int
    smooth_iteration_growth_flag: bool
    layer_norm_position: LayerNormPositionOptions
    normalization: NormalizationOptions = field(
        default=NormalizationOptions.LAYER_NORM, kw_only=True
    )
    residual_connection_option: type[ResidualConfig] | None
    residual_model_flag: bool
    gate: GateOptions
    halting: HaltingOptions
    memory: MemoryOptions | None


@dataclass(frozen=True, slots=True)
class ControlOptions:
    gate: GateOptions
    halting: HaltingOptions
    memory: MemoryOptions
    recurrent: RecurrentOptions


def submodule_stack_options(runtime: RuntimeOptions) -> StackOptions:
    return StackOptions(
        hidden_dim=runtime.submodule_stack_hidden_dim,
        num_layers=runtime.submodule_stack_num_layers,
        activation=runtime.submodule_stack_activation,
        dropout_probability=runtime.submodule_stack_dropout_probability,
        layer_norm_position=runtime.submodule_stack_layer_norm_position,
        normalization=runtime.submodule_stack_normalization,
        residual_connection_option=runtime.submodule_stack_residual_connection_option,
        residual_model_flag=runtime.submodule_stack_residual_model_flag,
        last_layer_bias_option=runtime.submodule_stack_last_layer_bias_option,
        apply_output_postprocessing_flag=runtime.submodule_stack_apply_output_postprocessing_flag,
        bias_flag=runtime.submodule_stack_bias_flag,
    )


def main_control_options(runtime: RuntimeOptions) -> ControlOptions:
    memory = MemoryOptions(
        enabled=runtime.memory_flag,
        implementation=runtime.memory_option,
        position=runtime.memory_position_option,
        test_time_training_learning_rate=(
            runtime.memory_test_time_training_learning_rate
        ),
        test_time_training_num_inner_steps=(
            runtime.memory_test_time_training_num_inner_steps
        ),
        stack=_main_memory_stack(runtime),
    )
    return ControlOptions(
        gate=GateOptions(
            enabled=runtime.stack_gate_flag,
            option=runtime.gate_option,
            activation=runtime.gate_activation,
            stack=_main_gate_stack(runtime),
        ),
        halting=HaltingOptions(
            enabled=runtime.stack_halting_flag,
            implementation=runtime.halting_option,
            threshold=runtime.halting_threshold,
            dropout_probability=runtime.halting_dropout,
            hidden_state_mode=runtime.halting_hidden_state_mode,
            stack=_main_halting_stack(runtime),
        ),
        memory=memory,
        recurrent=RecurrentOptions(
            enabled=runtime.recurrent_flag,
            max_steps=runtime.recurrent_max_steps,
            gradient_transition_count=runtime.recurrent_gradient_transition_count,
            no_gradient_transition_count=runtime.recurrent_no_gradient_transition_count,
            initial_iterations=runtime.recurrent_initial_iterations,
            iteration_increment=runtime.recurrent_iteration_increment,
            forward_calls_before_iteration_increment=(
                runtime.recurrent_forward_calls_before_iteration_increment
            ),
            smooth_iteration_growth_flag=(
                runtime.recurrent_smooth_iteration_growth_flag
            ),
            layer_norm_position=runtime.recurrent_layer_norm_position,
            normalization=runtime.recurrent_normalization,
            residual_connection_option=runtime.recurrent_residual_connection_option,
            residual_model_flag=runtime.recurrent_residual_model_flag,
            gate=GateOptions(
                enabled=runtime.recurrent_stack_gate_flag,
                option=runtime.recurrent_gate_option,
                activation=runtime.recurrent_gate_activation,
                stack=_main_recurrent_gate_stack(runtime),
            ),
            halting=HaltingOptions(
                enabled=runtime.recurrent_stack_halting_flag,
                implementation=runtime.recurrent_halting_option,
                threshold=runtime.recurrent_halting_threshold,
                dropout_probability=runtime.recurrent_halting_dropout,
                hidden_state_mode=runtime.recurrent_halting_hidden_state_mode,
                stack=_main_recurrent_halting_stack(runtime),
            ),
            memory=MemoryOptions(
                enabled=runtime.recurrent_memory_flag,
                implementation=memory.implementation,
                position=memory.position,
                test_time_training_learning_rate=(
                    memory.test_time_training_learning_rate
                ),
                test_time_training_num_inner_steps=(
                    memory.test_time_training_num_inner_steps
                ),
                stack=memory.stack,
            ),
        ),
    )


def token_mixer_control_options(runtime: RuntimeOptions) -> ControlOptions:
    return _token_mixer_control_options(runtime)


def channel_mixer_control_options(runtime: RuntimeOptions) -> ControlOptions:
    return _channel_mixer_control_options(runtime)


def _token_mixer_control_options(runtime: RuntimeOptions) -> ControlOptions:
    return ControlOptions(
        gate=GateOptions(
            enabled=runtime.token_mixer_stack_gate_flag,
            option=runtime.token_mixer_gate_option,
            activation=runtime.token_mixer_gate_activation,
            stack=ControllerStackSource(
                independent_flag=runtime.token_mixer_gate_stack_independent_flag,
                hidden_dim=runtime.token_mixer_gate_stack_hidden_dim,
                num_layers=runtime.token_mixer_gate_stack_num_layers,
                activation=runtime.token_mixer_gate_stack_activation,
                dropout_probability=(
                    runtime.token_mixer_gate_stack_dropout_probability
                ),
                layer_norm_position=(
                    runtime.token_mixer_gate_stack_layer_norm_position
                ),
                normalization=(runtime.token_mixer_gate_stack_normalization),
                residual_connection_option=(
                    runtime.token_mixer_gate_stack_residual_connection_option
                ),
                residual_model_flag=(
                    runtime.token_mixer_gate_stack_residual_model_flag
                ),
                last_layer_bias_option=(
                    runtime.token_mixer_gate_stack_last_layer_bias_option
                ),
                apply_output_postprocessing_flag=(
                    runtime.token_mixer_gate_stack_apply_output_postprocessing_flag
                ),
                bias_flag=runtime.token_mixer_gate_stack_bias_flag,
            ),
        ),
        halting=HaltingOptions(
            enabled=runtime.token_mixer_stack_halting_flag,
            implementation=runtime.token_mixer_halting_option,
            threshold=runtime.token_mixer_halting_threshold,
            dropout_probability=runtime.token_mixer_halting_dropout,
            hidden_state_mode=runtime.token_mixer_halting_hidden_state_mode,
            stack=ControllerStackSource(
                independent_flag=(runtime.token_mixer_halting_stack_independent_flag),
                hidden_dim=runtime.token_mixer_halting_stack_hidden_dim,
                num_layers=runtime.token_mixer_halting_stack_num_layers,
                activation=runtime.token_mixer_halting_stack_activation,
                dropout_probability=(
                    runtime.token_mixer_halting_stack_dropout_probability
                ),
                layer_norm_position=(
                    runtime.token_mixer_halting_stack_layer_norm_position
                ),
                normalization=(runtime.token_mixer_halting_stack_normalization),
                residual_connection_option=(
                    runtime.token_mixer_halting_stack_residual_connection_option
                ),
                residual_model_flag=(
                    runtime.token_mixer_halting_stack_residual_model_flag
                ),
                last_layer_bias_option=(
                    runtime.token_mixer_halting_stack_last_layer_bias_option
                ),
                apply_output_postprocessing_flag=(
                    runtime.token_mixer_halting_stack_apply_output_postprocessing_flag
                ),
                bias_flag=runtime.token_mixer_halting_stack_bias_flag,
            ),
        ),
        memory=MemoryOptions(
            enabled=runtime.token_mixer_memory_flag,
            implementation=runtime.token_mixer_memory_option,
            position=runtime.token_mixer_memory_position_option,
            test_time_training_learning_rate=(
                runtime.token_mixer_memory_test_time_training_learning_rate
            ),
            test_time_training_num_inner_steps=(
                runtime.token_mixer_memory_test_time_training_num_inner_steps
            ),
            stack=ControllerStackSource(
                independent_flag=runtime.token_mixer_memory_stack_independent_flag,
                hidden_dim=runtime.token_mixer_memory_stack_hidden_dim,
                num_layers=runtime.token_mixer_memory_stack_num_layers,
                activation=runtime.token_mixer_memory_stack_activation,
                dropout_probability=(
                    runtime.token_mixer_memory_stack_dropout_probability
                ),
                layer_norm_position=(
                    runtime.token_mixer_memory_stack_layer_norm_position
                ),
                normalization=(runtime.token_mixer_memory_stack_normalization),
                residual_connection_option=(
                    runtime.token_mixer_memory_stack_residual_connection_option
                ),
                residual_model_flag=(
                    runtime.token_mixer_memory_stack_residual_model_flag
                ),
                last_layer_bias_option=(
                    runtime.token_mixer_memory_stack_last_layer_bias_option
                ),
                apply_output_postprocessing_flag=(
                    runtime.token_mixer_memory_stack_apply_output_postprocessing_flag
                ),
                bias_flag=runtime.token_mixer_memory_stack_bias_flag,
            ),
        ),
        recurrent=RecurrentOptions(
            enabled=runtime.token_mixer_recurrent_flag,
            max_steps=runtime.token_mixer_recurrent_max_steps,
            gradient_transition_count=None,
            no_gradient_transition_count=None,
            initial_iterations=runtime.token_mixer_recurrent_max_steps,
            iteration_increment=runtime.recurrent_iteration_increment,
            forward_calls_before_iteration_increment=(
                runtime.recurrent_forward_calls_before_iteration_increment
            ),
            smooth_iteration_growth_flag=False,
            layer_norm_position=runtime.token_mixer_recurrent_layer_norm_position,
            normalization=runtime.token_mixer_recurrent_normalization,
            residual_connection_option=(
                runtime.token_mixer_recurrent_residual_connection_option
            ),
            residual_model_flag=runtime.token_mixer_recurrent_residual_model_flag,
            gate=GateOptions(
                enabled=runtime.token_mixer_recurrent_stack_gate_flag,
                option=runtime.token_mixer_recurrent_gate_option,
                activation=runtime.token_mixer_recurrent_gate_activation,
                stack=ControllerStackSource(
                    independent_flag=(
                        runtime.token_mixer_recurrent_gate_stack_independent_flag
                    ),
                    hidden_dim=runtime.token_mixer_recurrent_gate_stack_hidden_dim,
                    num_layers=runtime.token_mixer_recurrent_gate_stack_num_layers,
                    activation=runtime.token_mixer_recurrent_gate_stack_activation,
                    dropout_probability=(
                        runtime.token_mixer_recurrent_gate_stack_dropout_probability
                    ),
                    layer_norm_position=(
                        runtime.token_mixer_recurrent_gate_stack_layer_norm_position
                    ),
                    normalization=(
                        runtime.token_mixer_recurrent_gate_stack_normalization
                    ),
                    residual_connection_option=(
                        runtime.token_mixer_recurrent_gate_stack_residual_connection_option
                    ),
                    residual_model_flag=(
                        runtime.token_mixer_recurrent_gate_stack_residual_model_flag
                    ),
                    last_layer_bias_option=(
                        runtime.token_mixer_recurrent_gate_stack_last_layer_bias_option
                    ),
                    apply_output_postprocessing_flag=(
                        runtime.token_mixer_recurrent_gate_stack_apply_output_postprocessing_flag
                    ),
                    bias_flag=runtime.token_mixer_recurrent_gate_stack_bias_flag,
                ),
            ),
            halting=HaltingOptions(
                enabled=runtime.token_mixer_recurrent_stack_halting_flag,
                implementation=runtime.token_mixer_recurrent_halting_option,
                threshold=runtime.token_mixer_recurrent_halting_threshold,
                dropout_probability=runtime.token_mixer_recurrent_halting_dropout,
                hidden_state_mode=(
                    runtime.token_mixer_recurrent_halting_hidden_state_mode
                ),
                stack=ControllerStackSource(
                    independent_flag=(
                        runtime.token_mixer_recurrent_halting_stack_independent_flag
                    ),
                    hidden_dim=runtime.token_mixer_recurrent_halting_stack_hidden_dim,
                    num_layers=(runtime.token_mixer_recurrent_halting_stack_num_layers),
                    activation=(runtime.token_mixer_recurrent_halting_stack_activation),
                    dropout_probability=(
                        runtime.token_mixer_recurrent_halting_stack_dropout_probability
                    ),
                    layer_norm_position=(
                        runtime.token_mixer_recurrent_halting_stack_layer_norm_position
                    ),
                    normalization=(
                        runtime.token_mixer_recurrent_halting_stack_normalization
                    ),
                    residual_connection_option=(
                        runtime.token_mixer_recurrent_halting_stack_residual_connection_option
                    ),
                    residual_model_flag=(
                        runtime.token_mixer_recurrent_halting_stack_residual_model_flag
                    ),
                    last_layer_bias_option=(
                        runtime.token_mixer_recurrent_halting_stack_last_layer_bias_option
                    ),
                    apply_output_postprocessing_flag=(
                        runtime.token_mixer_recurrent_halting_stack_apply_output_postprocessing_flag
                    ),
                    bias_flag=(runtime.token_mixer_recurrent_halting_stack_bias_flag),
                ),
            ),
            memory=None,
        ),
    )


def _channel_mixer_control_options(runtime: RuntimeOptions) -> ControlOptions:
    return ControlOptions(
        gate=GateOptions(
            enabled=runtime.channel_mixer_stack_gate_flag,
            option=runtime.channel_mixer_gate_option,
            activation=runtime.channel_mixer_gate_activation,
            stack=ControllerStackSource(
                independent_flag=runtime.channel_mixer_gate_stack_independent_flag,
                hidden_dim=runtime.channel_mixer_gate_stack_hidden_dim,
                num_layers=runtime.channel_mixer_gate_stack_num_layers,
                activation=runtime.channel_mixer_gate_stack_activation,
                dropout_probability=(
                    runtime.channel_mixer_gate_stack_dropout_probability
                ),
                layer_norm_position=(
                    runtime.channel_mixer_gate_stack_layer_norm_position
                ),
                normalization=(runtime.channel_mixer_gate_stack_normalization),
                residual_connection_option=(
                    runtime.channel_mixer_gate_stack_residual_connection_option
                ),
                residual_model_flag=(
                    runtime.channel_mixer_gate_stack_residual_model_flag
                ),
                last_layer_bias_option=(
                    runtime.channel_mixer_gate_stack_last_layer_bias_option
                ),
                apply_output_postprocessing_flag=(
                    runtime.channel_mixer_gate_stack_apply_output_postprocessing_flag
                ),
                bias_flag=runtime.channel_mixer_gate_stack_bias_flag,
            ),
        ),
        halting=HaltingOptions(
            enabled=runtime.channel_mixer_stack_halting_flag,
            implementation=runtime.channel_mixer_halting_option,
            threshold=runtime.channel_mixer_halting_threshold,
            dropout_probability=runtime.channel_mixer_halting_dropout,
            hidden_state_mode=runtime.channel_mixer_halting_hidden_state_mode,
            stack=ControllerStackSource(
                independent_flag=(runtime.channel_mixer_halting_stack_independent_flag),
                hidden_dim=runtime.channel_mixer_halting_stack_hidden_dim,
                num_layers=runtime.channel_mixer_halting_stack_num_layers,
                activation=runtime.channel_mixer_halting_stack_activation,
                dropout_probability=(
                    runtime.channel_mixer_halting_stack_dropout_probability
                ),
                layer_norm_position=(
                    runtime.channel_mixer_halting_stack_layer_norm_position
                ),
                normalization=(runtime.channel_mixer_halting_stack_normalization),
                residual_connection_option=(
                    runtime.channel_mixer_halting_stack_residual_connection_option
                ),
                residual_model_flag=(
                    runtime.channel_mixer_halting_stack_residual_model_flag
                ),
                last_layer_bias_option=(
                    runtime.channel_mixer_halting_stack_last_layer_bias_option
                ),
                apply_output_postprocessing_flag=(
                    runtime.channel_mixer_halting_stack_apply_output_postprocessing_flag
                ),
                bias_flag=runtime.channel_mixer_halting_stack_bias_flag,
            ),
        ),
        memory=MemoryOptions(
            enabled=runtime.channel_mixer_memory_flag,
            implementation=runtime.channel_mixer_memory_option,
            position=runtime.channel_mixer_memory_position_option,
            test_time_training_learning_rate=(
                runtime.channel_mixer_memory_test_time_training_learning_rate
            ),
            test_time_training_num_inner_steps=(
                runtime.channel_mixer_memory_test_time_training_num_inner_steps
            ),
            stack=ControllerStackSource(
                independent_flag=(runtime.channel_mixer_memory_stack_independent_flag),
                hidden_dim=runtime.channel_mixer_memory_stack_hidden_dim,
                num_layers=runtime.channel_mixer_memory_stack_num_layers,
                activation=runtime.channel_mixer_memory_stack_activation,
                dropout_probability=(
                    runtime.channel_mixer_memory_stack_dropout_probability
                ),
                layer_norm_position=(
                    runtime.channel_mixer_memory_stack_layer_norm_position
                ),
                normalization=(runtime.channel_mixer_memory_stack_normalization),
                residual_connection_option=(
                    runtime.channel_mixer_memory_stack_residual_connection_option
                ),
                residual_model_flag=(
                    runtime.channel_mixer_memory_stack_residual_model_flag
                ),
                last_layer_bias_option=(
                    runtime.channel_mixer_memory_stack_last_layer_bias_option
                ),
                apply_output_postprocessing_flag=(
                    runtime.channel_mixer_memory_stack_apply_output_postprocessing_flag
                ),
                bias_flag=runtime.channel_mixer_memory_stack_bias_flag,
            ),
        ),
        recurrent=RecurrentOptions(
            enabled=runtime.channel_mixer_recurrent_flag,
            max_steps=runtime.channel_mixer_recurrent_max_steps,
            gradient_transition_count=None,
            no_gradient_transition_count=None,
            initial_iterations=runtime.channel_mixer_recurrent_max_steps,
            iteration_increment=runtime.recurrent_iteration_increment,
            forward_calls_before_iteration_increment=(
                runtime.recurrent_forward_calls_before_iteration_increment
            ),
            smooth_iteration_growth_flag=False,
            layer_norm_position=runtime.channel_mixer_recurrent_layer_norm_position,
            normalization=runtime.channel_mixer_recurrent_normalization,
            residual_connection_option=(
                runtime.channel_mixer_recurrent_residual_connection_option
            ),
            residual_model_flag=(runtime.channel_mixer_recurrent_residual_model_flag),
            gate=GateOptions(
                enabled=runtime.channel_mixer_recurrent_stack_gate_flag,
                option=runtime.channel_mixer_recurrent_gate_option,
                activation=runtime.channel_mixer_recurrent_gate_activation,
                stack=ControllerStackSource(
                    independent_flag=(
                        runtime.channel_mixer_recurrent_gate_stack_independent_flag
                    ),
                    hidden_dim=runtime.channel_mixer_recurrent_gate_stack_hidden_dim,
                    num_layers=runtime.channel_mixer_recurrent_gate_stack_num_layers,
                    activation=(runtime.channel_mixer_recurrent_gate_stack_activation),
                    dropout_probability=(
                        runtime.channel_mixer_recurrent_gate_stack_dropout_probability
                    ),
                    layer_norm_position=(
                        runtime.channel_mixer_recurrent_gate_stack_layer_norm_position
                    ),
                    normalization=(
                        runtime.channel_mixer_recurrent_gate_stack_normalization
                    ),
                    residual_connection_option=(
                        runtime.channel_mixer_recurrent_gate_stack_residual_connection_option
                    ),
                    residual_model_flag=(
                        runtime.channel_mixer_recurrent_gate_stack_residual_model_flag
                    ),
                    last_layer_bias_option=(
                        runtime.channel_mixer_recurrent_gate_stack_last_layer_bias_option
                    ),
                    apply_output_postprocessing_flag=(
                        runtime.channel_mixer_recurrent_gate_stack_apply_output_postprocessing_flag
                    ),
                    bias_flag=runtime.channel_mixer_recurrent_gate_stack_bias_flag,
                ),
            ),
            halting=HaltingOptions(
                enabled=runtime.channel_mixer_recurrent_stack_halting_flag,
                implementation=runtime.channel_mixer_recurrent_halting_option,
                threshold=runtime.channel_mixer_recurrent_halting_threshold,
                dropout_probability=(runtime.channel_mixer_recurrent_halting_dropout),
                hidden_state_mode=(
                    runtime.channel_mixer_recurrent_halting_hidden_state_mode
                ),
                stack=ControllerStackSource(
                    independent_flag=(
                        runtime.channel_mixer_recurrent_halting_stack_independent_flag
                    ),
                    hidden_dim=(
                        runtime.channel_mixer_recurrent_halting_stack_hidden_dim
                    ),
                    num_layers=(
                        runtime.channel_mixer_recurrent_halting_stack_num_layers
                    ),
                    activation=(
                        runtime.channel_mixer_recurrent_halting_stack_activation
                    ),
                    dropout_probability=(
                        runtime.channel_mixer_recurrent_halting_stack_dropout_probability
                    ),
                    layer_norm_position=(
                        runtime.channel_mixer_recurrent_halting_stack_layer_norm_position
                    ),
                    normalization=(
                        runtime.channel_mixer_recurrent_halting_stack_normalization
                    ),
                    residual_connection_option=(
                        runtime.channel_mixer_recurrent_halting_stack_residual_connection_option
                    ),
                    residual_model_flag=(
                        runtime.channel_mixer_recurrent_halting_stack_residual_model_flag
                    ),
                    last_layer_bias_option=(
                        runtime.channel_mixer_recurrent_halting_stack_last_layer_bias_option
                    ),
                    apply_output_postprocessing_flag=(
                        runtime.channel_mixer_recurrent_halting_stack_apply_output_postprocessing_flag
                    ),
                    bias_flag=(runtime.channel_mixer_recurrent_halting_stack_bias_flag),
                ),
            ),
            memory=None,
        ),
    )


def _main_gate_stack(runtime: RuntimeOptions) -> ControllerStackSource:
    return ControllerStackSource(
        independent_flag=runtime.gate_stack_independent_flag,
        hidden_dim=runtime.gate_stack_hidden_dim,
        num_layers=runtime.gate_stack_num_layers,
        activation=runtime.gate_stack_activation,
        dropout_probability=runtime.gate_stack_dropout_probability,
        layer_norm_position=runtime.gate_stack_layer_norm_position,
        normalization=runtime.gate_stack_normalization,
        residual_connection_option=runtime.gate_stack_residual_connection_option,
        residual_model_flag=runtime.gate_stack_residual_model_flag,
        last_layer_bias_option=runtime.gate_stack_last_layer_bias_option,
        apply_output_postprocessing_flag=runtime.gate_stack_apply_output_postprocessing_flag,
        bias_flag=runtime.gate_stack_bias_flag,
    )


def _main_halting_stack(runtime: RuntimeOptions) -> ControllerStackSource:
    return ControllerStackSource(
        independent_flag=runtime.halting_stack_independent_flag,
        hidden_dim=runtime.halting_stack_hidden_dim,
        num_layers=runtime.halting_stack_num_layers,
        activation=runtime.halting_stack_activation,
        dropout_probability=runtime.halting_stack_dropout_probability,
        layer_norm_position=runtime.halting_stack_layer_norm_position,
        normalization=runtime.halting_stack_normalization,
        residual_connection_option=runtime.halting_stack_residual_connection_option,
        residual_model_flag=runtime.halting_stack_residual_model_flag,
        last_layer_bias_option=runtime.halting_stack_last_layer_bias_option,
        apply_output_postprocessing_flag=runtime.halting_stack_apply_output_postprocessing_flag,
        bias_flag=runtime.halting_stack_bias_flag,
    )


def _main_memory_stack(runtime: RuntimeOptions) -> ControllerStackSource:
    return ControllerStackSource(
        independent_flag=runtime.memory_stack_independent_flag,
        hidden_dim=runtime.memory_stack_hidden_dim,
        num_layers=runtime.memory_stack_num_layers,
        activation=runtime.memory_stack_activation,
        dropout_probability=runtime.memory_stack_dropout_probability,
        layer_norm_position=runtime.memory_stack_layer_norm_position,
        normalization=runtime.memory_stack_normalization,
        residual_connection_option=runtime.memory_stack_residual_connection_option,
        residual_model_flag=runtime.memory_stack_residual_model_flag,
        last_layer_bias_option=runtime.memory_stack_last_layer_bias_option,
        apply_output_postprocessing_flag=runtime.memory_stack_apply_output_postprocessing_flag,
        bias_flag=runtime.memory_stack_bias_flag,
    )


def _main_recurrent_gate_stack(runtime: RuntimeOptions) -> ControllerStackSource:
    return ControllerStackSource(
        independent_flag=runtime.recurrent_gate_stack_independent_flag,
        hidden_dim=runtime.recurrent_gate_stack_hidden_dim,
        num_layers=runtime.recurrent_gate_stack_num_layers,
        activation=runtime.recurrent_gate_stack_activation,
        dropout_probability=runtime.recurrent_gate_stack_dropout_probability,
        layer_norm_position=runtime.recurrent_gate_stack_layer_norm_position,
        normalization=runtime.recurrent_gate_stack_normalization,
        residual_connection_option=(
            runtime.recurrent_gate_stack_residual_connection_option
        ),
        residual_model_flag=runtime.recurrent_gate_stack_residual_model_flag,
        last_layer_bias_option=runtime.recurrent_gate_stack_last_layer_bias_option,
        apply_output_postprocessing_flag=(
            runtime.recurrent_gate_stack_apply_output_postprocessing_flag
        ),
        bias_flag=runtime.recurrent_gate_stack_bias_flag,
    )


def _main_recurrent_halting_stack(runtime: RuntimeOptions) -> ControllerStackSource:
    return ControllerStackSource(
        independent_flag=runtime.recurrent_halting_stack_independent_flag,
        hidden_dim=runtime.recurrent_halting_stack_hidden_dim,
        num_layers=runtime.recurrent_halting_stack_num_layers,
        activation=runtime.recurrent_halting_stack_activation,
        dropout_probability=runtime.recurrent_halting_stack_dropout_probability,
        layer_norm_position=runtime.recurrent_halting_stack_layer_norm_position,
        normalization=runtime.recurrent_halting_stack_normalization,
        residual_connection_option=(
            runtime.recurrent_halting_stack_residual_connection_option
        ),
        residual_model_flag=runtime.recurrent_halting_stack_residual_model_flag,
        last_layer_bias_option=(runtime.recurrent_halting_stack_last_layer_bias_option),
        apply_output_postprocessing_flag=(
            runtime.recurrent_halting_stack_apply_output_postprocessing_flag
        ),
        bias_flag=runtime.recurrent_halting_stack_bias_flag,
    )

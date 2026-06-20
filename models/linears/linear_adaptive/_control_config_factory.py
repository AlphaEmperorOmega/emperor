from typing import Any

from emperor.augmentations.adaptive_parameters.config import (
    AdaptiveParameterAugmentationConfig,
)
from emperor.base.layer.config import (
    LayerConfig,
    LayerStackConfig,
    RecurrentLayerConfig,
)
from emperor.base.layer.gate import GateConfig
from emperor.base.layer.residual import ResidualConnectionOptions
from emperor.base.options import LastLayerBiasOptions
from emperor.halting.config import StickBreakingConfig
from emperor.linears.core.config import AdaptiveLinearLayerConfig
from emperor.memory.config import DynamicMemoryConfig

from models.linears.linear_adaptive._controller_stack import (
    ControllerStackOptions,
    ControllerStackSource,
    build_linear_controller_stack,
    resolve_controller_stack_options,
    resolve_enabled,
)


class ControlConfigFactory:
    def __init__(self, builder: Any) -> None:
        self.builder = builder

    def build(self) -> LayerStackConfig | RecurrentLayerConfig:
        gate_config = self.__build_gate_config()
        self.__validate_shared_gate_config(gate_config)
        halting_config = self.__build_halting_config()
        memory_config = self.__build_memory_config()
        layer_stack_config = self.__build_stack_config(
            gate_config=gate_config,
            halting_config=halting_config,
            memory_config=memory_config,
        )
        return self.__maybe_wrap_recurrent(layer_stack_config)

    def __build_stack_config(
        self,
        *,
        gate_config: GateConfig | None,
        halting_config: StickBreakingConfig | None,
        memory_config: DynamicMemoryConfig | None,
    ) -> LayerStackConfig:
        builder = self.builder
        return LayerStackConfig(
            hidden_dim=builder.hidden_dim,
            num_layers=builder.stack_num_layers,
            last_layer_bias_option=builder.stack_last_layer_bias_option,
            apply_output_pipeline_flag=builder.stack_apply_output_pipeline_flag,
            shared_gate_config=builder.shared_gate_config,
            shared_memory_config=memory_config,
            layer_config=LayerConfig(
                activation=builder.stack_activation,
                layer_norm_position=builder.stack_layer_norm_position,
                residual_connection_option=builder.stack_residual_connection_option,
                dropout_probability=builder.stack_dropout_probability,
                gate_config=gate_config,
                halting_config=halting_config,
                layer_model_config=AdaptiveLinearLayerConfig(
                    bias_flag=builder.bias_flag,
                    adaptive_augmentation_config=(
                        AdaptiveParameterAugmentationConfig(
                            weight_config=builder._build_weight_config(),
                            bias_config=builder._build_bias_config(),
                            diagonal_config=builder._build_diagonal_config(),
                            mask_config=builder._build_mask_config(),
                            model_config=builder._build_model_config(),
                        )
                    ),
                ),
            ),
        )

    def __validate_shared_gate_config(self, gate_config: GateConfig | None) -> None:
        if self.__is_active_gate_config(
            self.builder.shared_gate_config
        ) and self.__is_active_gate_config(gate_config):
            raise ValueError(
                "shared_gate_config cannot be provided when stack_gate_flag "
                "enables per-layer gate_config."
            )

    @staticmethod
    def __is_active_gate_config(gate_config: GateConfig | None) -> bool:
        return gate_config is not None

    def __maybe_wrap_recurrent(
        self, block_config: LayerStackConfig
    ) -> LayerStackConfig | RecurrentLayerConfig:
        if not self.builder.recurrent_flag:
            return block_config
        return RecurrentLayerConfig(
            max_steps=self.builder.recurrent_max_steps,
            recurrent_layer_norm_position=(
                self.builder.recurrent_layer_norm_position
            ),
            block_config=block_config,
            gate_config=self.__build_recurrent_gate_config(),
            residual_connection_option=ResidualConnectionOptions.DISABLED,
            halting_config=self.__build_recurrent_halting_config(),
        )

    def __build_gate_config(self, enabled: bool | None = None) -> GateConfig | None:
        enabled = resolve_enabled(enabled, self.builder.stack_gate_flag)
        if not enabled:
            return None
        model_config = self.__build_gate_model_config(enabled)
        return GateConfig(
            model_config=model_config,
            option=self.builder.gate_option,
            activation=self.builder.gate_activation,
        )

    def __build_gate_model_config(
        self, enabled: bool | None = None
    ) -> LayerStackConfig | None:
        enabled = resolve_enabled(enabled, self.builder.stack_gate_flag)
        if not enabled:
            return None
        options = resolve_controller_stack_options(
            self.__gate_stack_source(),
            self.__submodule_stack_defaults(),
        )
        return build_linear_controller_stack(options)

    def __build_recurrent_gate_config(self) -> GateConfig | None:
        if not self.builder.recurrent_gate_flag:
            return None
        gate_defaults = resolve_controller_stack_options(
            self.__gate_stack_source(),
            self.__submodule_stack_defaults(),
        )
        options = resolve_controller_stack_options(
            self.__recurrent_gate_stack_source(),
            gate_defaults,
        )
        return GateConfig(
            model_config=build_linear_controller_stack(options),
            option=self.builder.recurrent_gate_option,
            activation=self.builder.recurrent_gate_activation,
        )

    def __build_halting_config(
        self,
        enabled: bool | None = None,
    ) -> StickBreakingConfig | None:
        enabled = resolve_enabled(enabled, self.builder.stack_halting_flag)
        if not enabled:
            return None
        options = resolve_controller_stack_options(
            self.__halting_stack_source(),
            self.__submodule_stack_defaults(
                last_layer_bias_option=LastLayerBiasOptions.DISABLED
            ),
        )
        return StickBreakingConfig(
            threshold=self.builder.halting_threshold,
            halting_dropout=self.builder.halting_dropout,
            hidden_state_mode=self.builder.halting_hidden_state_mode,
            halting_gate_config=build_linear_controller_stack(
                options,
                hidden_dim=options.hidden_dim or self.builder.output_dim,
                output_dim=self.builder.halting_output_dim,
            ),
        )

    def __build_recurrent_halting_config(self) -> StickBreakingConfig | None:
        if not self.builder.recurrent_halting_flag:
            return None
        halting_defaults = resolve_controller_stack_options(
            self.__halting_stack_source(),
            self.__submodule_stack_defaults(
                last_layer_bias_option=LastLayerBiasOptions.DISABLED
            ),
        )
        options = resolve_controller_stack_options(
            self.__recurrent_halting_stack_source(),
            halting_defaults,
        )
        return StickBreakingConfig(
            threshold=self.builder.recurrent_halting_threshold,
            halting_dropout=self.builder.recurrent_halting_dropout,
            hidden_state_mode=self.builder.recurrent_halting_hidden_state_mode,
            halting_gate_config=build_linear_controller_stack(
                options,
                hidden_dim=options.hidden_dim or self.builder.output_dim,
                output_dim=self.builder.recurrent_halting_output_dim,
            ),
        )

    def __build_memory_config(
        self,
        enabled: bool | None = None,
    ) -> DynamicMemoryConfig | None:
        enabled = resolve_enabled(enabled, self.builder.memory_flag)
        if not enabled:
            return None
        options = resolve_controller_stack_options(
            self.__memory_stack_source(),
            self.__submodule_stack_defaults(),
        )
        return self.builder.memory_option(
            input_dim=self.builder.hidden_dim,
            output_dim=self.builder.hidden_dim,
            memory_position_option=self.builder.memory_position_option,
            test_time_training_learning_rate=(
                self.builder.memory_test_time_training_learning_rate
            ),
            test_time_training_num_inner_steps=(
                self.builder.memory_test_time_training_num_inner_steps
            ),
            model_config=build_linear_controller_stack(options),
        )

    def __submodule_stack_defaults(
        self,
        *,
        last_layer_bias_option: LastLayerBiasOptions | None = None,
    ) -> ControllerStackOptions:
        return ControllerStackOptions(
            hidden_dim=self.builder.submodule_hidden_dim,
            num_layers=self.builder.submodule_stack_num_layers,
            last_layer_bias_option=(
                self.builder.submodule_stack_last_layer_bias_option
                if last_layer_bias_option is None
                else last_layer_bias_option
            ),
            apply_output_pipeline_flag=(
                self.builder.submodule_stack_apply_output_pipeline_flag
            ),
            activation=self.builder.submodule_stack_activation,
            layer_norm_position=self.builder.submodule_layer_norm_position,
            residual_connection_option=(
                self.builder.submodule_stack_residual_connection_option
            ),
            dropout_probability=self.builder.submodule_stack_dropout_probability,
            bias_flag=self.builder.submodule_bias_flag,
        )

    def __gate_stack_source(self) -> ControllerStackSource:
        return ControllerStackSource(
            independent_flag=self.builder.gate_stack_independent_flag,
            hidden_dim=self.builder.gate_hidden_dim,
            num_layers=self.builder.gate_stack_num_layers,
            last_layer_bias_option=self.builder.gate_stack_last_layer_bias_option,
            apply_output_pipeline_flag=(
                self.builder.gate_stack_apply_output_pipeline_flag
            ),
            activation=self.builder.gate_stack_activation,
            layer_norm_position=self.builder.gate_layer_norm_position,
            residual_connection_option=(
                self.builder.gate_stack_residual_connection_option
            ),
            dropout_probability=self.builder.gate_stack_dropout_probability,
            bias_flag=self.builder.gate_bias_flag,
        )

    def __halting_stack_source(self) -> ControllerStackSource:
        return ControllerStackSource(
            independent_flag=self.builder.halting_stack_independent_flag,
            hidden_dim=self.builder.halting_hidden_dim,
            num_layers=self.builder.halting_stack_num_layers,
            last_layer_bias_option=(
                self.builder.halting_stack_last_layer_bias_option
            ),
            apply_output_pipeline_flag=(
                self.builder.halting_stack_apply_output_pipeline_flag
            ),
            activation=self.builder.halting_stack_activation,
            layer_norm_position=self.builder.halting_layer_norm_position,
            residual_connection_option=(
                self.builder.halting_stack_residual_connection_option
            ),
            dropout_probability=self.builder.halting_stack_dropout_probability,
            bias_flag=self.builder.halting_bias_flag,
        )

    def __memory_stack_source(self) -> ControllerStackSource:
        return ControllerStackSource(
            independent_flag=self.builder.memory_stack_independent_flag,
            hidden_dim=self.builder.memory_hidden_dim,
            num_layers=self.builder.memory_stack_num_layers,
            last_layer_bias_option=self.builder.memory_stack_last_layer_bias_option,
            apply_output_pipeline_flag=(
                self.builder.memory_stack_apply_output_pipeline_flag
            ),
            activation=self.builder.memory_stack_activation,
            layer_norm_position=self.builder.memory_layer_norm_position,
            residual_connection_option=(
                self.builder.memory_stack_residual_connection_option
            ),
            dropout_probability=self.builder.memory_stack_dropout_probability,
            bias_flag=self.builder.memory_bias_flag,
        )

    def __recurrent_gate_stack_source(self) -> ControllerStackSource:
        return ControllerStackSource(
            independent_flag=self.builder.recurrent_gate_stack_independent_flag,
            hidden_dim=self.builder.recurrent_gate_hidden_dim,
            num_layers=self.builder.recurrent_gate_stack_num_layers,
            last_layer_bias_option=(
                self.builder.recurrent_gate_stack_last_layer_bias_option
            ),
            apply_output_pipeline_flag=(
                self.builder.recurrent_gate_stack_apply_output_pipeline_flag
            ),
            activation=self.builder.recurrent_gate_stack_activation,
            layer_norm_position=self.builder.recurrent_gate_layer_norm_position,
            residual_connection_option=(
                self.builder.recurrent_gate_stack_residual_connection_option
            ),
            dropout_probability=self.builder.recurrent_gate_stack_dropout_probability,
            bias_flag=self.builder.recurrent_gate_bias_flag,
        )

    def __recurrent_halting_stack_source(self) -> ControllerStackSource:
        return ControllerStackSource(
            independent_flag=(
                self.builder.recurrent_halting_stack_independent_flag
            ),
            hidden_dim=self.builder.recurrent_halting_hidden_dim,
            num_layers=self.builder.recurrent_halting_stack_num_layers,
            last_layer_bias_option=(
                self.builder.recurrent_halting_stack_last_layer_bias_option
            ),
            apply_output_pipeline_flag=(
                self.builder.recurrent_halting_stack_apply_output_pipeline_flag
            ),
            activation=self.builder.recurrent_halting_stack_activation,
            layer_norm_position=self.builder.recurrent_halting_layer_norm_position,
            residual_connection_option=(
                self.builder.recurrent_halting_stack_residual_connection_option
            ),
            dropout_probability=(
                self.builder.recurrent_halting_stack_dropout_probability
            ),
            bias_flag=self.builder.recurrent_halting_bias_flag,
        )

from dataclasses import replace

from emperor.base.layer.config import LayerConfig, LayerStackConfig
from emperor.base.options import LastLayerBiasOptions
from emperor.halting.config import StickBreakingConfig
from emperor.linears.core.config import LinearLayerConfig

from models.linears._builder_options import (
    LayerControllerOptions,
    RecurrentControllerOptions,
)
from models.linears._controller_stack import (
    SubmoduleStackOptions,
    resolve_controller_stack_options,
)

STICK_BREAKING_GATE_OUTPUT_DIM = 2


class HaltingConfigFactory:
    def __init__(
        self,
        *,
        layer_controller_options: LayerControllerOptions,
        recurrent_controller_options: RecurrentControllerOptions,
        submodule_stack_options: SubmoduleStackOptions,
        output_dim: int,
    ) -> None:
        self.layer_controller_options = layer_controller_options
        self.recurrent_controller_options = recurrent_controller_options
        self.submodule_stack_options = submodule_stack_options
        self.output_dim = output_dim

    def build_halting_config(self) -> StickBreakingConfig | None:
        if not self.layer_controller_options.stack_halting_flag:
            return None
        halting_stack_source = self.layer_controller_options.halting_stack_source
        halting_stack_defaults = self.__submodule_stack_defaults(
            last_layer_bias_option=LastLayerBiasOptions.DISABLED
        )
        resolved_halting_stack_options = resolve_controller_stack_options(
            halting_stack_source,
            halting_stack_defaults,
        )
        halting_gate_config = self.__build_halting_gate_stack(
            resolved_halting_stack_options,
        )
        return StickBreakingConfig(
            threshold=self.layer_controller_options.halting_threshold,
            halting_dropout=self.layer_controller_options.halting_dropout,
            hidden_state_mode=self.layer_controller_options.halting_hidden_state_mode,
            halting_gate_config=halting_gate_config,
        )

    def build_recurrent_halting_config(self) -> StickBreakingConfig | None:
        if not self.recurrent_controller_options.recurrent_halting_flag:
            return None
        halting_stack_source = self.layer_controller_options.halting_stack_source
        halting_stack_defaults = self.__submodule_stack_defaults(
            last_layer_bias_option=LastLayerBiasOptions.DISABLED
        )
        resolved_halting_stack_defaults = resolve_controller_stack_options(
            halting_stack_source,
            halting_stack_defaults,
        )
        recurrent_halting_stack_source = (
            self.recurrent_controller_options.recurrent_halting_stack_source
        )
        resolved_recurrent_halting_stack_options = resolve_controller_stack_options(
            recurrent_halting_stack_source,
            resolved_halting_stack_defaults,
        )
        halting_gate_config = self.__build_halting_gate_stack(
            resolved_recurrent_halting_stack_options,
        )
        return StickBreakingConfig(
            threshold=self.recurrent_controller_options.recurrent_halting_threshold,
            halting_dropout=self.recurrent_controller_options.recurrent_halting_dropout,
            hidden_state_mode=(
                self.recurrent_controller_options.recurrent_halting_hidden_state_mode
            ),
            halting_gate_config=halting_gate_config,
        )

    def __build_halting_gate_stack(
        self,
        options: SubmoduleStackOptions,
    ) -> LayerStackConfig:
        halting_hidden_dim = options.hidden_dim or self.output_dim
        return self.__build_controller_stack(
            options,
            hidden_dim=halting_hidden_dim,
            output_dim=STICK_BREAKING_GATE_OUTPUT_DIM,
        )

    def __build_controller_stack(
        self,
        options: SubmoduleStackOptions,
        *,
        hidden_dim: int | None = None,
        output_dim: int | None = None,
    ) -> LayerStackConfig:
        return LayerStackConfig(
            hidden_dim=options.hidden_dim if hidden_dim is None else hidden_dim,
            output_dim=output_dim,
            num_layers=options.num_layers,
            last_layer_bias_option=options.last_layer_bias_option,
            apply_output_pipeline_flag=options.apply_output_pipeline_flag,
            layer_config=LayerConfig(
                activation=options.activation,
                layer_norm_position=options.layer_norm_position,
                residual_connection_option=options.residual_connection_option,
                dropout_probability=options.dropout_probability,
                halting_config=None,
                gate_config=None,
                memory_config=None,
                layer_model_config=LinearLayerConfig(
                    bias_flag=options.bias_flag,
                ),
            ),
        )

    def __submodule_stack_defaults(
        self,
        *,
        last_layer_bias_option: LastLayerBiasOptions | None = None,
    ) -> SubmoduleStackOptions:
        if last_layer_bias_option is None:
            return self.submodule_stack_options
        return replace(
            self.submodule_stack_options,
            last_layer_bias_option=last_layer_bias_option,
        )

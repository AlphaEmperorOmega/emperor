from dataclasses import replace

from emperor.base.layer.config import LayerStackConfig
from emperor.base.options import LastLayerBiasOptions
from emperor.halting.config import StickBreakingConfig

from models.experts._builder_options import (
    ExpertsSubmoduleStackOptions,
    ExpertsLayerControllerOptions,
    ExpertsRecurrentControllerOptions,
    resolve_experts_controller_stack_options,
)
from models.experts._controller_stack import build_linear_controller_stack


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

    def build_halting_config(self) -> StickBreakingConfig | None:
        if not self.layer_controller_options.stack_halting_flag:
            return None
        layer_controller = self.layer_controller_options
        halting_stack_options = resolve_experts_controller_stack_options(
            layer_controller.halting_stack_source,
            self.__halting_stack_defaults(),
        )
        halting_gate_config = self.__build_halting_gate_stack(
            halting_stack_options
        )
        return StickBreakingConfig(
            threshold=layer_controller.halting_threshold,
            halting_dropout=layer_controller.halting_dropout,
            hidden_state_mode=layer_controller.halting_hidden_state_mode,
            halting_gate_config=halting_gate_config,
        )

    def build_recurrent_halting_config(self) -> StickBreakingConfig | None:
        if not self.recurrent_controller_options.recurrent_halting_flag:
            return None
        layer_controller = self.layer_controller_options
        recurrent_controller = self.recurrent_controller_options
        halting_stack_defaults = self.__recurrent_halting_stack_defaults()
        recurrent_halting_stack_options = resolve_experts_controller_stack_options(
            recurrent_controller.recurrent_halting_stack_source,
            halting_stack_defaults,
        )
        halting_gate_config = self.__build_halting_gate_stack(
            recurrent_halting_stack_options
        )
        return StickBreakingConfig(
            threshold=recurrent_controller.recurrent_halting_threshold,
            halting_dropout=recurrent_controller.recurrent_halting_dropout,
            hidden_state_mode=(
                recurrent_controller.recurrent_halting_hidden_state_mode
            ),
            halting_gate_config=halting_gate_config,
        )

    def __build_halting_gate_stack(
        self,
        options: ExpertsSubmoduleStackOptions,
    ) -> LayerStackConfig:
        return build_linear_controller_stack(
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

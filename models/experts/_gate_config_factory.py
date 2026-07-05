from emperor.base.layer.config import LayerStackConfig
from emperor.base.layer.gate import GateConfig

from models.experts._builder_options import (
    ExpertsLayerControllerOptions,
    ExpertsRecurrentControllerOptions,
    ExpertsSubmoduleStackOptions,
    resolve_experts_controller_stack_options,
)
from models.experts._controller_stack import build_linear_controller_stack


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
        self.recurrent_stack_inherits_gate_stack = (
            recurrent_stack_inherits_gate_stack
        )

    def build_gate_config(self) -> GateConfig | None:
        if not self.layer_controller_options.stack_gate_flag:
            return None
        model_config = self.__build_gate_model_config()
        return GateConfig(
            model_config=model_config,
            option=self.layer_controller_options.gate_option,
            activation=self.layer_controller_options.gate_activation,
        )

    def build_recurrent_gate_config(self) -> GateConfig | None:
        if not self.recurrent_controller_options.recurrent_gate_flag:
            return None
        gate_stack_defaults = self.__recurrent_gate_stack_defaults()
        recurrent_gate_stack_options = resolve_experts_controller_stack_options(
            self.recurrent_controller_options.recurrent_gate_stack_source,
            gate_stack_defaults,
        )
        model_config = build_linear_controller_stack(recurrent_gate_stack_options)
        return GateConfig(
            model_config=model_config,
            option=self.recurrent_controller_options.recurrent_gate_option,
            activation=self.recurrent_controller_options.recurrent_gate_activation,
        )

    def __build_gate_model_config(self) -> LayerStackConfig:
        gate_stack_options = resolve_experts_controller_stack_options(
            self.layer_controller_options.gate_stack_source,
            self.submodule_stack_options,
        )
        return build_linear_controller_stack(gate_stack_options)

    def __recurrent_gate_stack_defaults(self) -> ExpertsSubmoduleStackOptions:
        if not self.recurrent_stack_inherits_gate_stack:
            return self.submodule_stack_options
        return resolve_experts_controller_stack_options(
            self.layer_controller_options.gate_stack_source,
            self.submodule_stack_options,
        )

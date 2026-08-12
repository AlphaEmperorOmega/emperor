from emperor.layers import GateConfig, LayerStackConfig
from models.vit.linear.runtime_options import (
    LayerControllerOptions,
    RecurrentControllerOptions,
    SubmoduleStackOptions,
    resolve_controller_stack_options,
)

from ._controller_stack_config import build_controller_stack_config


class GateConfigFactory:
    def __init__(
        self,
        *,
        layer_controller_options: LayerControllerOptions,
        recurrent_controller_options: RecurrentControllerOptions,
        submodule_stack_options: SubmoduleStackOptions,
        recurrent_stack_inherits_gate_stack: bool = True,
    ) -> None:
        self.layer_controller_options = layer_controller_options
        self.recurrent_controller_options = recurrent_controller_options
        self.submodule_stack_options = submodule_stack_options
        self.recurrent_stack_inherits_gate_stack = recurrent_stack_inherits_gate_stack

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
        if not self.recurrent_controller_options.recurrent_stack_gate_flag:
            return None
        resolved_gate_stack_defaults = self.__recurrent_gate_stack_defaults()
        recurrent_gate_stack_source = (
            self.recurrent_controller_options.recurrent_gate_stack_source
        )
        resolved_recurrent_gate_stack_options = resolve_controller_stack_options(
            recurrent_gate_stack_source, resolved_gate_stack_defaults
        )
        model_config = build_controller_stack_config(
            resolved_recurrent_gate_stack_options
        )
        return GateConfig(
            model_config=model_config,
            option=self.recurrent_controller_options.recurrent_gate_option,
            activation=self.recurrent_controller_options.recurrent_gate_activation,
        )

    def __build_gate_model_config(self) -> LayerStackConfig:
        gate_stack_source = self.layer_controller_options.gate_stack_source
        submodule_stack_defaults = self.submodule_stack_options
        resolved_gate_stack_options = resolve_controller_stack_options(
            gate_stack_source, submodule_stack_defaults
        )
        return build_controller_stack_config(resolved_gate_stack_options)

    def __recurrent_gate_stack_defaults(self) -> SubmoduleStackOptions:
        if not self.recurrent_stack_inherits_gate_stack:
            return self.submodule_stack_options
        return resolve_controller_stack_options(
            self.layer_controller_options.gate_stack_source,
            self.submodule_stack_options,
        )

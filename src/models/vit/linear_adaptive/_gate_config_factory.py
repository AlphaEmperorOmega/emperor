from emperor.layers import GateConfig, LayerConfig, LayerStackConfig, ResidualConfig
from emperor.linears import LinearLayerConfig
from models.vit.linear_adaptive.runtime_options import (
    LayerControllerOptions,
    RecurrentControllerOptions,
    SubmoduleStackOptions,
    resolve_controller_stack_options,
)


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
        if not self.recurrent_controller_options.recurrent_gate_flag:
            return None
        resolved_gate_stack_defaults = self.__recurrent_gate_stack_defaults()
        recurrent_gate_stack_source = (
            self.recurrent_controller_options.recurrent_gate_stack_source
        )
        resolved_recurrent_gate_stack_options = resolve_controller_stack_options(
            recurrent_gate_stack_source, resolved_gate_stack_defaults
        )
        model_config = self.__build_controller_stack(
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
        return self.__build_controller_stack(resolved_gate_stack_options)

    def __recurrent_gate_stack_defaults(self) -> SubmoduleStackOptions:
        if not self.recurrent_stack_inherits_gate_stack:
            return self.submodule_stack_options
        return resolve_controller_stack_options(
            self.layer_controller_options.gate_stack_source,
            self.submodule_stack_options,
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
                residual_config=None
                if options.residual_connection_option is None
                else ResidualConfig(option=options.residual_connection_option),
                dropout_probability=options.dropout_probability,
                halting_config=None,
                gate_config=None,
                memory_config=None,
                layer_model_config=LinearLayerConfig(
                    bias_flag=options.bias_flag,
                ),
            ),
        )

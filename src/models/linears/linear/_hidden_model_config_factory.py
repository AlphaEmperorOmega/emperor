from emperor.halting import HaltingConfig
from emperor.layers import (
    GateConfig,
    LayerConfig,
    LayerStackConfig,
    RecurrentLayerConfig,
    ResidualConfig,
)
from emperor.linears import LinearLayerConfig
from emperor.memory import DynamicMemoryConfig
from models.linears.linear._control_config_factory import ControlConfigFactory
from models.linears.linear.runtime_options import RuntimeOptions


class HiddenModelConfigFactory:
    def __init__(self, runtime: RuntimeOptions) -> None:
        self.runtime = runtime
        self.control_factory = ControlConfigFactory(runtime)

    def build_hidden_model_config(self) -> LayerStackConfig | RecurrentLayerConfig:
        gate_config = self.control_factory.build_gate_config()
        halting_config = self.control_factory.build_halting_config()
        memory_config = self.control_factory.build_memory_config()
        stack_config = self._stack_config(
            memory_config=memory_config,
            layer_config=self._layer_config(
                gate_config=gate_config,
                halting_config=halting_config,
            ),
        )
        return self.control_factory.wrap_recurrent(stack_config)

    def _stack_config(
        self,
        *,
        memory_config: DynamicMemoryConfig | None,
        layer_config: LayerConfig,
    ) -> LayerStackConfig:
        options = self.runtime.stack
        return LayerStackConfig(
            hidden_dim=self.runtime.hidden_dim,
            num_layers=options.num_layers,
            last_layer_bias_option=options.last_layer_bias_option,
            apply_output_pipeline_flag=options.apply_output_pipeline_flag,
            shared_gate_config=self.runtime.gate.shared_config,
            shared_memory_config=memory_config,
            layer_config=layer_config,
        )

    def _layer_config(
        self,
        *,
        gate_config: GateConfig | None,
        halting_config: HaltingConfig | None,
    ) -> LayerConfig:
        options = self.runtime.stack
        return LayerConfig(
            activation=options.activation,
            layer_norm_position=options.layer_norm_position,
            residual_config=None
            if options.residual_connection_option is None
            else ResidualConfig(option=options.residual_connection_option),
            dropout_probability=options.dropout_probability,
            gate_config=gate_config,
            halting_config=halting_config,
            layer_model_config=LinearLayerConfig(
                bias_flag=options.bias_flag,
            ),
        )

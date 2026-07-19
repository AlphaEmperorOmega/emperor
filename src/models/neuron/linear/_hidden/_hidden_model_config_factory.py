from emperor.base.layer.config import (
    LayerConfig,
    LayerStackConfig,
    RecurrentLayerConfig,
)
from emperor.base.layer.gate import GateConfig
from emperor.halting.config import StickBreakingConfig
from emperor.linears.core.config import LinearLayerConfig
from emperor.memory.config import DynamicMemoryConfig

from models.neuron.linear._hidden._control_config_factory import ControlConfigFactory
from models.neuron.linear._hidden.runtime_options import RuntimeOptions


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
        halting_config: StickBreakingConfig | None,
    ) -> LayerConfig:
        options = self.runtime.stack
        return LayerConfig(
            activation=options.activation,
            layer_norm_position=options.layer_norm_position,
            residual_connection_option=options.residual_connection_option,
            dropout_probability=options.dropout_probability,
            gate_config=gate_config,
            halting_config=halting_config,
            layer_model_config=LinearLayerConfig(
                bias_flag=options.bias_flag,
            ),
        )

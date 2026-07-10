from emperor.base.layer.config import (
    LayerConfig,
    LayerStackConfig,
    RecurrentLayerConfig,
)
from emperor.linears.core.config import AdaptiveLinearLayerConfig

from models.linears.linear_adaptive._adaptive_parameter_config_factory import (
    AdaptiveParameterConfigFactory,
)
from models.linears.linear_adaptive._control_config_factory import (
    ControlConfigFactory,
)
from models.linears.linear_adaptive.runtime_options import RuntimeOptions


class HiddenModelConfigFactory:
    def __init__(self, runtime: RuntimeOptions) -> None:
        self._runtime = runtime
        self._control_factory = ControlConfigFactory(runtime)
        self._adaptive_factory = AdaptiveParameterConfigFactory(runtime)

    def build_hidden_model_config(
        self,
    ) -> LayerStackConfig | RecurrentLayerConfig:
        runtime = self._runtime
        block_config = LayerStackConfig(
            hidden_dim=runtime.hidden_dim,
            num_layers=runtime.stack.num_layers,
            last_layer_bias_option=runtime.stack.last_layer_bias_option,
            apply_output_pipeline_flag=(runtime.stack.apply_output_pipeline_flag),
            shared_gate_config=runtime.gate.shared_config,
            shared_memory_config=self._control_factory.build_memory_config(),
            layer_config=LayerConfig(
                activation=runtime.stack.activation,
                layer_norm_position=runtime.stack.layer_norm_position,
                residual_connection_option=(runtime.stack.residual_connection_option),
                dropout_probability=runtime.stack.dropout_probability,
                gate_config=self._control_factory.build_gate_config(runtime.gate),
                halting_config=self._control_factory.build_halting_config(
                    runtime.halting
                ),
                layer_model_config=AdaptiveLinearLayerConfig(
                    bias_flag=runtime.stack.bias_flag,
                    adaptive_augmentation_config=(
                        self._adaptive_factory.build_hidden_config()
                    ),
                ),
            ),
        )
        return self._control_factory.apply_recurrence(block_config)

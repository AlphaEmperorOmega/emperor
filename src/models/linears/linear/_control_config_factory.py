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
from models.linears.linear.runtime_options import (
    ControllerStackOptions,
    GateOptions,
    HaltingOptions,
    RuntimeOptions,
)

_STICK_BREAKING_GATE_OUTPUT_DIM = 2


class ControlConfigFactory:
    def __init__(self, runtime: RuntimeOptions) -> None:
        self.runtime = runtime

    def build_gate_config(self) -> GateConfig | None:
        return self._gate_config(self.runtime.gate)

    def build_halting_config(self) -> HaltingConfig | None:
        return self._halting_config(
            self.runtime.halting,
            self.runtime.halting_option,
        )

    def build_memory_config(self) -> DynamicMemoryConfig | None:
        options = self.runtime.memory
        if not options.enabled:
            return None
        return options.implementation(
            input_dim=self.runtime.hidden_dim,
            output_dim=self.runtime.hidden_dim,
            memory_position_option=options.position,
            test_time_training_learning_rate=(options.test_time_training_learning_rate),
            test_time_training_num_inner_steps=(
                options.test_time_training_num_inner_steps
            ),
            model_config=self._controller_stack(options.stack),
        )

    def wrap_recurrent(
        self,
        block_config: LayerStackConfig,
    ) -> LayerStackConfig | RecurrentLayerConfig:
        options = self.runtime.recurrence
        if not options.enabled:
            return block_config
        return RecurrentLayerConfig(
            max_steps=options.max_steps,
            recurrent_layer_norm_position=options.layer_norm_position,
            block_config=block_config,
            gate_config=self._gate_config(options.gate),
            residual_config=None,
            halting_config=self._halting_config(
                options.halting,
                self.runtime.recurrent_halting_option,
            ),
        )

    def _gate_config(self, options: GateOptions) -> GateConfig | None:
        if not options.enabled:
            return None
        return GateConfig(
            model_config=self._controller_stack(options.stack),
            option=options.option,
            activation=options.activation,
        )

    def _halting_config(
        self,
        options: HaltingOptions,
        option: type[HaltingConfig],
    ) -> HaltingConfig | None:
        if not options.enabled:
            return None
        return option(
            threshold=options.threshold,
            dropout_probability=options.dropout_probability,
            hidden_state_mode=options.hidden_state_mode,
            halting_gate_config=self._controller_stack(
                options.stack,
                output_dim=_STICK_BREAKING_GATE_OUTPUT_DIM,
            ),
        )

    def _controller_stack(
        self,
        options: ControllerStackOptions,
        *,
        output_dim: int | None = None,
    ) -> LayerStackConfig:
        return LayerStackConfig(
            hidden_dim=options.hidden_dim,
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

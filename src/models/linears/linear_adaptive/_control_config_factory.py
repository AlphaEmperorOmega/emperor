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
from models.linears.linear_adaptive.runtime_options import (
    GateOptions,
    HaltingOptions,
    RuntimeOptions,
    StackOptions,
)

STICK_BREAKING_GATE_OUTPUT_DIM = 2


class ControlConfigFactory:
    def __init__(self, runtime: RuntimeOptions) -> None:
        self._runtime = runtime

    def build_gate_config(self, options: GateOptions) -> GateConfig | None:
        if not options.enabled:
            return None
        return GateConfig(
            model_config=self._build_stack(options.stack),
            option=options.option,
            activation=options.activation,
        )

    def build_halting_config(
        self,
        options: HaltingOptions,
        option: type[HaltingConfig] | None = None,
    ) -> HaltingConfig | None:
        if not options.enabled:
            return None
        halting_option = self._runtime.halting_option if option is None else option
        return halting_option(
            threshold=options.threshold,
            dropout_probability=options.dropout_probability,
            hidden_state_mode=options.hidden_state_mode,
            halting_gate_config=self._build_stack(
                options.stack,
                hidden_dim=options.stack.hidden_dim or self._runtime.output_dim,
                output_dim=STICK_BREAKING_GATE_OUTPUT_DIM,
            ),
        )

    def build_memory_config(self) -> DynamicMemoryConfig | None:
        options = self._runtime.memory
        if not options.enabled:
            return None
        return options.option(
            input_dim=self._runtime.hidden_dim,
            output_dim=self._runtime.hidden_dim,
            memory_position_option=options.position,
            test_time_training_learning_rate=(options.test_time_training_learning_rate),
            test_time_training_num_inner_steps=(
                options.test_time_training_num_inner_steps
            ),
            model_config=self._build_stack(options.stack),
        )

    def apply_recurrence(
        self,
        block_config: LayerStackConfig,
    ) -> LayerStackConfig | RecurrentLayerConfig:
        recurrence = self._runtime.recurrence
        if not recurrence.enabled:
            return block_config
        return RecurrentLayerConfig(
            max_steps=recurrence.max_steps,
            recurrent_layer_norm_position=recurrence.layer_norm_position,
            block_config=block_config,
            gate_config=self.build_gate_config(recurrence.gate),
            residual_config=None,
            halting_config=self.build_halting_config(
                recurrence.halting,
                self._runtime.recurrent_halting_option,
            ),
        )

    @staticmethod
    def _build_stack(
        options: StackOptions,
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
                layer_model_config=LinearLayerConfig(bias_flag=options.bias_flag),
            ),
        )

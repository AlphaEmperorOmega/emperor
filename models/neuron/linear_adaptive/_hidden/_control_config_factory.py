from emperor.base.layer.config import (
    LayerConfig,
    LayerStackConfig,
    RecurrentLayerConfig,
)
from emperor.base.layer.gate import GateConfig
from emperor.base.layer.residual import ResidualConnectionOptions
from emperor.halting.config import StickBreakingConfig
from emperor.linears.core.config import LinearLayerConfig
from emperor.memory.config import DynamicMemoryConfig

from models.neuron.linear_adaptive._hidden.runtime_options import (
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
    ) -> StickBreakingConfig | None:
        if not options.enabled:
            return None
        return StickBreakingConfig(
            threshold=options.threshold,
            halting_dropout=options.dropout_probability,
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
            residual_connection_option=ResidualConnectionOptions.DISABLED,
            halting_config=self.build_halting_config(recurrence.halting),
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
                residual_connection_option=options.residual_connection_option,
                dropout_probability=options.dropout_probability,
                halting_config=None,
                gate_config=None,
                memory_config=None,
                layer_model_config=LinearLayerConfig(bias_flag=options.bias_flag),
            ),
        )

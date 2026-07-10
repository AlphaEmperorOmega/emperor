from __future__ import annotations

from emperor.base.layer.config import (
    LayerConfig,
    LayerStackConfig,
    RecurrentLayerConfig,
)
from emperor.base.layer.gate import GateConfig
from emperor.base.layer.residual import ResidualConnectionOptions
from emperor.base.options import LastLayerBiasOptions
from emperor.halting.config import StickBreakingConfig
from emperor.linears.core.config import LinearLayerConfig

from .options import (
    ControllerStackOptions,
    SubmoduleStackOptions,
    TransformerAttentionOptions,
    TransformerFeedForwardOptions,
    resolve_controller_stack_options,
)


def _controller_stack(
    source: ControllerStackOptions,
    defaults: SubmoduleStackOptions,
    *,
    output_dim: int | None = None,
) -> LayerStackConfig:
    options = resolve_controller_stack_options(source, defaults)
    return LayerStackConfig(
        hidden_dim=options.hidden_dim,
        output_dim=output_dim,
        num_layers=options.num_layers,
        last_layer_bias_option=(
            options.last_layer_bias_option
            if output_dim is None
            else LastLayerBiasOptions.DISABLED
        ),
        apply_output_pipeline_flag=options.apply_output_pipeline_flag,
        layer_config=LayerConfig(
            activation=options.activation,
            layer_norm_position=options.layer_norm_position,
            residual_connection_option=options.residual_connection_option,
            dropout_probability=options.dropout_probability,
            gate_config=None,
            halting_config=None,
            memory_config=None,
            layer_model_config=LinearLayerConfig(bias_flag=options.bias_flag),
        ),
    )


def _gate_config(
    path_options: TransformerAttentionOptions | TransformerFeedForwardOptions,
    *,
    recurrent: bool,
) -> GateConfig | None:
    stack = path_options.stack_options
    if recurrent:
        options = path_options.recurrent_controller_options
        if not options.recurrent_gate_flag:
            return None
        return GateConfig(
            option=options.recurrent_gate_option,
            activation=options.recurrent_gate_activation,
            model_config=_controller_stack(
                options.recurrent_gate_stack_options,
                stack,
            ),
        )
    options = path_options.layer_controller_options
    if not options.stack_gate_flag:
        return None
    return GateConfig(
        option=options.gate_option,
        activation=options.gate_activation,
        model_config=_controller_stack(options.gate_stack_options, stack),
    )


def _halting_config(
    path_options: TransformerAttentionOptions | TransformerFeedForwardOptions,
    *,
    recurrent: bool,
) -> StickBreakingConfig | None:
    stack = path_options.stack_options
    if recurrent:
        options = path_options.recurrent_controller_options
        if not options.recurrent_halting_flag:
            return None
        return StickBreakingConfig(
            threshold=options.recurrent_halting_threshold,
            halting_dropout=options.recurrent_halting_dropout,
            hidden_state_mode=options.recurrent_halting_hidden_state_mode,
            halting_gate_config=_controller_stack(
                options.recurrent_halting_stack_options,
                stack,
                output_dim=2,
            ),
        )
    options = path_options.layer_controller_options
    if not options.stack_halting_flag:
        return None
    return StickBreakingConfig(
        threshold=options.halting_threshold,
        halting_dropout=options.halting_dropout,
        hidden_state_mode=options.halting_hidden_state_mode,
        halting_gate_config=_controller_stack(
            options.halting_stack_options,
            stack,
            output_dim=2,
        ),
    )


def _memory_config(
    path_options: TransformerAttentionOptions | TransformerFeedForwardOptions,
    *,
    model_dim: int,
):
    options = path_options.dynamic_memory_options
    if not options.memory_flag:
        return None
    return options.memory_option(
        input_dim=model_dim,
        output_dim=model_dim,
        memory_position_option=options.memory_position_option,
        test_time_training_learning_rate=(
            options.memory_test_time_training_learning_rate
        ),
        test_time_training_num_inner_steps=(
            options.memory_test_time_training_num_inner_steps
        ),
        model_config=_controller_stack(
            options.memory_stack_options,
            path_options.stack_options,
        ),
    )


def configure_transformer_submodule(
    model_config,
    *,
    control_stack: LayerStackConfig,
    path_options: TransformerAttentionOptions | TransformerFeedForwardOptions,
    model_dim: int,
):
    """Apply path controllers and optionally wrap a package-local primary backend."""

    control_stack.layer_config.gate_config = _gate_config(path_options, recurrent=False)
    control_stack.layer_config.halting_config = _halting_config(
        path_options, recurrent=False
    )
    control_stack.shared_gate_config = (
        path_options.layer_controller_options.shared_gate_config
    )
    control_stack.shared_memory_config = _memory_config(
        path_options, model_dim=model_dim
    )
    recurrent = path_options.recurrent_controller_options
    if not recurrent.recurrent_flag:
        return model_config
    return RecurrentLayerConfig(
        input_dim=model_dim,
        output_dim=model_dim,
        max_steps=recurrent.recurrent_max_steps,
        recurrent_layer_norm_position=recurrent.recurrent_layer_norm_position,
        block_config=model_config,
        gate_config=_gate_config(path_options, recurrent=True),
        residual_connection_option=ResidualConnectionOptions.DISABLED,
        halting_config=_halting_config(path_options, recurrent=True),
        memory_config=None,
    )


__all__ = ["configure_transformer_submodule"]

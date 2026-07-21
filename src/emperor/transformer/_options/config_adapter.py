from types import ModuleType

from emperor.transformer._options.records import (
    ControllerStackOptions,
    DynamicMemoryOptions,
    LayerControllerOptions,
    RecurrentControllerOptions,
    SubmoduleStackOptions,
    TransformerAttentionOptions,
    TransformerFeedForwardOptions,
)


def _controller_stack_from_config(
    config: ModuleType,
    prefix: str,
) -> ControllerStackOptions:
    return ControllerStackOptions(
        independent_flag=getattr(config, f"{prefix}_INDEPENDENT_FLAG"),
        hidden_dim=getattr(config, f"{prefix}_HIDDEN_DIM"),
        num_layers=getattr(config, f"{prefix}_NUM_LAYERS"),
        last_layer_bias_option=getattr(config, f"{prefix}_LAST_LAYER_BIAS_OPTION"),
        apply_output_pipeline_flag=getattr(
            config, f"{prefix}_APPLY_OUTPUT_PIPELINE_FLAG"
        ),
        activation=getattr(config, f"{prefix}_ACTIVATION"),
        layer_norm_position=getattr(config, f"{prefix}_LAYER_NORM_POSITION"),
        residual_connection_option=getattr(
            config, f"{prefix}_RESIDUAL_CONNECTION_OPTION"
        ),
        dropout_probability=getattr(config, f"{prefix}_DROPOUT_PROBABILITY"),
        bias_flag=getattr(config, f"{prefix}_BIAS_FLAG"),
    )


def _layer_controller_from_config(
    config: ModuleType,
    prefix: str,
) -> LayerControllerOptions:
    return LayerControllerOptions(
        stack_gate_flag=getattr(config, f"{prefix}_GATE_FLAG"),
        gate_option=getattr(config, f"{prefix}_GATE_OPTION"),
        gate_activation=getattr(config, f"{prefix}_GATE_ACTIVATION"),
        gate_stack_options=_controller_stack_from_config(
            config, f"{prefix}_GATE_STACK"
        ),
        stack_halting_flag=getattr(config, f"{prefix}_HALTING_FLAG"),
        halting_option=getattr(config, f"{prefix}_HALTING_OPTION"),
        halting_threshold=getattr(config, f"{prefix}_HALTING_THRESHOLD"),
        halting_dropout=getattr(config, f"{prefix}_HALTING_DROPOUT"),
        halting_hidden_state_mode=getattr(
            config, f"{prefix}_HALTING_HIDDEN_STATE_MODE"
        ),
        halting_stack_options=_controller_stack_from_config(
            config, f"{prefix}_HALTING_STACK"
        ),
    )


def _memory_from_config(config: ModuleType, prefix: str) -> DynamicMemoryOptions:
    return DynamicMemoryOptions(
        memory_flag=getattr(config, f"{prefix}_MEMORY_FLAG"),
        memory_option=getattr(config, f"{prefix}_MEMORY_OPTION"),
        memory_position_option=getattr(config, f"{prefix}_MEMORY_POSITION_OPTION"),
        memory_test_time_training_learning_rate=getattr(
            config, f"{prefix}_MEMORY_TEST_TIME_TRAINING_LEARNING_RATE"
        ),
        memory_test_time_training_num_inner_steps=getattr(
            config, f"{prefix}_MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS"
        ),
        memory_stack_options=_controller_stack_from_config(
            config, f"{prefix}_MEMORY_STACK"
        ),
    )


def _recurrent_from_config(
    config: ModuleType,
    prefix: str,
) -> RecurrentControllerOptions:
    return RecurrentControllerOptions(
        recurrent_flag=getattr(config, f"{prefix}_RECURRENT_FLAG"),
        recurrent_max_steps=getattr(config, f"{prefix}_RECURRENT_MAX_STEPS"),
        recurrent_layer_norm_position=getattr(
            config, f"{prefix}_RECURRENT_LAYER_NORM_POSITION"
        ),
        recurrent_gate_flag=getattr(config, f"{prefix}_RECURRENT_GATE_FLAG"),
        recurrent_gate_option=getattr(config, f"{prefix}_RECURRENT_GATE_OPTION"),
        recurrent_gate_activation=getattr(
            config, f"{prefix}_RECURRENT_GATE_ACTIVATION"
        ),
        recurrent_gate_stack_options=_controller_stack_from_config(
            config, f"{prefix}_RECURRENT_GATE_STACK"
        ),
        recurrent_halting_flag=getattr(config, f"{prefix}_RECURRENT_HALTING_FLAG"),
        recurrent_halting_option=getattr(
            config,
            f"{prefix}_RECURRENT_HALTING_OPTION",
        ),
        recurrent_halting_threshold=getattr(
            config, f"{prefix}_RECURRENT_HALTING_THRESHOLD"
        ),
        recurrent_halting_dropout=getattr(
            config, f"{prefix}_RECURRENT_HALTING_DROPOUT"
        ),
        recurrent_halting_hidden_state_mode=getattr(
            config, f"{prefix}_RECURRENT_HALTING_HIDDEN_STATE_MODE"
        ),
        recurrent_halting_stack_options=_controller_stack_from_config(
            config, f"{prefix}_RECURRENT_HALTING_STACK"
        ),
    )


def attention_options_from_config(
    config: ModuleType,
) -> TransformerAttentionOptions:
    stack = SubmoduleStackOptions(
        hidden_dim=config.ATTN_STACK_HIDDEN_DIM,
        num_layers=config.ATTN_NUM_LAYERS,
        last_layer_bias_option=config.ATTN_STACK_LAST_LAYER_BIAS_OPTION,
        apply_output_pipeline_flag=config.ATTN_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        activation=config.ATTN_STACK_ACTIVATION,
        layer_norm_position=config.ATTN_STACK_LAYER_NORM_POSITION,
        residual_connection_option=config.ATTN_STACK_RESIDUAL_CONNECTION_OPTION,
        dropout_probability=config.ATTN_STACK_DROPOUT_PROBABILITY,
        bias_flag=config.ATTN_BIAS_FLAG,
    )
    return TransformerAttentionOptions(
        num_heads=config.ATTN_NUM_HEADS,
        add_key_value_bias_flag=config.ATTN_ADD_KEY_VALUE_BIAS_FLAG,
        zero_attention_flag=config.ATTN_ZERO_ATTENTION_FLAG,
        stack_options=stack,
        layer_controller_options=_layer_controller_from_config(config, "ATTN"),
        dynamic_memory_options=_memory_from_config(config, "ATTN"),
        recurrent_controller_options=_recurrent_from_config(config, "ATTN"),
    )


def feed_forward_options_from_config(
    config: ModuleType,
) -> TransformerFeedForwardOptions:
    stack = SubmoduleStackOptions(
        hidden_dim=config.FF_STACK_HIDDEN_DIM,
        num_layers=config.FF_NUM_LAYERS,
        last_layer_bias_option=config.FF_STACK_LAST_LAYER_BIAS_OPTION,
        apply_output_pipeline_flag=config.FF_STACK_APPLY_OUTPUT_PIPELINE_FLAG,
        activation=config.FF_STACK_ACTIVATION,
        layer_norm_position=config.FF_STACK_LAYER_NORM_POSITION,
        residual_connection_option=config.FF_STACK_RESIDUAL_CONNECTION_OPTION,
        dropout_probability=config.FF_STACK_DROPOUT_PROBABILITY,
        bias_flag=config.FF_BIAS_FLAG,
    )
    return TransformerFeedForwardOptions(
        stack_options=stack,
        layer_controller_options=_layer_controller_from_config(config, "FF"),
        dynamic_memory_options=_memory_from_config(config, "FF"),
        recurrent_controller_options=_recurrent_from_config(config, "FF"),
    )

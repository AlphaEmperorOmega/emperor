from __future__ import annotations

from emperor.attention import (
    IndependentAttentionConfig,
    SelfAttentionConfig,
    SelfAttentionProjectionStrategy,
)
from emperor.halting import HaltingConfig, HaltingHiddenStateModeOptions
from emperor.layers import (
    ActivationOptions,
    GateConfig,
    LastLayerBiasOptions,
    LayerConfig,
    LayerGateOptions,
    LayerNormPositionOptions,
    LayerStackConfig,
    RecurrentLayerConfig,
    ResidualConfig,
    ResidualConnectionOptions,
)
from emperor.linears import LinearLayerConfig
from emperor.memory import GatedResidualDynamicMemoryConfig, MemoryPositionOptions
from emperor.transformer import (
    FeedForwardConfig,
    TransformerDecoderBlockLayerConfig,
    TransformerDecoderLayerConfig,
    TransformerEncoderBlockLayerConfig,
    TransformerEncoderLayerConfig,
)

from ._transformer_submodule import configure_transformer_submodule
from .experiment_config import ExperimentConfig
from .runtime_options import (
    RuntimeOptions,
    TransformerAttentionOptions,
    TransformerFeedForwardOptions,
    TransformerStackOptions,
)


def _linear_stack(
    *,
    hidden_dim: int,
    num_layers: int = 1,
    bias_flag: bool = True,
    activation: ActivationOptions = ActivationOptions.RELU,
    dropout_probability: float = 0.0,
) -> LayerStackConfig:
    return LayerStackConfig(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        apply_output_pipeline_flag=False,
        last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
        layer_config=LayerConfig(
            activation=activation,
            residual_config=None,
            dropout_probability=dropout_probability,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            gate_config=None,
            halting_config=None,
            memory_config=None,
            layer_model_config=LinearLayerConfig(bias_flag=bias_flag),
        ),
    )


def _controller_stack(model_dim: int, output_dim: int | None = None):
    return (
        _linear_stack(
            hidden_dim=model_dim,
            num_layers=1,
            bias_flag=True,
            activation=ActivationOptions.SIGMOID,
        )
        if output_dim is None
        else LayerStackConfig(
            hidden_dim=model_dim,
            output_dim=output_dim,
            num_layers=1,
            apply_output_pipeline_flag=False,
            last_layer_bias_option=LastLayerBiasOptions.DISABLED,
            layer_config=LayerConfig(
                activation=ActivationOptions.SIGMOID,
                residual_config=None,
                dropout_probability=0.0,
                layer_norm_position=LayerNormPositionOptions.DISABLED,
                gate_config=None,
                halting_config=None,
                memory_config=None,
                layer_model_config=LinearLayerConfig(bias_flag=True),
            ),
        )
    )


def _gate(model_dim: int, enabled: bool):
    if not enabled:
        return None
    return GateConfig(
        option=LayerGateOptions.MULTIPLIER,
        activation=ActivationOptions.SIGMOID,
        model_config=_controller_stack(model_dim),
    )


def _halting(
    model_dim: int,
    enabled: bool,
    option: type[HaltingConfig],
    threshold: float | None,
):
    if not enabled:
        return None
    return option(
        threshold=threshold,
        dropout_probability=0.0,
        hidden_state_mode=HaltingHiddenStateModeOptions.RAW,
        halting_gate_config=_controller_stack(model_dim, output_dim=2),
    )


def _memory(model_dim: int, enabled: bool):
    if not enabled:
        return None
    return GatedResidualDynamicMemoryConfig(
        input_dim=model_dim,
        output_dim=model_dim,
        memory_position_option=MemoryPositionOptions.AFTER_AFFINE,
        test_time_training_learning_rate=None,
        test_time_training_num_inner_steps=None,
        model_config=_linear_stack(hidden_dim=model_dim),
    )


def _projection_stack(model_dim: int, options: TransformerAttentionOptions):
    stack_options = options.stack_options
    stack = LayerStackConfig(
        hidden_dim=stack_options.hidden_dim,
        num_layers=stack_options.num_layers,
        apply_output_pipeline_flag=(stack_options.apply_output_pipeline_flag),
        last_layer_bias_option=stack_options.last_layer_bias_option,
        layer_config=LayerConfig(
            activation=stack_options.activation,
            residual_config=None
            if (stack_options.residual_connection_option) is None
            else ResidualConfig(option=(stack_options.residual_connection_option)),
            dropout_probability=stack_options.dropout_probability,
            layer_norm_position=stack_options.layer_norm_position,
            gate_config=None,
            halting_config=None,
            memory_config=None,
            layer_model_config=LinearLayerConfig(bias_flag=stack_options.bias_flag),
        ),
    )
    return configure_transformer_submodule(
        stack,
        control_stack=stack,
        path_options=options,
        model_dim=model_dim,
    )


def _self_attention(
    runtime: RuntimeOptions,
    options: TransformerAttentionOptions,
    *,
    maximum_length: int,
    causal: bool,
):
    return SelfAttentionConfig(
        batch_size=runtime.batch_size,
        num_heads=options.num_heads,
        embedding_dim=runtime.model_dim,
        query_key_projection_dim=runtime.model_dim,
        value_projection_dim=runtime.model_dim,
        target_sequence_length=maximum_length,
        source_sequence_length=maximum_length,
        target_dtype=__import__("torch").float32,
        dropout_probability=runtime.dropout_probability,
        zero_attention_flag=options.zero_attention_flag,
        causal_attention_mask_flag=causal,
        add_key_value_bias_flag=options.add_key_value_bias_flag,
        average_attention_weights_flag=True,
        return_attention_weights_flag=False,
        batch_first_flag=True,
        projection_model_config=_projection_stack(runtime.model_dim, options),
        relative_positional_embedding_config=None,
        projection_strategy=SelfAttentionProjectionStrategy.SEPARATE,
    )


def _cross_attention(
    runtime: RuntimeOptions,
    options: TransformerAttentionOptions,
):
    return IndependentAttentionConfig(
        batch_size=runtime.batch_size,
        num_heads=options.num_heads,
        embedding_dim=runtime.model_dim,
        query_key_projection_dim=runtime.model_dim,
        value_projection_dim=runtime.model_dim,
        target_sequence_length=runtime.target_sequence_length,
        source_sequence_length=runtime.source_sequence_length,
        target_dtype=__import__("torch").float32,
        dropout_probability=runtime.dropout_probability,
        zero_attention_flag=options.zero_attention_flag,
        causal_attention_mask_flag=False,
        add_key_value_bias_flag=options.add_key_value_bias_flag,
        average_attention_weights_flag=True,
        return_attention_weights_flag=False,
        batch_first_flag=True,
        projection_model_config=_projection_stack(runtime.model_dim, options),
        relative_positional_embedding_config=None,
    )


def _feed_forward(
    runtime: RuntimeOptions,
    options: TransformerFeedForwardOptions,
):
    stack_options = options.stack_options
    stack = LayerStackConfig(
        hidden_dim=stack_options.hidden_dim,
        num_layers=stack_options.num_layers,
        apply_output_pipeline_flag=(stack_options.apply_output_pipeline_flag),
        last_layer_bias_option=stack_options.last_layer_bias_option,
        layer_config=LayerConfig(
            activation=stack_options.activation,
            residual_config=None
            if (stack_options.residual_connection_option) is None
            else ResidualConfig(option=(stack_options.residual_connection_option)),
            dropout_probability=stack_options.dropout_probability,
            layer_norm_position=stack_options.layer_norm_position,
            gate_config=None,
            halting_config=None,
            memory_config=None,
            layer_model_config=LinearLayerConfig(bias_flag=stack_options.bias_flag),
        ),
    )
    configured = configure_transformer_submodule(
        stack,
        control_stack=stack,
        path_options=options,
        model_dim=runtime.model_dim,
    )
    return FeedForwardConfig(
        input_dim=runtime.model_dim,
        output_dim=runtime.model_dim,
        stack_config=configured,
    )


def _controlled_stack(
    runtime: RuntimeOptions,
    options: TransformerStackOptions,
    layer_config: LayerConfig,
):
    shared_halting_config = layer_config.halting_config
    layer_config.halting_config = None
    stack = LayerStackConfig(
        input_dim=runtime.model_dim,
        hidden_dim=runtime.model_dim,
        output_dim=runtime.model_dim,
        num_layers=options.num_layers,
        apply_output_pipeline_flag=True,
        last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
        shared_gate_config=None,
        shared_halting_config=shared_halting_config,
        shared_memory_config=_memory(runtime.model_dim, options.memory_flag),
        layer_config=layer_config,
    )
    if not options.recurrent_flag:
        return stack
    return RecurrentLayerConfig(
        input_dim=runtime.model_dim,
        output_dim=runtime.model_dim,
        max_steps=options.recurrent_max_steps,
        recurrent_layer_norm_position=LayerNormPositionOptions.DISABLED,
        block_config=stack,
        gate_config=_gate(runtime.model_dim, options.recurrent_gate_flag),
        residual_config=None
        if options.recurrent_residual_connection_option is None
        else ResidualConfig(option=options.recurrent_residual_connection_option),
        halting_config=_halting(
            runtime.model_dim,
            options.recurrent_halting_flag,
            options.recurrent_halting_option,
            options.recurrent_halting_threshold,
        ),
        memory_config=None,
    )


def _encoder(runtime: RuntimeOptions):
    options = runtime.encoder_options
    transformer_layer = TransformerEncoderLayerConfig(
        embedding_dim=runtime.model_dim,
        layer_norm_position=options.layer_norm_position,
        dropout_probability=runtime.dropout_probability,
        residual_config=ResidualConfig(option=ResidualConnectionOptions.RESIDUAL),
        attention_config=_self_attention(
            runtime,
            runtime.encoder_attention_options,
            maximum_length=runtime.source_sequence_length,
            causal=False,
        ),
        feed_forward_config=_feed_forward(
            runtime, runtime.encoder_feed_forward_options
        ),
    )
    layer = TransformerEncoderBlockLayerConfig(
        input_dim=runtime.model_dim,
        output_dim=runtime.model_dim,
        activation=ActivationOptions.DISABLED,
        residual_config=None
        if options.stack_residual_connection_option is None
        else ResidualConfig(option=options.stack_residual_connection_option),
        dropout_probability=0.0,
        layer_norm_position=LayerNormPositionOptions.DISABLED,
        gate_config=_gate(runtime.model_dim, options.stack_gate_flag),
        halting_config=_halting(
            runtime.model_dim,
            options.stack_halting_flag,
            options.halting_option,
            options.halting_threshold,
        ),
        memory_config=None,
        layer_model_config=transformer_layer,
    )
    return _controlled_stack(runtime, options, layer)


def _decoder(runtime: RuntimeOptions):
    options = runtime.decoder_options
    transformer_layer = TransformerDecoderLayerConfig(
        embedding_dim=runtime.model_dim,
        layer_norm_position=options.layer_norm_position,
        dropout_probability=runtime.dropout_probability,
        residual_config=ResidualConfig(option=ResidualConnectionOptions.RESIDUAL),
        self_attention_config=_self_attention(
            runtime,
            runtime.decoder_self_attention_options,
            maximum_length=runtime.target_sequence_length,
            causal=True,
        ),
        cross_attention_config=_cross_attention(
            runtime, runtime.decoder_cross_attention_options
        ),
        feed_forward_config=_feed_forward(
            runtime, runtime.decoder_feed_forward_options
        ),
    )
    layer = TransformerDecoderBlockLayerConfig(
        input_dim=runtime.model_dim,
        output_dim=runtime.model_dim,
        activation=ActivationOptions.DISABLED,
        residual_config=None
        if options.stack_residual_connection_option is None
        else ResidualConfig(option=options.stack_residual_connection_option),
        dropout_probability=0.0,
        layer_norm_position=LayerNormPositionOptions.DISABLED,
        gate_config=_gate(runtime.model_dim, options.stack_gate_flag),
        halting_config=_halting(
            runtime.model_dim,
            options.stack_halting_flag,
            options.halting_option,
            options.halting_threshold,
        ),
        memory_config=None,
        layer_model_config=transformer_layer,
    )
    return _controlled_stack(runtime, options, layer)


def build_experiment_config(runtime: RuntimeOptions) -> ExperimentConfig:
    positional = runtime.positional_embedding_option
    source_position = positional(
        num_embeddings=runtime.source_sequence_length,
        embedding_dim=runtime.model_dim,
        init_size=runtime.source_sequence_length,
        padding_idx=0,
        auto_expand_flag=False,
    )
    target_position = positional(
        num_embeddings=runtime.target_sequence_length,
        embedding_dim=runtime.model_dim,
        init_size=runtime.target_sequence_length,
        padding_idx=0,
        auto_expand_flag=False,
    )
    return ExperimentConfig(
        source_positional_embedding_config=source_position,
        target_positional_embedding_config=target_position,
        encoder_config=_encoder(runtime),
        decoder_config=_decoder(runtime),
        vocab_size=runtime.vocab_size,
        model_dim=runtime.model_dim,
        source_sequence_length=runtime.source_sequence_length,
        target_sequence_length=runtime.target_sequence_length,
        dropout_probability=runtime.dropout_probability,
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=3,
        label_smoothing=0.1,
        warmup_steps=4_000,
        generation_metrics_flag=True,
    )

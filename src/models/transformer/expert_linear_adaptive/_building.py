from __future__ import annotations

from dataclasses import fields

import torch

from emperor.attention import MixtureOfAttentionHeadsConfig
from emperor.augmentations.adaptive_parameters import (
    AdaptiveLinearLayerConfig,
    AdaptiveParameterAugmentationConfig,
    BankExpansionFactorOptions,
    DynamicDepthOptions,
    MaskDimensionOptions,
    WeightDecayScheduleOptions,
    WeightNormalizationOptions,
    WeightNormalizationPositionOptions,
)
from emperor.experts import (
    ExpertWeightingPositionOptions,
    MixtureOfExpertsConfig,
    MixtureOfExpertsLayerConfig,
    MixtureOfExpertsModelConfig,
    RoutingInitializationMode,
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
from emperor.sampler import RouterConfig, SamplerConfig
from emperor.transformer import (
    FeedForwardConfig,
    TransformerDecoderBlockLayerConfig,
    TransformerDecoderLayerConfig,
    TransformerEncoderBlockLayerConfig,
    TransformerEncoderLayerConfig,
    configure_transformer_submodule,
)

from .experiment_config import ExperimentConfig
from .runtime_options import (
    AdaptiveParameterOptions,
    ExpertOptions,
    RuntimeOptions,
    TransformerAttentionOptions,
    TransformerFeedForwardOptions,
    TransformerStackOptions,
)


def _plain_stack(hidden_dim: int, output_dim: int | None = None):
    return LayerStackConfig(
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=1,
        apply_output_pipeline_flag=False,
        last_layer_bias_option=(
            LastLayerBiasOptions.DEFAULT
            if output_dim is None
            else LastLayerBiasOptions.DISABLED
        ),
        layer_config=LayerConfig(
            activation=ActivationOptions.RELU,
            residual_config=None,
            dropout_probability=0.0,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            gate_config=None,
            halting_config=None,
            memory_config=None,
            layer_model_config=LinearLayerConfig(bias_flag=True),
        ),
    )


def _leaf_config(option: type | None, values: dict):
    if option is None:
        return None
    accepted = {field.name for field in fields(option)}
    return option(**{key: value for key, value in values.items() if key in accepted})


def _adaptive_augmentation(options: AdaptiveParameterOptions):
    generator = _plain_stack(hidden_dim=64)
    weight = _leaf_config(
        options.weight_option,
        {
            "generator_depth": DynamicDepthOptions.DEPTH_OF_ONE,
            "decay_schedule": WeightDecayScheduleOptions.DISABLED,
            "decay_rate": 0.0,
            "decay_warmup_batches": 0,
            "normalization_option": WeightNormalizationOptions.DISABLED,
            "normalization_position_option": (
                WeightNormalizationPositionOptions.DISABLED
            ),
            "bank_expansion_factor": BankExpansionFactorOptions.FACTOR_OF_ONE,
            "model_config": generator,
        },
    )
    bias = _leaf_config(
        options.bias_option,
        {
            "decay_schedule": WeightDecayScheduleOptions.DISABLED,
            "decay_rate": 0.0,
            "decay_warmup_batches": 0,
            "bank_expansion_factor": BankExpansionFactorOptions.FACTOR_OF_ONE,
            "model_config": generator,
        },
    )
    diagonal = _leaf_config(
        options.diagonal_option,
        {"model_config": generator},
    )
    mask = _leaf_config(
        options.row_mask_option,
        {
            "mask_threshold": 0.5,
            "mask_surrogate_scale": 1.0,
            "mask_floor": 0.0,
            "mask_dimension_option": MaskDimensionOptions.ROW,
            "mask_transition_width": 0.1,
            "model_config": generator,
        },
    )
    return AdaptiveParameterAugmentationConfig(
        weight_config=weight,
        bias_config=bias,
        diagonal_config=diagonal,
        mask_config=mask,
        model_config=generator,
    )


def _adaptive_stack(
    *,
    hidden_dim: int | None = None,
    num_layers: int = 1,
    bias_flag: bool = True,
    activation: ActivationOptions = ActivationOptions.RELU,
    dropout_probability: float = 0.0,
    adaptive_options: AdaptiveParameterOptions,
    stack_options=None,
):
    if stack_options is not None:
        hidden_dim = stack_options.hidden_dim
        num_layers = stack_options.num_layers
        bias_flag = stack_options.bias_flag
        activation = stack_options.activation
        dropout_probability = stack_options.dropout_probability
        apply_output_pipeline_flag = stack_options.apply_output_pipeline_flag
        last_layer_bias_option = stack_options.last_layer_bias_option
        residual_connection_option = stack_options.residual_connection_option
        layer_norm_position = stack_options.layer_norm_position
    else:
        apply_output_pipeline_flag = False
        last_layer_bias_option = LastLayerBiasOptions.DEFAULT
        residual_connection_option = None
        layer_norm_position = LayerNormPositionOptions.DISABLED
    return LayerStackConfig(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        apply_output_pipeline_flag=apply_output_pipeline_flag,
        last_layer_bias_option=last_layer_bias_option,
        layer_config=LayerConfig(
            activation=activation,
            residual_config=None
            if residual_connection_option is None
            else ResidualConfig(option=residual_connection_option),
            dropout_probability=dropout_probability,
            layer_norm_position=layer_norm_position,
            gate_config=None,
            halting_config=None,
            memory_config=None,
            layer_model_config=AdaptiveLinearLayerConfig(
                bias_flag=bias_flag,
                adaptive_augmentation_config=_adaptive_augmentation(adaptive_options),
            ),
        ),
    )


def _gate(model_dim: int, enabled: bool):
    if not enabled:
        return None
    return GateConfig(
        option=LayerGateOptions.MULTIPLIER,
        activation=ActivationOptions.SIGMOID,
        model_config=_plain_stack(model_dim),
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
        halting_gate_config=_plain_stack(model_dim, output_dim=2),
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
        model_config=_plain_stack(model_dim),
    )


def _sampler(
    runtime: RuntimeOptions,
    input_dim: int,
    options: ExpertOptions,
) -> SamplerConfig:
    router = RouterConfig(
        input_dim=input_dim,
        num_experts=options.num_experts,
        noisy_topk_flag=False,
        model_config=_adaptive_stack(
            hidden_dim=max(input_dim, options.num_experts),
            adaptive_options=runtime.router_adaptive_options,
        ),
    )
    return SamplerConfig(
        top_k=options.top_k,
        threshold=0.0,
        filter_above_threshold=False,
        num_topk_samples=0,
        normalize_probabilities_flag=options.normalize_probabilities_flag,
        noisy_topk_flag=False,
        num_experts=options.num_experts,
        coefficient_of_variation_loss_weight=0.0,
        switch_loss_weight=options.switch_loss_weight,
        zero_centred_loss_weight=0.0,
        mutual_information_loss_weight=0.0,
        router_config=router,
    )


def _experts_config(
    runtime: RuntimeOptions,
    input_dim: int,
    output_dim: int,
    options: ExpertOptions,
    adaptive_options: AdaptiveParameterOptions,
    *,
    bias_flag: bool = True,
):
    return MixtureOfExpertsConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        top_k=options.top_k,
        num_experts=options.num_experts,
        capacity_factor=options.capacity_factor,
        dropped_token_behavior=None,
        compute_expert_mixture_flag=True,
        weighted_parameters_flag=False,
        weighting_position_option=ExpertWeightingPositionOptions.BEFORE_EXPERTS,
        routing_initialization_mode=RoutingInitializationMode.LAYER,
        sampler_config=_sampler(runtime, input_dim, options),
        expert_model_config=_adaptive_stack(
            hidden_dim=max(input_dim, output_dim),
            bias_flag=bias_flag,
            adaptive_options=adaptive_options,
        ),
    )


def _attention(
    runtime: RuntimeOptions,
    attention_options: TransformerAttentionOptions,
    *,
    target_length: int,
    source_length: int,
    causal: bool,
    self_attention: bool,
):
    experts = runtime.attention_expert_options
    projection_stack = _adaptive_stack(
        stack_options=attention_options.stack_options,
        adaptive_options=runtime.attention_projection_adaptive_options,
    )
    projection_config = configure_transformer_submodule(
        projection_stack,
        control_stack=projection_stack,
        path_options=attention_options,
        model_dim=runtime.model_dim,
    )
    return MixtureOfAttentionHeadsConfig(
        batch_size=runtime.batch_size,
        num_heads=attention_options.num_heads,
        embedding_dim=runtime.model_dim,
        query_key_projection_dim=runtime.model_dim,
        value_projection_dim=runtime.model_dim,
        target_sequence_length=target_length,
        source_sequence_length=source_length,
        target_dtype=torch.float32,
        dropout_probability=runtime.dropout_probability,
        zero_attention_flag=attention_options.zero_attention_flag,
        causal_attention_mask_flag=causal,
        add_key_value_bias_flag=attention_options.add_key_value_bias_flag,
        average_attention_weights_flag=True,
        return_attention_weights_flag=False,
        batch_first_flag=True,
        projection_model_config=projection_config,
        relative_positional_embedding_config=None,
        experts_config=_experts_config(
            runtime,
            runtime.model_dim,
            runtime.model_dim,
            experts,
            runtime.attention_expert_adaptive_options,
        ),
        use_kv_expert_models_flag=self_attention,
    )


def _expert_feed_forward(
    runtime: RuntimeOptions,
    feed_forward_options: TransformerFeedForwardOptions,
):
    options = runtime.feed_forward_expert_options
    stack_options = feed_forward_options.stack_options
    leaf = _experts_config(
        runtime,
        runtime.model_dim,
        runtime.model_dim,
        options,
        runtime.feed_forward_adaptive_options,
        bias_flag=stack_options.bias_flag,
    )
    stack = LayerStackConfig(
        input_dim=runtime.model_dim,
        hidden_dim=stack_options.hidden_dim,
        output_dim=runtime.model_dim,
        num_layers=stack_options.num_layers,
        apply_output_pipeline_flag=(stack_options.apply_output_pipeline_flag),
        last_layer_bias_option=stack_options.last_layer_bias_option,
        shared_gate_config=None,
        shared_halting_config=None,
        shared_memory_config=None,
        layer_config=MixtureOfExpertsLayerConfig(
            activation=stack_options.activation,
            residual_config=None
            if (stack_options.residual_connection_option) is None
            else ResidualConfig(option=(stack_options.residual_connection_option)),
            dropout_probability=stack_options.dropout_probability,
            layer_norm_position=stack_options.layer_norm_position,
            gate_config=None,
            halting_config=None,
            memory_config=None,
            layer_model_config=leaf,
        ),
    )
    model = MixtureOfExpertsModelConfig(
        input_dim=runtime.model_dim,
        output_dim=runtime.model_dim,
        top_k=options.top_k,
        routing_initialization_mode=RoutingInitializationMode.LAYER,
        sampler_config=_sampler(runtime, runtime.model_dim, options),
        stack_config=stack,
    )
    configured = configure_transformer_submodule(
        model,
        control_stack=stack,
        path_options=feed_forward_options,
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
    layer_config,
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
    inner = TransformerEncoderLayerConfig(
        embedding_dim=runtime.model_dim,
        layer_norm_position=options.layer_norm_position,
        dropout_probability=runtime.dropout_probability,
        residual_config=ResidualConfig(option=ResidualConnectionOptions.RESIDUAL),
        causal_attention_mask_flag=False,
        attention_config=_attention(
            runtime,
            runtime.encoder_attention_options,
            target_length=runtime.source_sequence_length,
            source_length=runtime.source_sequence_length,
            causal=False,
            self_attention=True,
        ),
        feed_forward_config=_expert_feed_forward(
            runtime,
            runtime.encoder_feed_forward_options,
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
        layer_model_config=inner,
    )
    return _controlled_stack(runtime, options, layer)


def _decoder(runtime: RuntimeOptions):
    options = runtime.decoder_options
    inner = TransformerDecoderLayerConfig(
        embedding_dim=runtime.model_dim,
        layer_norm_position=options.layer_norm_position,
        dropout_probability=runtime.dropout_probability,
        residual_config=ResidualConfig(option=ResidualConnectionOptions.RESIDUAL),
        causal_attention_mask_flag=True,
        self_attention_config=_attention(
            runtime,
            runtime.decoder_self_attention_options,
            target_length=runtime.target_sequence_length,
            source_length=runtime.target_sequence_length,
            causal=True,
            self_attention=True,
        ),
        cross_attention_config=_attention(
            runtime,
            runtime.decoder_cross_attention_options,
            target_length=runtime.target_sequence_length,
            source_length=runtime.source_sequence_length,
            causal=False,
            self_attention=False,
        ),
        feed_forward_config=_expert_feed_forward(
            runtime,
            runtime.decoder_feed_forward_options,
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
        layer_model_config=inner,
    )
    return _controlled_stack(runtime, options, layer)


def build_experiment_config(runtime: RuntimeOptions) -> ExperimentConfig:
    positional = runtime.positional_embedding_option
    position_kwargs = {
        "embedding_dim": runtime.model_dim,
        "padding_idx": 0,
        "auto_expand_flag": False,
    }
    return ExperimentConfig(
        source_positional_embedding_config=positional(
            num_embeddings=runtime.source_sequence_length,
            init_size=runtime.source_sequence_length,
            **position_kwargs,
        ),
        target_positional_embedding_config=positional(
            num_embeddings=runtime.target_sequence_length,
            init_size=runtime.target_sequence_length,
            **position_kwargs,
        ),
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

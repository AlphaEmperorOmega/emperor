from __future__ import annotations

from dataclasses import fields

import torch

from emperor.attention import MixtureOfAttentionHeadsConfig
from emperor.augmentations.adaptive_parameters import (
    AdaptiveLinearLayerConfig,
    AdaptiveParameterAugmentationConfig,
    MatrixBiasMixtureConfig,
    MatrixWeightsMixtureConfig,
)
from emperor.experts import (
    MixtureOfExpertsConfig,
    MixtureOfExpertsLayerConfig,
    MixtureOfExpertsModelConfig,
)
from emperor.halting import HaltingConfig, HaltingHiddenStateModeOptions
from emperor.layers import (
    ActivationOptions,
    AdditiveResidualConfig,
    GateConfig,
    LastLayerBiasOptions,
    LayerConfig,
    LayerGateOptions,
    LayerNormPositionOptions,
    LayerStackConfig,
    RecurrentLayerConfig,
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
)

from ._residual import (
    ResidualStackOptions,
    ResidualStackSource,
    build_residual_config,
    resolve_residual_stack_options,
)
from ._transformer_submodule import configure_transformer_submodule
from .experiment_config import ExperimentConfig
from .runtime_options import (
    AdaptiveParameterOptions,
    ExpertOptions,
    RuntimeOptions,
    TransformerAttentionOptions,
    TransformerFeedForwardOptions,
    TransformerStackOptions,
    resolve_controller_stack_options,
)


def _plain_stack(hidden_dim: int, output_dim: int | None = None):
    return LayerStackConfig(
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=1,
        apply_output_postprocessing_flag=False,
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
    if option in (MatrixWeightsMixtureConfig, MatrixBiasMixtureConfig):
        accepted.discard("generator_depth")
    return option(**{key: value for key, value in values.items() if key in accepted})


def _generator_stack(stack_options, residual_stack_options):
    return LayerStackConfig(
        hidden_dim=stack_options.hidden_dim,
        num_layers=stack_options.num_layers,
        apply_output_postprocessing_flag=stack_options.apply_output_postprocessing_flag,
        last_layer_bias_option=stack_options.last_layer_bias_option,
        layer_config=LayerConfig(
            activation=stack_options.activation,
            residual_config=build_residual_config(
                stack_options.residual_connection_option,
                stack_options.residual_model_flag,
                residual_stack_options,
            ),
            dropout_probability=stack_options.dropout_probability,
            layer_norm_position=stack_options.layer_norm_position,
            gate_config=None,
            halting_config=None,
            memory_config=None,
            layer_model_config=LinearLayerConfig(bias_flag=stack_options.bias_flag),
        ),
    )


def _component_generator_stack(
    source,
    defaults,
    residual_stack_options,
):
    if not source.independent_flag:
        return None
    return _generator_stack(
        resolve_controller_stack_options(source, defaults),
        residual_stack_options,
    )


def _adaptive_augmentation(
    options: AdaptiveParameterOptions,
    residual_stack_options: ResidualStackOptions,
):
    generator = _generator_stack(
        options.generator_stack_options,
        residual_stack_options,
    )
    factor_defaults = resolve_controller_stack_options(
        options.weight_generator_stack_options, options.generator_stack_options
    )
    weight = _leaf_config(
        options.weight_option if options.weight_option_flag else None,
        {
            "num_experts": options.weight_mixture_num_experts,
            "top_k": options.weight_mixture_top_k,
            "sampler_config": SamplerConfig(
                normalize_probabilities_flag=options.weight_mixture_normalize_probabilities_flag,
                router_config=RouterConfig(
                    model_config=_component_generator_stack(
                        options.weight_mixture_router_generator_stack_options,
                        resolve_controller_stack_options(
                            options.weight_generator_stack_options,
                            options.generator_stack_options,
                        ),
                        residual_stack_options,
                    )
                ),
            ),
            "generator_depth": options.generator_depth,
            "input_factor_source": options.weight_input_factor_source,
            "output_factor_source": options.weight_output_factor_source,
            "input_factor_model_config": _component_generator_stack(
                options.weight_input_factor_generator_stack_options,
                factor_defaults,
                residual_stack_options,
            ),
            "output_factor_model_config": _component_generator_stack(
                options.weight_output_factor_generator_stack_options,
                factor_defaults,
                residual_stack_options,
            ),
            "coefficient_model_config": _component_generator_stack(
                options.weight_coefficient_generator_stack_options,
                factor_defaults,
                residual_stack_options,
            ),
            "decay_schedule": options.weight_decay_schedule,
            "decay_rate": options.weight_decay_rate,
            "decay_warmup_batches": options.weight_decay_warmup_batches,
            "normalization_option": options.weight_normalization_option,
            "normalization_position_option": (
                options.weight_normalization_position_option
            ),
            "bank_expansion_factor": options.weight_bank_expansion_factor,
            "model_config": _component_generator_stack(
                options.weight_generator_stack_options,
                options.generator_stack_options,
                residual_stack_options,
            ),
        },
    )
    bias = _leaf_config(
        options.bias_option if options.bias_option_flag else None,
        {
            "num_experts": options.bias_mixture_num_experts,
            "top_k": options.bias_mixture_top_k,
            "sampler_config": SamplerConfig(
                normalize_probabilities_flag=options.bias_mixture_normalize_probabilities_flag,
                router_config=RouterConfig(
                    model_config=_component_generator_stack(
                        options.bias_mixture_router_generator_stack_options,
                        resolve_controller_stack_options(
                            options.bias_generator_stack_options,
                            options.generator_stack_options,
                        ),
                        residual_stack_options,
                    )
                ),
            ),
            "decay_schedule": options.bias_decay_schedule,
            "decay_rate": options.bias_decay_rate,
            "decay_warmup_batches": options.bias_decay_warmup_batches,
            "bank_expansion_factor": options.bias_bank_expansion_factor,
            "model_config": _component_generator_stack(
                options.bias_generator_stack_options,
                options.generator_stack_options,
                residual_stack_options,
            ),
        },
    )
    diagonal = _leaf_config(
        options.diagonal_option if options.diagonal_option_flag else None,
        {
            "model_config": _component_generator_stack(
                options.diagonal_generator_stack_options,
                options.generator_stack_options,
                residual_stack_options,
            )
        },
    )
    mask = _leaf_config(
        options.row_mask_option if options.mask_option_flag else None,
        {
            "mask_threshold": options.mask_threshold,
            "mask_surrogate_scale": options.mask_surrogate_scale,
            "mask_floor": options.mask_floor,
            "mask_dimension_option": options.mask_dimension_option,
            "mask_transition_width": options.mask_transition_width,
            "model_config": _component_generator_stack(
                options.mask_generator_stack_options,
                options.generator_stack_options,
                residual_stack_options,
            ),
        },
    )
    return AdaptiveParameterAugmentationConfig(
        grouping_config=options.grouping_config,
        weight_config=weight,
        bias_config=bias,
        diagonal_config=diagonal,
        mask_config=mask,
        model_config=generator,
    )


def _residual_stack(runtime: RuntimeOptions) -> ResidualStackOptions:
    return resolve_residual_stack_options(
        ResidualStackSource(
            independent_flag=runtime.residual_stack_independent_flag,
            hidden_dim=runtime.residual_stack_hidden_dim,
            num_layers=runtime.residual_stack_num_layers,
            activation=runtime.residual_stack_activation,
            layer_norm_position=runtime.residual_stack_layer_norm_position,
            residual_connection_option=(
                runtime.residual_stack_residual_connection_option
            ),
            residual_model_flag=runtime.residual_stack_residual_model_flag,
            dropout_probability=runtime.residual_stack_dropout_probability,
            last_layer_bias_option=(runtime.residual_stack_last_layer_bias_option),
            apply_output_postprocessing_flag=(
                runtime.residual_stack_apply_output_postprocessing_flag
            ),
            bias_flag=runtime.residual_stack_bias_flag,
        ),
        runtime.encoder_attention_options.stack_options,
    )


def _adaptive_stack(
    *,
    hidden_dim: int | None = None,
    num_layers: int = 1,
    bias_flag: bool = True,
    activation: ActivationOptions = ActivationOptions.RELU,
    dropout_probability: float = 0.0,
    adaptive_options: AdaptiveParameterOptions,
    residual_stack_options: ResidualStackOptions,
    stack_options=None,
):
    if stack_options is not None:
        hidden_dim = stack_options.hidden_dim
        num_layers = stack_options.num_layers
        bias_flag = stack_options.bias_flag
        activation = stack_options.activation
        dropout_probability = stack_options.dropout_probability
        apply_output_postprocessing_flag = (
            stack_options.apply_output_postprocessing_flag
        )
        last_layer_bias_option = stack_options.last_layer_bias_option
        residual_connection_option = stack_options.residual_connection_option
        residual_model_flag = stack_options.residual_model_flag
        layer_norm_position = stack_options.layer_norm_position
    else:
        apply_output_postprocessing_flag = False
        last_layer_bias_option = LastLayerBiasOptions.DEFAULT
        residual_connection_option = None
        residual_model_flag = False
        layer_norm_position = LayerNormPositionOptions.DISABLED
    return LayerStackConfig(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        apply_output_postprocessing_flag=apply_output_postprocessing_flag,
        last_layer_bias_option=last_layer_bias_option,
        layer_config=LayerConfig(
            activation=activation,
            residual_config=build_residual_config(
                residual_connection_option,
                residual_model_flag,
                residual_stack_options,
            ),
            dropout_probability=dropout_probability,
            layer_norm_position=layer_norm_position,
            gate_config=None,
            halting_config=None,
            memory_config=None,
            layer_model_config=AdaptiveLinearLayerConfig(
                bias_flag=bias_flag,
                adaptive_augmentation_config=_adaptive_augmentation(
                    adaptive_options,
                    residual_stack_options,
                ),
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
        min_steps=1,
        ponder_cost_weight=1.0,
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
    router_stack = _adaptive_stack(
        stack_options=options.router_path_options.stack_options,
        adaptive_options=runtime.router_adaptive_options,
        residual_stack_options=_residual_stack(runtime),
    )
    router = RouterConfig(
        input_dim=input_dim,
        num_experts=options.num_experts,
        noisy_topk_flag=options.router_noisy_topk_flag,
        model_config=configure_transformer_submodule(
            router_stack,
            control_stack=router_stack,
            path_options=options.router_path_options,
            model_dim=runtime.model_dim,
            residual_stack_options=_residual_stack(runtime),
        ),
    )
    return SamplerConfig(
        top_k=options.top_k,
        threshold=options.sampler_threshold,
        filter_above_threshold=options.sampler_filter_above_threshold,
        num_topk_samples=options.sampler_num_topk_samples,
        normalize_probabilities_flag=options.normalize_probabilities_flag,
        noisy_topk_flag=options.sampler_noisy_topk_flag,
        num_experts=options.num_experts,
        coefficient_of_variation_loss_weight=(
            options.coefficient_of_variation_loss_weight
        ),
        switch_loss_weight=options.switch_loss_weight,
        zero_centred_loss_weight=options.zero_centred_loss_weight,
        mutual_information_loss_weight=options.mutual_information_loss_weight,
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
    expert_stack = _adaptive_stack(
        stack_options=options.expert_path_options.stack_options,
        adaptive_options=adaptive_options,
        residual_stack_options=_residual_stack(runtime),
    )
    return MixtureOfExpertsConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        top_k=options.top_k,
        num_experts=options.num_experts,
        capacity_factor=options.capacity_factor,
        dropped_token_behavior=options.dropped_token_behavior,
        compute_expert_mixture_flag=options.compute_expert_mixture_flag,
        weighted_parameters_flag=options.weighted_parameters_flag,
        weighting_position_option=options.weighting_position_option,
        routing_initialization_mode=options.routing_initialization_mode,
        sampler_config=_sampler(runtime, input_dim, options),
        expert_model_config=configure_transformer_submodule(
            expert_stack,
            control_stack=expert_stack,
            path_options=options.expert_path_options,
            model_dim=runtime.model_dim,
            residual_stack_options=_residual_stack(runtime),
        ),
    )


def _attention(
    runtime: RuntimeOptions,
    attention_options: TransformerAttentionOptions,
    projection_adaptive_options: AdaptiveParameterOptions,
    attention_expert_adaptive_options: AdaptiveParameterOptions,
    *,
    target_length: int,
    source_length: int,
    causal: bool,
    self_attention: bool,
):
    experts = runtime.attention_expert_options
    projection_stack = _adaptive_stack(
        stack_options=attention_options.stack_options,
        adaptive_options=projection_adaptive_options,
        residual_stack_options=_residual_stack(runtime),
    )
    projection_config = configure_transformer_submodule(
        projection_stack,
        control_stack=projection_stack,
        path_options=attention_options,
        model_dim=runtime.model_dim,
        residual_stack_options=_residual_stack(runtime),
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
            attention_expert_adaptive_options,
        ),
        use_kv_expert_models_flag=(
            self_attention
            if experts.use_kv_expert_models_flag is None
            else experts.use_kv_expert_models_flag
        ),
    )


def _expert_feed_forward(
    runtime: RuntimeOptions,
    feed_forward_options: TransformerFeedForwardOptions,
    adaptive_options: AdaptiveParameterOptions,
):
    options = runtime.feed_forward_expert_options
    stack_options = feed_forward_options.stack_options
    leaf = _experts_config(
        runtime,
        runtime.model_dim,
        runtime.model_dim,
        options,
        adaptive_options,
        bias_flag=stack_options.bias_flag,
    )
    stack = LayerStackConfig(
        input_dim=runtime.model_dim,
        hidden_dim=stack_options.hidden_dim,
        output_dim=runtime.model_dim,
        num_layers=stack_options.num_layers,
        apply_output_postprocessing_flag=(
            stack_options.apply_output_postprocessing_flag
        ),
        last_layer_bias_option=stack_options.last_layer_bias_option,
        shared_gate_config=None,
        shared_halting_config=None,
        shared_memory_config=None,
        layer_config=MixtureOfExpertsLayerConfig(
            activation=stack_options.activation,
            residual_config=build_residual_config(
                stack_options.residual_connection_option,
                stack_options.residual_model_flag,
                _residual_stack(runtime),
            ),
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
        routing_initialization_mode=options.routing_initialization_mode,
        sampler_config=_sampler(runtime, runtime.model_dim, options),
        stack_config=stack,
    )
    configured = configure_transformer_submodule(
        model,
        control_stack=stack,
        path_options=feed_forward_options,
        model_dim=runtime.model_dim,
        residual_stack_options=_residual_stack(runtime),
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
        apply_output_postprocessing_flag=True,
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
        gradient_transition_count=options.recurrent_gradient_transition_count,
        no_gradient_transition_count=options.recurrent_no_gradient_transition_count,
        initial_iterations=options.recurrent_initial_iterations,
        iteration_increment=options.recurrent_iteration_increment,
        forward_calls_before_iteration_increment=(
            options.recurrent_forward_calls_before_iteration_increment
        ),
        smooth_iteration_growth_flag=(options.recurrent_smooth_iteration_growth_flag),
        recurrent_layer_norm_position=LayerNormPositionOptions.DISABLED,
        block_config=stack,
        gate_config=_gate(runtime.model_dim, options.recurrent_stack_gate_flag),
        residual_config=build_residual_config(
            options.recurrent_residual_connection_option,
            options.recurrent_residual_model_flag,
            _residual_stack(runtime),
        ),
        halting_config=_halting(
            runtime.model_dim,
            options.recurrent_stack_halting_flag,
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
        residual_config=AdditiveResidualConfig(),
        attention_config=_attention(
            runtime,
            runtime.encoder_attention_options,
            runtime.encoder_attention_adaptive_options,
            runtime.encoder_attention_expert_adaptive_options
            or runtime.attention_expert_adaptive_options,
            target_length=runtime.source_sequence_length,
            source_length=runtime.source_sequence_length,
            causal=False,
            self_attention=True,
        ),
        feed_forward_config=_expert_feed_forward(
            runtime,
            runtime.encoder_feed_forward_options,
            runtime.encoder_feed_forward_adaptive_options,
        ),
    )
    layer = TransformerEncoderBlockLayerConfig(
        input_dim=runtime.model_dim,
        output_dim=runtime.model_dim,
        activation=ActivationOptions.DISABLED,
        residual_config=build_residual_config(
            options.stack_residual_connection_option,
            options.stack_residual_model_flag,
            _residual_stack(runtime),
        ),
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
        residual_config=AdditiveResidualConfig(),
        self_attention_config=_attention(
            runtime,
            runtime.decoder_self_attention_options,
            runtime.decoder_self_attention_adaptive_options,
            runtime.decoder_self_attention_expert_adaptive_options
            or runtime.attention_expert_adaptive_options,
            target_length=runtime.target_sequence_length,
            source_length=runtime.target_sequence_length,
            causal=True,
            self_attention=True,
        ),
        cross_attention_config=_attention(
            runtime,
            runtime.decoder_cross_attention_options,
            runtime.decoder_cross_attention_adaptive_options,
            runtime.decoder_cross_attention_expert_adaptive_options
            or runtime.attention_expert_adaptive_options,
            target_length=runtime.target_sequence_length,
            source_length=runtime.source_sequence_length,
            causal=False,
            self_attention=False,
        ),
        feed_forward_config=_expert_feed_forward(
            runtime,
            runtime.decoder_feed_forward_options,
            runtime.decoder_feed_forward_adaptive_options,
        ),
    )
    layer = TransformerDecoderBlockLayerConfig(
        input_dim=runtime.model_dim,
        output_dim=runtime.model_dim,
        activation=ActivationOptions.DISABLED,
        residual_config=build_residual_config(
            options.stack_residual_connection_option,
            options.stack_residual_model_flag,
            _residual_stack(runtime),
        ),
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
    available_values = {
        "embedding_dim": runtime.model_dim,
        "padding_idx": 0,
        "auto_expand_flag": False,
    }
    active_fields = {field.name for field in fields(positional)}
    position_kwargs = {
        name: value for name, value in available_values.items() if name in active_fields
    }
    return ExperimentConfig(
        source_positional_embedding_config=positional(
            num_embeddings=runtime.source_sequence_length,
            **position_kwargs,
        ),
        target_positional_embedding_config=positional(
            num_embeddings=runtime.target_sequence_length,
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

from __future__ import annotations

from emperor.attention import MixerAttentionConfig
from emperor.augmentations.adaptive_parameters import (
    AdaptiveLinearLayerConfig,
    AdaptiveParameterAugmentationConfig,
    AdaptiveParameterGroupingScopeOptions,
    AxisMaskConfig,
    DiagonalAxisMaskConfig,
    DualModelDynamicWeightConfig,
    DynamicBiasConfig,
    DynamicDiagonalConfig,
    DynamicWeightConfig,
    HypernetworkDynamicWeightConfig,
    LayeredWeightedBankDynamicWeightConfig,
    LowRankDynamicWeightConfig,
    PerAxisScoreMaskConfig,
    SingleModelDynamicWeightConfig,
    SoftWeightedBankDynamicWeightConfig,
    TopSliceAxisMaskConfig,
    WeightedBankDynamicBiasConfig,
    WeightInformedScoreAxisMaskConfig,
)
from emperor.experts import (
    MixtureOfExpertsConfig,
    MixtureOfExpertsLayerConfig,
    MixtureOfExpertsModelConfig,
    RoutingInitializationMode,
)
from emperor.layers import (
    ActivationOptions,
    GateConfig,
    LastLayerBiasOptions,
    LayerConfig,
    LayerNormPositionOptions,
    LayerStackConfig,
    MirroredLayerStackConfig,
    RecurrentLayerConfig,
)
from emperor.linears import LinearLayerConfig
from emperor.patch import LinearPatchEmbeddingConfig
from emperor.sampler import RouterConfig, SamplerConfig
from emperor.transformer import (
    FeedForwardConfig,
    TransformerEncoderBlockLayerConfig,
    TransformerEncoderLayerConfig,
)

from ._adaptive_options import (
    BiasOptions,
    DiagonalOptions,
    GeneratorStackSource,
    MaskOptions,
    WeightOptions,
    adaptive_options,
)
from ._control_options import (
    ControllerStackSource,
    ControlOptions,
    GateOptions,
    HaltingOptions,
    MemoryOptions,
    StackOptions,
    channel_mixer_control_options,
    expert_control_options,
    main_control_options,
    submodule_stack_options,
    token_mixer_control_options,
)
from ._residual import (
    ResidualStackSource,
    build_residual_config,
    resolve_residual_stack_options,
)
from .runtime_options import RuntimeOptions


def sequence_length(runtime: RuntimeOptions) -> int:
    image_height = runtime.image_height
    patch_size = runtime.image_patch_size
    if image_height <= 0 or patch_size <= 0:
        raise ValueError("image_height and image_patch_size must be positive")
    if image_height % patch_size != 0:
        raise ValueError(
            "image_height must be exactly divisible by image_patch_size, got "
            f"image_height={image_height}, image_patch_size={patch_size}"
        )
    patches_per_side = image_height // patch_size
    return patches_per_side**2


def _residual(runtime: RuntimeOptions, option, model_flag):
    return build_residual_config(
        option,
        model_flag,
        resolve_residual_stack_options(
            ResidualStackSource(
                independent_flag=runtime.residual_stack_independent_flag,
                hidden_dim=runtime.residual_stack_hidden_dim,
                num_layers=runtime.residual_stack_num_layers,
                activation=runtime.residual_stack_activation,
                layer_norm_position=(runtime.residual_stack_layer_norm_position),
                residual_connection_option=(
                    runtime.residual_stack_residual_connection_option
                ),
                residual_model_flag=(runtime.residual_stack_residual_model_flag),
                dropout_probability=(runtime.residual_stack_dropout_probability),
                last_layer_bias_option=(runtime.residual_stack_last_layer_bias_option),
                apply_output_pipeline_flag=(
                    runtime.residual_stack_apply_output_pipeline_flag
                ),
                bias_flag=runtime.residual_stack_bias_flag,
            ),
            submodule_stack_options(runtime),
        ),
    )


def _plain_linear_config(*, bias_flag: bool) -> LinearLayerConfig:
    return LinearLayerConfig(bias_flag=bias_flag)


_WEIGHT_OPTION_FIELDS: dict[type[DynamicWeightConfig], tuple[str, ...]] = {
    SingleModelDynamicWeightConfig: (
        "normalization_option",
        "normalization_position_option",
    ),
    DualModelDynamicWeightConfig: (
        "normalization_option",
        "normalization_position_option",
    ),
    LowRankDynamicWeightConfig: ("normalization_option",),
    HypernetworkDynamicWeightConfig: ("normalization_option",),
    LayeredWeightedBankDynamicWeightConfig: ("bank_expansion_factor",),
    SoftWeightedBankDynamicWeightConfig: ("bank_expansion_factor",),
}

_BIAS_OPTION_FIELDS: dict[type[DynamicBiasConfig], tuple[str, ...]] = {
    WeightedBankDynamicBiasConfig: ("bank_expansion_factor",),
}

_MASK_OPTION_FIELDS: dict[type[AxisMaskConfig], tuple[str, ...]] = {
    WeightInformedScoreAxisMaskConfig: ("mask_dimension_option",),
    PerAxisScoreMaskConfig: ("mask_dimension_option",),
    TopSliceAxisMaskConfig: (
        "mask_dimension_option",
        "mask_transition_width",
    ),
    DiagonalAxisMaskConfig: ("mask_transition_width",),
}


def _generator_stack(
    runtime: RuntimeOptions,
    options: StackOptions,
) -> LayerStackConfig:
    return LayerStackConfig(
        input_dim=None,
        hidden_dim=options.hidden_dim,
        output_dim=None,
        num_layers=options.num_layers,
        apply_output_pipeline_flag=options.apply_output_pipeline_flag,
        last_layer_bias_option=options.last_layer_bias_option,
        shared_gate_config=None,
        shared_halting_config=None,
        shared_memory_config=None,
        layer_config=LayerConfig(
            activation=options.activation,
            residual_config=_residual(
                runtime,
                options.residual_connection_option,
                options.residual_model_flag,
            ),
            dropout_probability=options.dropout_probability,
            layer_norm_position=options.layer_norm_position,
            gate_config=None,
            halting_config=None,
            memory_config=None,
            layer_model_config=_plain_linear_config(bias_flag=options.bias_flag),
        ),
    )


def _independent_generator_stack(
    runtime: RuntimeOptions,
    source: GeneratorStackSource,
    defaults: StackOptions,
) -> LayerStackConfig | None:
    options = source.resolve(defaults)
    if options is None:
        return None
    return _generator_stack(runtime, options)


def _selected_kwargs(
    field_table: dict[type, tuple[str, ...]],
    option: type,
    optional_kwargs: dict[str, object],
) -> dict[str, object]:
    return {name: optional_kwargs[name] for name in field_table.get(option, ())}


def _weight_config(
    runtime: RuntimeOptions,
    options: WeightOptions,
    generator_defaults: StackOptions,
) -> DynamicWeightConfig | None:
    if not options.enabled:
        return None
    option = options.implementation
    if option is None:
        raise ValueError("weight_option must be set when weight_option_flag is True")
    kwargs = {
        "generator_depth": options.generator_depth,
        "decay_schedule": options.decay_schedule,
        "decay_rate": options.decay_rate,
        "decay_warmup_batches": options.decay_warmup_batches,
        "model_config": _independent_generator_stack(
            runtime,
            options.generator_stack,
            generator_defaults,
        ),
    }
    optional_kwargs = {
        "normalization_option": options.normalization_option,
        "normalization_position_option": options.normalization_position_option,
        "bank_expansion_factor": options.bank_expansion_factor,
    }
    kwargs.update(_selected_kwargs(_WEIGHT_OPTION_FIELDS, option, optional_kwargs))
    return option(**kwargs)


def _bias_config(
    runtime: RuntimeOptions,
    options: BiasOptions,
    generator_defaults: StackOptions,
) -> DynamicBiasConfig | None:
    if not options.enabled:
        return None
    option = options.implementation
    if option is None:
        raise ValueError("bias_option must be set when bias_option_flag is True")
    kwargs = {
        "decay_schedule": options.decay_schedule,
        "decay_rate": options.decay_rate,
        "decay_warmup_batches": options.decay_warmup_batches,
        "model_config": _independent_generator_stack(
            runtime,
            options.generator_stack,
            generator_defaults,
        ),
    }
    optional_kwargs = {
        "bank_expansion_factor": options.bank_expansion_factor,
    }
    kwargs.update(_selected_kwargs(_BIAS_OPTION_FIELDS, option, optional_kwargs))
    return option(**kwargs)


def _diagonal_config(
    runtime: RuntimeOptions,
    options: DiagonalOptions,
    generator_defaults: StackOptions,
) -> DynamicDiagonalConfig | None:
    if not options.enabled:
        return None
    option = options.implementation
    if option is None:
        raise ValueError(
            "diagonal_option must be set when diagonal_option_flag is True"
        )
    return option(
        model_config=_independent_generator_stack(
            runtime,
            options.generator_stack,
            generator_defaults,
        )
    )


def _mask_config(
    runtime: RuntimeOptions,
    options: MaskOptions,
    generator_defaults: StackOptions,
) -> AxisMaskConfig | None:
    if not options.enabled:
        return None
    option = options.implementation
    if option is None:
        raise ValueError("row_mask_option must be set when mask_option_flag is True")
    kwargs = {
        "mask_threshold": options.threshold,
        "mask_surrogate_scale": options.surrogate_scale,
        "mask_floor": options.floor,
        "model_config": _independent_generator_stack(
            runtime,
            options.generator_stack,
            generator_defaults,
        ),
    }
    optional_kwargs = {
        "mask_dimension_option": options.dimension_option,
        "mask_transition_width": options.transition_width,
    }
    kwargs.update(_selected_kwargs(_MASK_OPTION_FIELDS, option, optional_kwargs))
    return option(**kwargs)


def _backend_linear_config(
    runtime: RuntimeOptions,
    *,
    bias_flag: bool,
):
    options = adaptive_options(runtime)
    bias_config = _bias_config(runtime, options.bias, options.generator_stack)
    return AdaptiveLinearLayerConfig(
        bias_flag=bias_flag or bias_config is not None,
        adaptive_augmentation_config=AdaptiveParameterAugmentationConfig(
            grouping_scope=AdaptiveParameterGroupingScopeOptions.DISABLED,
            diagonal_config=_diagonal_config(
                runtime,
                options.diagonal,
                options.generator_stack,
            ),
            weight_config=_weight_config(
                runtime,
                options.weight,
                options.generator_stack,
            ),
            bias_config=bias_config,
            mask_config=_mask_config(runtime, options.mask, options.generator_stack),
            model_config=_generator_stack(runtime, options.generator_stack),
        ),
    )


def _affine_stack(
    runtime: RuntimeOptions,
    *,
    input_dim: int | None,
    hidden_dim: int,
    output_dim: int | None,
    num_layers: int,
    activation,
    dropout_probability: float,
    layer_norm_position,
    residual_connection_option,
    residual_model_flag,
    last_layer_bias_option,
    apply_output_pipeline_flag: bool,
    bias_flag: bool,
    mirrored: bool = False,
    backend: bool = True,
    control_options: ControlOptions | None = None,
    control_name: str | None = None,
):
    stack_type = MirroredLayerStackConfig if mirrored else LayerStackConfig
    stack_depth = num_layers
    if mirrored:
        if num_layers <= 0 or num_layers % 2:
            raise ValueError(
                "channel_mixer_num_layers must be a positive even integer, got "
                f"{num_layers}"
            )
        stack_depth = num_layers // 2
    linear_config = (
        _backend_linear_config(runtime, bias_flag=bias_flag)
        if backend
        else _plain_linear_config(bias_flag=bias_flag)
    )
    stack = stack_type(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=stack_depth,
        apply_output_pipeline_flag=apply_output_pipeline_flag,
        last_layer_bias_option=last_layer_bias_option,
        shared_gate_config=None,
        shared_halting_config=None,
        shared_memory_config=None,
        layer_config=LayerConfig(
            activation=activation,
            residual_config=_residual(
                runtime, residual_connection_option, residual_model_flag
            ),
            dropout_probability=dropout_probability,
            layer_norm_position=layer_norm_position,
            gate_config=None,
            halting_config=None,
            memory_config=None,
            layer_model_config=linear_config,
        ),
    )
    if control_options is None:
        return stack
    if input_dim is None or output_dim is None or input_dim != output_dim:
        raise ValueError(
            f"{control_name} controls require equal concrete input/output "
            f"dimensions, got input_dim={input_dim}, output_dim={output_dim}"
        )
    defaults = StackOptions(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        activation=activation,
        dropout_probability=dropout_probability,
        layer_norm_position=layer_norm_position,
        residual_connection_option=residual_connection_option,
        residual_model_flag=residual_model_flag,
        last_layer_bias_option=last_layer_bias_option,
        apply_output_pipeline_flag=apply_output_pipeline_flag,
        bias_flag=bias_flag,
    )
    return _configure_controls(
        runtime,
        options=control_options,
        model_config=stack,
        control_stack=stack,
        defaults=defaults,
        model_dim=input_dim,
    )


def patch_config(runtime: RuntimeOptions) -> LinearPatchEmbeddingConfig:
    projection = _affine_stack(
        runtime,
        input_dim=None,
        hidden_dim=runtime.hidden_dim,
        output_dim=None,
        num_layers=1,
        activation=ActivationOptions.DISABLED,
        dropout_probability=0.0,
        layer_norm_position=LayerNormPositionOptions.DISABLED,
        residual_connection_option=None,
        residual_model_flag=False,
        last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
        apply_output_pipeline_flag=False,
        bias_flag=runtime.patch_bias_flag,
        backend=False,
    )
    return LinearPatchEmbeddingConfig(
        embedding_dim=runtime.hidden_dim,
        num_input_channels=runtime.input_channels,
        patch_size=runtime.image_patch_size,
        dropout_probability=runtime.patch_dropout_probability,
        class_token_flag=False,
        stride=runtime.image_patch_size,
        padding=0,
        embedding_stack_config=projection,
    )


def _router_stack(runtime: RuntimeOptions) -> LayerStackConfig:
    return _affine_stack(
        runtime,
        input_dim=None,
        hidden_dim=runtime.router_stack_hidden_dim,
        output_dim=None,
        num_layers=runtime.router_stack_num_layers,
        activation=runtime.router_stack_activation,
        dropout_probability=runtime.router_stack_dropout_probability,
        layer_norm_position=runtime.router_stack_layer_norm_position,
        residual_connection_option=runtime.router_stack_residual_connection_option,
        residual_model_flag=runtime.router_stack_residual_model_flag,
        last_layer_bias_option=runtime.router_stack_last_layer_bias_option,
        apply_output_pipeline_flag=runtime.router_stack_apply_output_pipeline_flag,
        bias_flag=runtime.router_bias_flag,
        backend=False,
    )


def _sampler_config(runtime: RuntimeOptions) -> SamplerConfig:
    return SamplerConfig(
        top_k=runtime.top_k,
        threshold=runtime.sampler_threshold,
        filter_above_threshold=runtime.sampler_filter_above_threshold,
        num_topk_samples=runtime.sampler_num_topk_samples,
        normalize_probabilities_flag=runtime.sampler_normalize_probabilities_flag,
        noisy_topk_flag=runtime.sampler_noisy_topk_flag,
        num_experts=runtime.num_experts,
        coefficient_of_variation_loss_weight=(
            runtime.sampler_coefficient_of_variation_loss_weight
        ),
        switch_loss_weight=runtime.sampler_switch_loss_weight,
        zero_centred_loss_weight=runtime.sampler_zero_centred_loss_weight,
        mutual_information_loss_weight=(runtime.sampler_mutual_information_loss_weight),
        router_config=RouterConfig(
            input_dim=None,
            num_experts=runtime.num_experts,
            noisy_topk_flag=runtime.router_noisy_topk_flag,
            model_config=_router_stack(runtime),
        ),
    )


def _expert_stack(
    runtime: RuntimeOptions,
) -> LayerStackConfig | RecurrentLayerConfig:
    stack = _affine_stack(
        runtime,
        input_dim=None,
        hidden_dim=runtime.expert_stack_hidden_dim,
        output_dim=None,
        num_layers=runtime.expert_stack_num_layers,
        activation=runtime.expert_stack_activation,
        dropout_probability=runtime.expert_stack_dropout_probability,
        layer_norm_position=runtime.expert_stack_layer_norm_position,
        residual_connection_option=(runtime.expert_stack_residual_connection_option),
        residual_model_flag=runtime.expert_stack_residual_model_flag,
        last_layer_bias_option=runtime.expert_stack_last_layer_bias_option,
        apply_output_pipeline_flag=(runtime.expert_stack_apply_output_pipeline_flag),
        bias_flag=runtime.expert_bias_flag,
        backend=True,
    )
    defaults = StackOptions(
        hidden_dim=runtime.expert_stack_hidden_dim,
        num_layers=runtime.expert_stack_num_layers,
        activation=runtime.expert_stack_activation,
        dropout_probability=runtime.expert_stack_dropout_probability,
        layer_norm_position=runtime.expert_stack_layer_norm_position,
        residual_connection_option=runtime.expert_stack_residual_connection_option,
        residual_model_flag=runtime.expert_stack_residual_model_flag,
        last_layer_bias_option=runtime.expert_stack_last_layer_bias_option,
        apply_output_pipeline_flag=(runtime.expert_stack_apply_output_pipeline_flag),
        bias_flag=runtime.expert_bias_flag,
    )
    return _configure_controls(
        runtime,
        options=expert_control_options(runtime),
        model_config=stack,
        control_stack=stack,
        defaults=defaults,
        model_dim=None,
        shared_halting=False,
        shared_memory=False,
    )


def _mixture_model_config(
    runtime: RuntimeOptions,
    *,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int,
    activation,
    dropout_probability: float,
    layer_norm_position,
    residual_connection_option,
    residual_model_flag,
    last_layer_bias_option,
    apply_output_pipeline_flag: bool,
    mirrored: bool,
    control_options: ControlOptions,
) -> MixtureOfExpertsModelConfig:
    if runtime.routing_initialization_mode not in (
        RoutingInitializationMode.LAYER,
        RoutingInitializationMode.SHARED,
    ):
        raise ValueError(
            "routing_initialization_mode must be LAYER or SHARED for an "
            "MLP-Mixer expert branch."
        )
    layer_sampler_config = (
        _sampler_config(runtime)
        if runtime.routing_initialization_mode == RoutingInitializationMode.LAYER
        else None
    )
    mixture_config = MixtureOfExpertsConfig(
        input_dim=None,
        output_dim=None,
        top_k=runtime.top_k,
        num_experts=runtime.num_experts,
        capacity_factor=runtime.capacity_factor,
        dropped_token_behavior=runtime.dropped_token_behavior,
        compute_expert_mixture_flag=runtime.compute_expert_mixture_flag,
        weighted_parameters_flag=runtime.weighted_parameters_flag,
        weighting_position_option=runtime.weighting_position_option,
        routing_initialization_mode=runtime.routing_initialization_mode,
        sampler_config=layer_sampler_config,
        expert_model_config=_expert_stack(runtime),
    )
    stack_type = MirroredLayerStackConfig if mirrored else LayerStackConfig
    stack_depth = num_layers
    if mirrored:
        if num_layers <= 0 or num_layers % 2:
            raise ValueError(
                "channel_mixer_num_layers must be a positive even integer, got "
                f"{num_layers}"
            )
        stack_depth = num_layers // 2
    mixture_stack = stack_type(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=stack_depth,
        apply_output_pipeline_flag=apply_output_pipeline_flag,
        last_layer_bias_option=last_layer_bias_option,
        shared_gate_config=None,
        shared_halting_config=None,
        shared_memory_config=None,
        layer_config=MixtureOfExpertsLayerConfig(
            activation=activation,
            residual_config=_residual(
                runtime, residual_connection_option, residual_model_flag
            ),
            dropout_probability=dropout_probability,
            layer_norm_position=layer_norm_position,
            gate_config=None,
            halting_config=None,
            memory_config=None,
            layer_model_config=mixture_config,
        ),
    )
    model_config = MixtureOfExpertsModelConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        top_k=runtime.top_k,
        routing_initialization_mode=runtime.routing_initialization_mode,
        sampler_config=(
            _sampler_config(runtime)
            if runtime.routing_initialization_mode == RoutingInitializationMode.SHARED
            else None
        ),
        stack_config=mixture_stack,
    )
    defaults = StackOptions(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        activation=activation,
        dropout_probability=dropout_probability,
        layer_norm_position=layer_norm_position,
        residual_connection_option=residual_connection_option,
        residual_model_flag=residual_model_flag,
        last_layer_bias_option=last_layer_bias_option,
        apply_output_pipeline_flag=apply_output_pipeline_flag,
        bias_flag=runtime.stack_bias_flag,
    )
    return _configure_controls(
        runtime,
        options=control_options,
        model_config=model_config,
        control_stack=mixture_stack,
        defaults=defaults,
        model_dim=input_dim,
    )


def _token_mixing_model(runtime: RuntimeOptions, tokens: int):
    return _mixture_model_config(
        runtime,
        input_dim=tokens,
        hidden_dim=runtime.token_mixer_stack_hidden_dim,
        output_dim=tokens,
        num_layers=runtime.token_mixer_num_layers,
        activation=runtime.token_mixer_stack_activation,
        dropout_probability=runtime.token_mixer_stack_dropout_probability,
        layer_norm_position=runtime.token_mixer_stack_layer_norm_position,
        residual_connection_option=(
            runtime.token_mixer_stack_residual_connection_option
        ),
        residual_model_flag=runtime.token_mixer_stack_residual_model_flag,
        last_layer_bias_option=(runtime.token_mixer_stack_last_layer_bias_option),
        apply_output_pipeline_flag=(
            runtime.token_mixer_stack_apply_output_pipeline_flag
        ),
        mirrored=False,
        control_options=token_mixer_control_options(runtime),
    )


def _channel_mixing_model(runtime: RuntimeOptions):
    return _mixture_model_config(
        runtime,
        input_dim=runtime.hidden_dim,
        hidden_dim=runtime.channel_mixer_stack_hidden_dim,
        output_dim=runtime.hidden_dim,
        num_layers=runtime.channel_mixer_num_layers,
        activation=runtime.channel_mixer_stack_activation,
        dropout_probability=runtime.channel_mixer_stack_dropout_probability,
        layer_norm_position=runtime.channel_mixer_stack_layer_norm_position,
        residual_connection_option=(
            runtime.channel_mixer_stack_residual_connection_option
        ),
        residual_model_flag=runtime.channel_mixer_stack_residual_model_flag,
        last_layer_bias_option=(runtime.channel_mixer_stack_last_layer_bias_option),
        apply_output_pipeline_flag=(
            runtime.channel_mixer_stack_apply_output_pipeline_flag
        ),
        mirrored=True,
        control_options=channel_mixer_control_options(runtime),
    )


def _controller_stack_config(
    runtime: RuntimeOptions,
    *,
    source: ControllerStackSource,
    defaults: StackOptions,
    output_dim: int | None = None,
) -> LayerStackConfig:
    options = source.resolve(defaults)
    return _affine_stack(
        runtime,
        input_dim=None,
        hidden_dim=options.hidden_dim,
        output_dim=output_dim,
        num_layers=options.num_layers,
        activation=options.activation,
        dropout_probability=options.dropout_probability,
        layer_norm_position=options.layer_norm_position,
        residual_connection_option=options.residual_connection_option,
        residual_model_flag=options.residual_model_flag,
        last_layer_bias_option=(
            options.last_layer_bias_option
            if output_dim is None
            else LastLayerBiasOptions.DISABLED
        ),
        apply_output_pipeline_flag=options.apply_output_pipeline_flag,
        bias_flag=options.bias_flag,
        backend=False,
    )


def _configured_gate(
    runtime: RuntimeOptions,
    *,
    options: GateOptions,
    defaults: StackOptions,
    model_dim: int | None,
):
    if not options.enabled:
        return None
    return GateConfig(
        gate_dim=model_dim,
        option=options.option,
        activation=options.activation,
        model_config=_controller_stack_config(
            runtime,
            source=options.stack,
            defaults=defaults,
        ),
    )


def _configured_halting(
    runtime: RuntimeOptions,
    *,
    options: HaltingOptions,
    defaults: StackOptions,
    model_dim: int | None,
):
    if not options.enabled:
        return None
    return options.implementation(
        input_dim=model_dim,
        threshold=options.threshold,
        min_steps=1,
        ponder_cost_weight=1.0,
        dropout_probability=options.dropout_probability,
        hidden_state_mode=options.hidden_state_mode,
        halting_gate_config=_controller_stack_config(
            runtime,
            source=options.stack,
            defaults=defaults,
            output_dim=2,
        ),
    )


def _configured_memory(
    runtime: RuntimeOptions,
    *,
    options: MemoryOptions | None,
    defaults: StackOptions,
    model_dim: int | None,
):
    if options is None or not options.enabled:
        return None
    return options.implementation(
        input_dim=model_dim,
        output_dim=model_dim,
        memory_position_option=options.position,
        test_time_training_learning_rate=(options.test_time_training_learning_rate),
        test_time_training_num_inner_steps=(options.test_time_training_num_inner_steps),
        model_config=_controller_stack_config(
            runtime,
            source=options.stack,
            defaults=defaults,
        ),
    )


def _configure_controls(
    runtime: RuntimeOptions,
    *,
    options: ControlOptions,
    model_config,
    control_stack,
    defaults: StackOptions,
    model_dim: int | None,
    shared_halting: bool = True,
    shared_memory: bool = True,
):
    control_stack.layer_config.gate_config = _configured_gate(
        runtime,
        options=options.gate,
        defaults=defaults,
        model_dim=model_dim,
    )
    halting_config = _configured_halting(
        runtime,
        options=options.halting,
        defaults=defaults,
        model_dim=model_dim,
    )
    if shared_halting:
        control_stack.shared_halting_config = halting_config
    else:
        control_stack.layer_config.halting_config = halting_config
    memory_config = _configured_memory(
        runtime,
        options=options.memory,
        defaults=defaults,
        model_dim=model_dim,
    )
    if shared_memory:
        control_stack.shared_memory_config = memory_config
    else:
        control_stack.layer_config.memory_config = memory_config
    recurrent = options.recurrent
    if not recurrent.enabled:
        return model_config
    return RecurrentLayerConfig(
        input_dim=model_dim,
        output_dim=model_dim,
        max_steps=recurrent.max_steps,
        gradient_transition_count=recurrent.gradient_transition_count,
        initial_iterations=recurrent.initial_iterations,
        iteration_increment=recurrent.iteration_increment,
        forward_calls_before_iteration_increment=(
            recurrent.forward_calls_before_iteration_increment
        ),
        recurrent_layer_norm_position=recurrent.layer_norm_position,
        block_config=model_config,
        gate_config=_configured_gate(
            runtime,
            options=recurrent.gate,
            defaults=defaults,
            model_dim=model_dim,
        ),
        residual_config=_residual(
            runtime,
            recurrent.residual_connection_option,
            recurrent.residual_model_flag,
        ),
        halting_config=_configured_halting(
            runtime,
            options=recurrent.halting,
            defaults=defaults,
            model_dim=model_dim,
        ),
        memory_config=_configured_memory(
            runtime,
            options=recurrent.memory,
            defaults=defaults,
            model_dim=model_dim,
        ),
    )


def encoder_config(runtime: RuntimeOptions, tokens: int):
    mixer_layer = TransformerEncoderLayerConfig(
        embedding_dim=runtime.hidden_dim,
        layer_norm_position=runtime.layer_norm_position,
        dropout_probability=runtime.stack_dropout_probability,
        residual_config=_residual(
            runtime,
            runtime.mixer_residual_connection_option,
            runtime.mixer_residual_model_flag,
        ),
        attention_config=MixerAttentionConfig(
            embedding_dim=runtime.hidden_dim,
            sequence_length=tokens,
            batch_first_flag=True,
            mixing_model_config=_token_mixing_model(runtime, tokens),
        ),
        feed_forward_config=FeedForwardConfig(
            input_dim=runtime.hidden_dim,
            output_dim=runtime.hidden_dim,
            stack_config=_channel_mixing_model(runtime),
        ),
    )
    block_layer = TransformerEncoderBlockLayerConfig(
        activation=ActivationOptions.DISABLED,
        residual_config=_residual(
            runtime,
            runtime.stack_residual_connection_option,
            runtime.stack_residual_model_flag,
        ),
        dropout_probability=0.0,
        layer_norm_position=LayerNormPositionOptions.DISABLED,
        gate_config=None,
        halting_config=None,
        memory_config=None,
        layer_model_config=mixer_layer,
    )
    stack = LayerStackConfig(
        input_dim=runtime.hidden_dim,
        hidden_dim=runtime.hidden_dim,
        output_dim=runtime.hidden_dim,
        num_layers=runtime.stack_num_layers,
        apply_output_pipeline_flag=runtime.stack_apply_output_pipeline_flag,
        last_layer_bias_option=runtime.stack_last_layer_bias_option,
        shared_gate_config=None,
        shared_halting_config=None,
        shared_memory_config=None,
        layer_config=block_layer,
    )
    return _configure_controls(
        runtime,
        options=main_control_options(runtime),
        model_config=stack,
        control_stack=stack,
        defaults=submodule_stack_options(runtime),
        model_dim=runtime.hidden_dim,
    )


def output_config(runtime: RuntimeOptions) -> LayerConfig:
    return LayerConfig(
        input_dim=runtime.hidden_dim,
        output_dim=runtime.output_dim,
        activation=ActivationOptions.DISABLED,
        residual_config=None,
        dropout_probability=0.0,
        layer_norm_position=LayerNormPositionOptions.DISABLED,
        gate_config=None,
        halting_config=None,
        memory_config=None,
        layer_model_config=_plain_linear_config(bias_flag=runtime.output_bias_flag),
    )

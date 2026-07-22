from __future__ import annotations

from dataclasses import dataclass

from emperor.attention import MixerAttentionConfig
from emperor.augmentations.adaptive_parameters import (
    AdaptiveLinearLayerConfig,
    AdaptiveParameterAugmentationConfig,
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
    ResidualConfig,
    ResidualConnectionOptions,
)
from emperor.linears import LinearLayerConfig
from emperor.patch import LinearPatchEmbeddingConfig
from emperor.sampler import RouterConfig, SamplerConfig
from emperor.transformer import (
    FeedForwardConfig,
    TransformerEncoderBlockLayerConfig,
    TransformerEncoderLayerConfig,
)

from .runtime_options import RuntimeOptions


@dataclass(frozen=True)
class _StackOptions:
    hidden_dim: int
    num_layers: int
    activation: ActivationOptions
    dropout_probability: float
    layer_norm_position: LayerNormPositionOptions
    residual_connection_option: ResidualConnectionOptions | None
    last_layer_bias_option: LastLayerBiasOptions
    apply_output_pipeline_flag: bool
    bias_flag: bool


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


def _residual(option):
    return None if option is None else ResidualConfig(option=option)


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


def _adaptive_generator_defaults(runtime: RuntimeOptions) -> _StackOptions:
    return _StackOptions(
        hidden_dim=runtime.adaptive_generator_stack_hidden_dim,
        num_layers=runtime.adaptive_generator_stack_num_layers,
        activation=runtime.adaptive_generator_stack_activation,
        dropout_probability=runtime.adaptive_generator_stack_dropout_probability,
        layer_norm_position=runtime.adaptive_generator_stack_layer_norm_position,
        residual_connection_option=(
            runtime.adaptive_generator_stack_residual_connection_option
        ),
        last_layer_bias_option=(
            runtime.adaptive_generator_stack_last_layer_bias_option
        ),
        apply_output_pipeline_flag=(
            runtime.adaptive_generator_stack_apply_output_pipeline_flag
        ),
        bias_flag=runtime.adaptive_generator_stack_bias_flag,
    )


def _adaptive_generator_options(
    runtime: RuntimeOptions,
    source_prefix: str | None,
) -> _StackOptions:
    defaults = _adaptive_generator_defaults(runtime)
    if source_prefix is None:
        return defaults

    def resolved(field_name: str):
        value = getattr(runtime, f"{source_prefix}_generator_stack_{field_name}")
        return getattr(defaults, field_name) if value is None else value

    return _StackOptions(
        hidden_dim=resolved("hidden_dim"),
        num_layers=resolved("num_layers"),
        activation=resolved("activation"),
        dropout_probability=resolved("dropout_probability"),
        layer_norm_position=resolved("layer_norm_position"),
        residual_connection_option=resolved("residual_connection_option"),
        last_layer_bias_option=resolved("last_layer_bias_option"),
        apply_output_pipeline_flag=resolved("apply_output_pipeline_flag"),
        bias_flag=resolved("bias_flag"),
    )


def _generator_stack(
    runtime: RuntimeOptions,
    *,
    source_prefix: str | None = None,
) -> LayerStackConfig:
    options = _adaptive_generator_options(runtime, source_prefix)
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
            residual_config=_residual(options.residual_connection_option),
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
    source_prefix: str,
) -> LayerStackConfig | None:
    if not getattr(runtime, f"{source_prefix}_generator_stack_independent_flag"):
        return None
    return _generator_stack(runtime, source_prefix=source_prefix)


def _enabled_option(runtime: RuntimeOptions, prefix: str):
    if not getattr(runtime, f"{prefix}_option_flag"):
        return None
    option_field = "row_mask_option" if prefix == "mask" else f"{prefix}_option"
    option = getattr(runtime, option_field)
    if option is None:
        raise ValueError(
            f"{option_field} must be set when {prefix}_option_flag is True"
        )
    return option


def _selected_kwargs(
    field_table: dict[type, tuple[str, ...]],
    option: type,
    optional_kwargs: dict[str, object],
) -> dict[str, object]:
    return {name: optional_kwargs[name] for name in field_table.get(option, ())}


def _weight_config(runtime: RuntimeOptions) -> DynamicWeightConfig | None:
    option = _enabled_option(runtime, "weight")
    if option is None:
        return None
    kwargs = {
        "generator_depth": runtime.generator_depth,
        "decay_schedule": runtime.weight_decay_schedule,
        "decay_rate": runtime.weight_decay_rate,
        "decay_warmup_batches": runtime.weight_decay_warmup_batches,
        "model_config": _independent_generator_stack(runtime, "weight"),
    }
    optional_kwargs = {
        "normalization_option": runtime.weight_normalization_option,
        "normalization_position_option": (runtime.weight_normalization_position_option),
        "bank_expansion_factor": runtime.weight_bank_expansion_factor,
    }
    kwargs.update(_selected_kwargs(_WEIGHT_OPTION_FIELDS, option, optional_kwargs))
    return option(**kwargs)


def _bias_config(runtime: RuntimeOptions) -> DynamicBiasConfig | None:
    option = _enabled_option(runtime, "bias")
    if option is None:
        return None
    kwargs = {
        "decay_schedule": runtime.bias_decay_schedule,
        "decay_rate": runtime.bias_decay_rate,
        "decay_warmup_batches": runtime.bias_decay_warmup_batches,
        "model_config": _independent_generator_stack(runtime, "bias"),
    }
    optional_kwargs = {
        "bank_expansion_factor": runtime.bias_bank_expansion_factor,
    }
    kwargs.update(_selected_kwargs(_BIAS_OPTION_FIELDS, option, optional_kwargs))
    return option(**kwargs)


def _diagonal_config(runtime: RuntimeOptions) -> DynamicDiagonalConfig | None:
    option = _enabled_option(runtime, "diagonal")
    if option is None:
        return None
    return option(model_config=_independent_generator_stack(runtime, "diagonal"))


def _mask_config(runtime: RuntimeOptions) -> AxisMaskConfig | None:
    option = _enabled_option(runtime, "mask")
    if option is None:
        return None
    kwargs = {
        "mask_threshold": runtime.mask_threshold,
        "mask_surrogate_scale": runtime.mask_surrogate_scale,
        "mask_floor": runtime.mask_floor,
        "model_config": _independent_generator_stack(runtime, "mask"),
    }
    optional_kwargs = {
        "mask_dimension_option": runtime.mask_dimension_option,
        "mask_transition_width": runtime.mask_transition_width,
    }
    kwargs.update(_selected_kwargs(_MASK_OPTION_FIELDS, option, optional_kwargs))
    return option(**kwargs)


def _backend_linear_config(
    runtime: RuntimeOptions,
    *,
    bias_flag: bool,
):
    bias_config = _bias_config(runtime)
    return AdaptiveLinearLayerConfig(
        bias_flag=bias_flag or bias_config is not None,
        adaptive_augmentation_config=AdaptiveParameterAugmentationConfig(
            diagonal_config=_diagonal_config(runtime),
            weight_config=_weight_config(runtime),
            bias_config=bias_config,
            mask_config=_mask_config(runtime),
            model_config=_generator_stack(runtime),
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
    last_layer_bias_option,
    apply_output_pipeline_flag: bool,
    bias_flag: bool,
    mirrored: bool = False,
    backend: bool = True,
    control_prefix: str | None = None,
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
            residual_config=_residual(residual_connection_option),
            dropout_probability=dropout_probability,
            layer_norm_position=layer_norm_position,
            gate_config=None,
            halting_config=None,
            memory_config=None,
            layer_model_config=linear_config,
        ),
    )
    if control_prefix is None:
        return stack
    if input_dim is None or output_dim is None or input_dim != output_dim:
        raise ValueError(
            f"{control_prefix} controls require equal concrete input/output "
            f"dimensions, got input_dim={input_dim}, output_dim={output_dim}"
        )
    defaults = _StackOptions(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        activation=activation,
        dropout_probability=dropout_probability,
        layer_norm_position=layer_norm_position,
        residual_connection_option=residual_connection_option,
        last_layer_bias_option=last_layer_bias_option,
        apply_output_pipeline_flag=apply_output_pipeline_flag,
        bias_flag=bias_flag,
    )
    return _configure_controls(
        runtime,
        prefix=control_prefix,
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
        last_layer_bias_option=runtime.expert_stack_last_layer_bias_option,
        apply_output_pipeline_flag=(runtime.expert_stack_apply_output_pipeline_flag),
        bias_flag=runtime.expert_bias_flag,
        backend=True,
    )
    defaults = _StackOptions(
        hidden_dim=runtime.expert_stack_hidden_dim,
        num_layers=runtime.expert_stack_num_layers,
        activation=runtime.expert_stack_activation,
        dropout_probability=runtime.expert_stack_dropout_probability,
        layer_norm_position=runtime.expert_stack_layer_norm_position,
        residual_connection_option=runtime.expert_stack_residual_connection_option,
        last_layer_bias_option=runtime.expert_stack_last_layer_bias_option,
        apply_output_pipeline_flag=(runtime.expert_stack_apply_output_pipeline_flag),
        bias_flag=runtime.expert_bias_flag,
    )
    return _configure_controls(
        runtime,
        prefix="expert",
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
    last_layer_bias_option,
    apply_output_pipeline_flag: bool,
    mirrored: bool,
    control_prefix: str,
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
            residual_config=_residual(residual_connection_option),
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
    defaults = _StackOptions(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        activation=activation,
        dropout_probability=dropout_probability,
        layer_norm_position=layer_norm_position,
        residual_connection_option=residual_connection_option,
        last_layer_bias_option=last_layer_bias_option,
        apply_output_pipeline_flag=apply_output_pipeline_flag,
        bias_flag=runtime.stack_bias_flag,
    )
    return _configure_controls(
        runtime,
        prefix=control_prefix,
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
        last_layer_bias_option=(runtime.token_mixer_stack_last_layer_bias_option),
        apply_output_pipeline_flag=(
            runtime.token_mixer_stack_apply_output_pipeline_flag
        ),
        mirrored=False,
        control_prefix="token_mixer",
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
        last_layer_bias_option=(runtime.channel_mixer_stack_last_layer_bias_option),
        apply_output_pipeline_flag=(
            runtime.channel_mixer_stack_apply_output_pipeline_flag
        ),
        mirrored=True,
        control_prefix="channel_mixer",
    )


def _option_name(prefix: str, suffix: str) -> str:
    return f"{prefix}_{suffix}" if prefix else suffix


def _submodule_stack_defaults(runtime: RuntimeOptions) -> _StackOptions:
    return _StackOptions(
        hidden_dim=runtime.submodule_stack_hidden_dim,
        num_layers=runtime.submodule_stack_num_layers,
        activation=runtime.submodule_stack_activation,
        dropout_probability=runtime.submodule_stack_dropout_probability,
        layer_norm_position=runtime.submodule_stack_layer_norm_position,
        residual_connection_option=(runtime.submodule_stack_residual_connection_option),
        last_layer_bias_option=runtime.submodule_stack_last_layer_bias_option,
        apply_output_pipeline_flag=(runtime.submodule_stack_apply_output_pipeline_flag),
        bias_flag=runtime.submodule_stack_bias_flag,
    )


def _resolved_controller_options(
    runtime: RuntimeOptions,
    *,
    source_prefix: str,
    defaults: _StackOptions,
) -> _StackOptions:
    if not getattr(runtime, f"{source_prefix}_independent_flag"):
        return defaults

    def resolved(field_name: str):
        value = getattr(runtime, f"{source_prefix}_{field_name}")
        return getattr(defaults, field_name) if value is None else value

    return _StackOptions(
        hidden_dim=resolved("hidden_dim"),
        num_layers=resolved("num_layers"),
        activation=resolved("activation"),
        dropout_probability=resolved("dropout_probability"),
        layer_norm_position=resolved("layer_norm_position"),
        residual_connection_option=resolved("residual_connection_option"),
        last_layer_bias_option=resolved("last_layer_bias_option"),
        apply_output_pipeline_flag=resolved("apply_output_pipeline_flag"),
        bias_flag=resolved("bias_flag"),
    )


def _controller_stack_config(
    runtime: RuntimeOptions,
    *,
    source_prefix: str,
    defaults: _StackOptions,
    output_dim: int | None = None,
) -> LayerStackConfig:
    options = _resolved_controller_options(
        runtime,
        source_prefix=source_prefix,
        defaults=defaults,
    )
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
    prefix: str,
    defaults: _StackOptions,
    model_dim: int | None,
    recurrent: bool,
):
    role = "recurrent_" if recurrent else ""
    if not getattr(runtime, _option_name(prefix, f"{role}stack_gate_flag")):
        return None
    return GateConfig(
        gate_dim=model_dim,
        option=getattr(runtime, _option_name(prefix, f"{role}gate_option")),
        activation=getattr(
            runtime,
            _option_name(prefix, f"{role}gate_activation"),
        ),
        model_config=_controller_stack_config(
            runtime,
            source_prefix=_option_name(prefix, f"{role}gate_stack"),
            defaults=defaults,
        ),
    )


def _configured_halting(
    runtime: RuntimeOptions,
    *,
    prefix: str,
    defaults: _StackOptions,
    model_dim: int | None,
    recurrent: bool,
):
    role = "recurrent_" if recurrent else ""
    if not getattr(runtime, _option_name(prefix, f"{role}stack_halting_flag")):
        return None
    option = getattr(runtime, _option_name(prefix, f"{role}halting_option"))
    return option(
        input_dim=model_dim,
        threshold=getattr(
            runtime,
            _option_name(prefix, f"{role}halting_threshold"),
        ),
        dropout_probability=getattr(
            runtime,
            _option_name(prefix, f"{role}halting_dropout"),
        ),
        hidden_state_mode=getattr(
            runtime,
            _option_name(prefix, f"{role}halting_hidden_state_mode"),
        ),
        halting_gate_config=_controller_stack_config(
            runtime,
            source_prefix=_option_name(prefix, f"{role}halting_stack"),
            defaults=defaults,
            output_dim=2,
        ),
    )


def _configured_memory(
    runtime: RuntimeOptions,
    *,
    prefix: str,
    defaults: _StackOptions,
    model_dim: int | None,
    recurrent: bool,
):
    if recurrent:
        if prefix or not runtime.recurrent_memory_flag:
            return None
    elif not getattr(runtime, _option_name(prefix, "memory_flag")):
        return None
    option_prefix = prefix
    option = getattr(runtime, _option_name(option_prefix, "memory_option"))
    return option(
        input_dim=model_dim,
        output_dim=model_dim,
        memory_position_option=getattr(
            runtime,
            _option_name(option_prefix, "memory_position_option"),
        ),
        test_time_training_learning_rate=getattr(
            runtime,
            _option_name(
                option_prefix,
                "memory_test_time_training_learning_rate",
            ),
        ),
        test_time_training_num_inner_steps=getattr(
            runtime,
            _option_name(
                option_prefix,
                "memory_test_time_training_num_inner_steps",
            ),
        ),
        model_config=_controller_stack_config(
            runtime,
            source_prefix=_option_name(option_prefix, "memory_stack"),
            defaults=defaults,
        ),
    )


def _configure_controls(
    runtime: RuntimeOptions,
    *,
    prefix: str,
    model_config,
    control_stack,
    defaults: _StackOptions,
    model_dim: int | None,
    shared_halting: bool = True,
    shared_memory: bool = True,
):
    control_stack.layer_config.gate_config = _configured_gate(
        runtime,
        prefix=prefix,
        defaults=defaults,
        model_dim=model_dim,
        recurrent=False,
    )
    halting_config = _configured_halting(
        runtime,
        prefix=prefix,
        defaults=defaults,
        model_dim=model_dim,
        recurrent=False,
    )
    if shared_halting:
        control_stack.shared_halting_config = halting_config
    else:
        control_stack.layer_config.halting_config = halting_config
    memory_config = _configured_memory(
        runtime,
        prefix=prefix,
        defaults=defaults,
        model_dim=model_dim,
        recurrent=False,
    )
    if shared_memory:
        control_stack.shared_memory_config = memory_config
    else:
        control_stack.layer_config.memory_config = memory_config
    if not getattr(runtime, _option_name(prefix, "recurrent_flag")):
        return model_config
    return RecurrentLayerConfig(
        input_dim=model_dim,
        output_dim=model_dim,
        max_steps=getattr(
            runtime,
            _option_name(prefix, "recurrent_max_steps"),
        ),
        recurrent_layer_norm_position=getattr(
            runtime,
            _option_name(prefix, "recurrent_layer_norm_position"),
        ),
        block_config=model_config,
        gate_config=_configured_gate(
            runtime,
            prefix=prefix,
            defaults=defaults,
            model_dim=model_dim,
            recurrent=True,
        ),
        residual_config=_residual(
            getattr(
                runtime,
                _option_name(prefix, "recurrent_residual_connection_option"),
            )
        ),
        halting_config=_configured_halting(
            runtime,
            prefix=prefix,
            defaults=defaults,
            model_dim=model_dim,
            recurrent=True,
        ),
        memory_config=_configured_memory(
            runtime,
            prefix=prefix,
            defaults=defaults,
            model_dim=model_dim,
            recurrent=True,
        ),
    )


def encoder_config(runtime: RuntimeOptions, tokens: int):
    mixer_layer = TransformerEncoderLayerConfig(
        embedding_dim=runtime.hidden_dim,
        layer_norm_position=runtime.layer_norm_position,
        dropout_probability=runtime.stack_dropout_probability,
        residual_config=_residual(runtime.mixer_residual_connection_option),
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
        residual_config=_residual(runtime.stack_residual_connection_option),
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
        prefix="",
        model_config=stack,
        control_stack=stack,
        defaults=_submodule_stack_defaults(runtime),
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

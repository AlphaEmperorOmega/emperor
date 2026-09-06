"""Concrete public chunking configurations exercised by behavioral suites."""

from emperor.augmentations.adaptive_parameters import (
    AttentionGroupingConfig,
    MeanGroupingConfig,
    MeanStdGroupingConfig,
    RMSGroupingConfig,
    SumGroupingConfig,
)

GROUPING_CONFIGS = (
    SumGroupingConfig,
    MeanGroupingConfig,
    MeanStdGroupingConfig,
    AttentionGroupingConfig,
    RMSGroupingConfig,
)


def grouping_model_config(config_type, *, width=64):
    """Explicit test networks; production groupers do not choose these defaults."""
    from emperor.layers import ActivationOptions, LastLayerBiasOptions
    from support.adaptive_grouping import linear_stack_config

    if config_type is AttentionGroupingConfig:
        config = linear_stack_config(2, 1)
        config.hidden_dim = width
        config.num_layers = 2
        config.layer_config.activation = ActivationOptions.TANH
        config.last_layer_bias_option = LastLayerBiasOptions.DISABLED
        return config
    if config_type is MeanStdGroupingConfig:
        return linear_stack_config(4, 2)
    return None


def grouping_config(config_type, **options):
    from emperor.augmentations.adaptive_parameters import (
        AdaptiveParameterGroupingScopeOptions,
    )

    options.setdefault("scope", AdaptiveParameterGroupingScopeOptions.ROWS)
    if "chunk_size" not in options:
        options.setdefault("group_count", 2)
    model_config = grouping_model_config(config_type)
    if model_config is not None:
        options.setdefault("model_config", model_config)
    return config_type(**options)


def initialize_mean_summary(grouper):
    """Set analytical test fixtures to uniform attention / mean-only projection."""
    import torch

    with torch.no_grad():
        if isinstance(grouper.cfg, AttentionGroupingConfig):
            grouper.scorer[-1].model.weight_params.zero_()
        elif isinstance(grouper.cfg, MeanStdGroupingConfig):
            linear = grouper.projection[-1].model
            linear.weight_params.zero_()
            linear.weight_params[: grouper.feature_dim].copy_(
                torch.eye(grouper.feature_dim, device=linear.weight_params.device)
            )
            linear.bias_params.zero_()
    return grouper


def variable_chunk_stack():
    from emperor.augmentations.adaptive_parameters import (
        AdaptiveParameterGroupingScopeOptions,
    )
    from support.adaptive_grouping import bias_linear, linear_stack_config

    stack = linear_stack_config(2, 2)
    stack.layer_config.layer_model_config = bias_linear(
        MeanGroupingConfig(
            scope=AdaptiveParameterGroupingScopeOptions.ROWS, chunk_size=5
        )
    ).cfg
    return stack


def reference_mixture(model, inputs, probabilities, indices, *, reduce_input=False):
    """Independent real-member affine equations; no production grouping helpers."""
    import torch

    from emperor.experts import DroppedTokenOptions, ExpertWeightingPositionOptions

    top_k = model.top_k
    if indices is None:
        indices = torch.arange(model.num_experts).expand(inputs.size(0), -1)
    indices = indices.reshape(-1)
    probabilities = probabilities.reshape(-1)
    rows = inputs if reduce_input else inputs.repeat_interleave(top_k, dim=0)
    output = torch.zeros_like(rows)
    for expert_index, stack in enumerate(model.expert_modules):
        positions = (indices == expert_index).nonzero().flatten()
        part = rows[positions]
        before = (
            model.weighting_position_option
            is ExpertWeightingPositionOptions.BEFORE_EXPERTS
        )
        if before and model.weighted_parameters_flag:
            part = part * probabilities[positions, None]
        capacity = len(positions)
        if model.capacity_factor and not reduce_input:
            import math

            capacity = max(
                1,
                math.ceil(
                    inputs.size(0) * top_k / model.num_experts * model.capacity_factor
                ),
            )
        linear = stack[0].model
        generator = linear.adaptive_behaviour.bias_model.model[0].model
        computed = []
        for chunk in part[:capacity].split(5):
            if chunk.size(0):
                bias = chunk.mean(0) @ generator.weight_params + generator.bias_params
                computed.append(
                    chunk @ linear.weight_params + linear.bias_params + bias
                )
        if not computed:
            continue
        retained = torch.cat(computed)
        dropped = part[capacity:]
        if model.cfg.dropped_token_behavior is DroppedTokenOptions.ZEROS:
            dropped = torch.zeros_like(dropped)
        output = output.index_copy(0, positions, torch.cat((retained, dropped)))
    if (
        model.weighted_parameters_flag
        and model.weighting_position_option
        is ExpertWeightingPositionOptions.AFTER_EXPERTS
    ):
        output = output * probabilities[:, None]
    if model.compute_expert_mixture_flag:
        output = output.reshape(-1, top_k, inputs.size(-1)).sum(1)
    return output


def with_reduction(layout, variant):
    """Build a fresh test variant using the same token layout."""
    from dataclasses import replace

    return replace(
        variant,
        scope=layout.scope,
        group_count=layout.group_count,
        chunk_size=layout.chunk_size,
        sequence_length=layout.sequence_length,
        input_order=layout.input_order,
    )

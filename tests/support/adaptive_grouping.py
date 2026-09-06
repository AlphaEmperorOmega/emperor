"""Real small adaptive models used by grouping behavioral tests."""

import torch

from emperor.augmentations.adaptive_parameters import (
    AdaptiveLinearLayerConfig,
    AdaptiveParameterAugmentationConfig,
    AdaptiveParameterGroupingScopeOptions,
    AdditiveDynamicBiasConfig,
    DualModelDynamicWeightConfig,
    DynamicDepthOptions,
    MaskDimensionOptions,
    PerAxisScoreMaskConfig,
    StandardDynamicDiagonalConfig,
    WeightDecayScheduleOptions,
    WeightNormalizationOptions,
    WeightNormalizationPositionOptions,
)
from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerConfig,
    LayerNormPositionOptions,
    LayerStackConfig,
)
from emperor.linears import LinearLayerConfig


def linear_stack_config(input_dim: int, output_dim: int) -> LayerStackConfig:
    return LayerStackConfig(
        input_dim=input_dim,
        hidden_dim=max(input_dim, output_dim),
        output_dim=output_dim,
        num_layers=1,
        last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
        apply_output_postprocessing_flag=False,
        layer_config=LayerConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            activation=ActivationOptions.DISABLED,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            residual_config=None,
            dropout_probability=0.0,
            gate_config=None,
            halting_config=None,
            memory_config=None,
            layer_model_config=LinearLayerConfig(
                input_dim=input_dim,
                output_dim=output_dim,
                bias_flag=True,
            ),
        ),
    )


def bias_linear(grouping_config=None):
    model = AdaptiveLinearLayerConfig(
        input_dim=2,
        output_dim=2,
        bias_flag=True,
        adaptive_augmentation_config=AdaptiveParameterAugmentationConfig(
            grouping_config=grouping_config,
            bias_config=AdditiveDynamicBiasConfig(
                decay_schedule=WeightDecayScheduleOptions.DISABLED,
                decay_rate=0.0,
                decay_warmup_batches=0,
                model_config=linear_stack_config(2, 2),
            ),
        ),
    ).build()
    with torch.no_grad():
        model.weight_params.zero_()
        model.bias_params.zero_()
        generator = model.adaptive_behaviour.bias_model.model[0].model
        generator.weight_params.copy_(torch.eye(2))
        generator.bias_params.zero_()
    return model


def grouping_value(scope, group_count, *, sequence_length=4, input_order="BATCH_FIRST"):
    from emperor.augmentations.adaptive_parameters import (
        AdaptiveParameterGroupingScopeOptions,
        AdaptiveParameterInputOrderOptions,
        SumGroupingConfig,
    )

    if scope is None:
        return None
    if scope is AdaptiveParameterGroupingScopeOptions.SEQUENCE:
        return SumGroupingConfig(
            scope=scope,
            group_count=group_count,
            sequence_length=sequence_length,
            input_order=AdaptiveParameterInputOrderOptions[input_order],
        )
    return SumGroupingConfig(scope=scope, group_count=group_count)


def combined_linear():
    AdaptiveLinearLayer = AdaptiveLinearLayerConfig().registry_owner()
    generator_config = linear_stack_config(2, 3)
    model = AdaptiveLinearLayer(
        AdaptiveLinearLayerConfig(
            input_dim=2,
            output_dim=3,
            bias_flag=True,
            adaptive_augmentation_config=AdaptiveParameterAugmentationConfig(
                weight_config=DualModelDynamicWeightConfig(
                    input_dim=2,
                    output_dim=3,
                    generator_depth=DynamicDepthOptions.DEPTH_OF_ONE,
                    decay_schedule=WeightDecayScheduleOptions.DISABLED,
                    decay_rate=0.0,
                    decay_warmup_batches=0,
                    normalization_option=WeightNormalizationOptions.DISABLED,
                    normalization_position_option=(
                        WeightNormalizationPositionOptions.DISABLED
                    ),
                    model_config=generator_config,
                ),
                diagonal_config=StandardDynamicDiagonalConfig(
                    input_dim=2,
                    output_dim=3,
                    model_config=generator_config,
                ),
                bias_config=AdditiveDynamicBiasConfig(
                    input_dim=2,
                    output_dim=3,
                    decay_schedule=WeightDecayScheduleOptions.DISABLED,
                    decay_rate=0.0,
                    decay_warmup_batches=0,
                    model_config=generator_config,
                ),
                mask_config=PerAxisScoreMaskConfig(
                    input_dim=2,
                    output_dim=3,
                    mask_dimension_option=MaskDimensionOptions.COLUMN,
                    mask_threshold=0.5,
                    mask_surrogate_scale=1.0,
                    mask_floor=0.0,
                    model_config=generator_config,
                ),
                grouping_config=grouping_value(
                    AdaptiveParameterGroupingScopeOptions.ROWS,
                    2,
                    input_order="BATCH_FIRST",
                ),
            ),
        )
    )
    return model

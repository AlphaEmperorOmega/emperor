"""Small real configurations for parameter generation acceptance tests."""

from emperor.augmentations.adaptive_parameters import (
    DiagonallyModulatedLowRankDynamicWeightConfig,
    DynamicDepthOptions,
    WeightDecayScheduleOptions,
    WeightNormalizationOptions,
)
from emperor.sampler import SamplerConfig
from support.adaptive_grouping import linear_stack_config


def weight_mixture_config(**overrides):
    from emperor.augmentations.adaptive_parameters import MatrixWeightsMixtureConfig

    return MatrixWeightsMixtureConfig(
        **(
            dict(
                input_dim=2,
                output_dim=3,
                num_experts=3,
                top_k=2,
                sampler_config=SamplerConfig(),
                model_config=linear_stack_config(2, 3),
            )
            | overrides
        )
    )


def bias_mixture_config(**overrides):
    from emperor.augmentations.adaptive_parameters import MatrixBiasMixtureConfig

    return MatrixBiasMixtureConfig(
        **(
            dict(
                input_dim=2,
                output_dim=3,
                num_experts=3,
                top_k=2,
                sampler_config=SamplerConfig(),
                model_config=linear_stack_config(2, 3),
            )
            | overrides
        )
    )


def modulated_config(**overrides):
    values = dict(
        input_dim=3,
        output_dim=2,
        generator_depth=DynamicDepthOptions.DEPTH_OF_TWO,
        normalization_option=WeightNormalizationOptions.DISABLED,
        decay_schedule=WeightDecayScheduleOptions.DISABLED,
        decay_rate=0.0,
        decay_warmup_batches=0,
        model_config=linear_stack_config(3, 2),
    )
    return DiagonallyModulatedLowRankDynamicWeightConfig(**(values | overrides))

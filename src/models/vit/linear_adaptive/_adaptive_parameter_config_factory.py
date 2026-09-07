from emperor.augmentations.adaptive_parameters import (
    AxisMaskConfig,
    BankExpansionFactorOptions,
    DiagonalAxisMaskConfig,
    DiagonallyModulatedLowRankDynamicWeightConfig,
    DualModelDynamicWeightConfig,
    DynamicBiasConfig,
    DynamicDepthOptions,
    DynamicDiagonalConfig,
    DynamicWeightConfig,
    HypernetworkDynamicWeightConfig,
    LayeredWeightedBankDynamicWeightConfig,
    LowRankDynamicWeightConfig,
    MaskDimensionOptions,
    MatrixBiasMixtureConfig,
    MatrixWeightsMixtureConfig,
    PerAxisScoreMaskConfig,
    SingleModelDynamicWeightConfig,
    SoftWeightedBankDynamicWeightConfig,
    TopSliceAxisMaskConfig,
    WeightDecayScheduleOptions,
    WeightedBankDynamicBiasConfig,
    WeightInformedScoreAxisMaskConfig,
    WeightNormalizationOptions,
    WeightNormalizationPositionOptions,
)
from emperor.layers import LayerStackConfig

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
    DiagonallyModulatedLowRankDynamicWeightConfig: (
        "normalization_option",
        "input_factor_source",
        "output_factor_source",
        "input_factor_model_config",
        "output_factor_model_config",
        "coefficient_model_config",
    ),
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


def build_weight_config(
    weight_option: type[DynamicWeightConfig] | None,
    *,
    generation_fields: dict | None = None,
    generator_depth: DynamicDepthOptions,
    decay_schedule: WeightDecayScheduleOptions,
    decay_rate: float,
    decay_warmup_batches: int,
    normalization_option: WeightNormalizationOptions,
    normalization_position_option: WeightNormalizationPositionOptions,
    bank_expansion_factor: BankExpansionFactorOptions,
    model_config: LayerStackConfig | None = None,
) -> DynamicWeightConfig | None:
    if weight_option is None:
        return None
    if weight_option is MatrixWeightsMixtureConfig:
        mixture_fields = {
            name: value
            for name, value in (generation_fields or {}).items()
            if name in ("num_experts", "top_k", "sampler_config")
        }
        return MatrixWeightsMixtureConfig(
            decay_schedule=decay_schedule,
            decay_rate=decay_rate,
            decay_warmup_batches=decay_warmup_batches,
            model_config=model_config,
            **mixture_fields,
        )
    kwargs = {
        "generator_depth": generator_depth,
        "decay_schedule": decay_schedule,
        "decay_rate": decay_rate,
        "decay_warmup_batches": decay_warmup_batches,
        "model_config": model_config,
    }
    optional_kwargs = {
        "normalization_option": normalization_option,
        "normalization_position_option": normalization_position_option,
        "bank_expansion_factor": bank_expansion_factor,
    }
    optional_kwargs.update(generation_fields or {})
    kwargs.update(
        _selected_kwargs(_WEIGHT_OPTION_FIELDS, weight_option, optional_kwargs)
    )
    return weight_option(**kwargs)


def build_bias_config(
    bias_option: type[DynamicBiasConfig] | None,
    *,
    generation_fields: dict | None = None,
    decay_schedule: WeightDecayScheduleOptions,
    decay_rate: float,
    decay_warmup_batches: int,
    bank_expansion_factor: BankExpansionFactorOptions,
    model_config: LayerStackConfig | None = None,
) -> DynamicBiasConfig | None:
    if bias_option is None:
        return None
    if bias_option is MatrixBiasMixtureConfig:
        mixture_fields = {
            name: value
            for name, value in (generation_fields or {}).items()
            if name in ("num_experts", "top_k", "sampler_config")
        }
        return MatrixBiasMixtureConfig(
            decay_schedule=decay_schedule,
            decay_rate=decay_rate,
            decay_warmup_batches=decay_warmup_batches,
            model_config=model_config,
            **mixture_fields,
        )
    kwargs = {
        "decay_schedule": decay_schedule,
        "decay_rate": decay_rate,
        "decay_warmup_batches": decay_warmup_batches,
        "model_config": model_config,
    }
    optional_kwargs = {
        "bank_expansion_factor": bank_expansion_factor,
    }
    kwargs.update(_selected_kwargs(_BIAS_OPTION_FIELDS, bias_option, optional_kwargs))
    return bias_option(**kwargs)


def build_diagonal_config(
    diagonal_option: type[DynamicDiagonalConfig] | None,
    *,
    model_config: LayerStackConfig | None = None,
) -> DynamicDiagonalConfig | None:
    if diagonal_option is None:
        return None
    return diagonal_option(model_config=model_config)


def build_mask_config(
    row_mask_option: type[AxisMaskConfig] | None,
    *,
    mask_dimension_option: MaskDimensionOptions,
    mask_threshold: float,
    mask_surrogate_scale: float,
    mask_floor: float,
    mask_transition_width: float,
    model_config: LayerStackConfig | None = None,
) -> AxisMaskConfig | None:
    if row_mask_option is None:
        return None
    kwargs = {
        "mask_threshold": mask_threshold,
        "mask_surrogate_scale": mask_surrogate_scale,
        "mask_floor": mask_floor,
        "model_config": model_config,
    }
    optional_kwargs = {
        "mask_dimension_option": mask_dimension_option,
        "mask_transition_width": mask_transition_width,
    }
    kwargs.update(
        _selected_kwargs(_MASK_OPTION_FIELDS, row_mask_option, optional_kwargs)
    )
    return row_mask_option(**kwargs)


def _selected_kwargs(
    field_table: dict[type, tuple[str, ...]],
    option: type,
    optional_kwargs: dict[str, object],
) -> dict[str, object]:
    return {name: optional_kwargs.get(name) for name in field_table.get(option, ())}

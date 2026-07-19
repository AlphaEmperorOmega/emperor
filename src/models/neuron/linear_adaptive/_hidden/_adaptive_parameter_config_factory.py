from __future__ import annotations

from emperor.augmentations.adaptive_parameters import (
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
from emperor.layers import LayerConfig, LayerStackConfig, ResidualConfig
from emperor.linears import LinearLayerConfig
from models.neuron.linear_adaptive._hidden.runtime_options import (
    AdaptiveProjectionOptions,
    GeneratorStackOptions,
    RuntimeOptions,
    StackOptions,
)

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


def _selected_kwargs(
    table: dict[type, tuple[str, ...]],
    option: type,
    available: dict[str, object],
) -> dict[str, object]:
    return {name: available[name] for name in table.get(option, ())}


def _stack_config(options: StackOptions) -> LayerStackConfig:
    return LayerStackConfig(
        hidden_dim=options.hidden_dim,
        num_layers=options.num_layers,
        last_layer_bias_option=options.last_layer_bias_option,
        apply_output_pipeline_flag=options.apply_output_pipeline_flag,
        layer_config=LayerConfig(
            activation=options.activation,
            layer_norm_position=options.layer_norm_position,
            residual_config=None
            if options.residual_connection_option is None
            else ResidualConfig(option=options.residual_connection_option),
            dropout_probability=options.dropout_probability,
            gate_config=None,
            halting_config=None,
            layer_model_config=LinearLayerConfig(bias_flag=options.bias_flag),
        ),
    )


def _independent_stack_config(
    options: GeneratorStackOptions,
) -> LayerStackConfig | None:
    return _stack_config(options.stack) if options.independent else None


class AdaptiveParameterConfigFactory:
    def __init__(self, runtime: RuntimeOptions) -> None:
        self._runtime = runtime

    def build_hidden_config(self) -> AdaptiveParameterAugmentationConfig:
        runtime = self._runtime
        weight_config = None
        if runtime.weight.enabled:
            weight_config = self._weight_config(
                runtime.weight.option,
                generator_depth=runtime.weight.generator_depth,
                decay_schedule=runtime.weight.decay_schedule,
                decay_rate=runtime.weight.decay_rate,
                decay_warmup_batches=runtime.weight.decay_warmup_batches,
                normalization_option=runtime.weight.normalization_option,
                normalization_position_option=(
                    runtime.weight.normalization_position_option
                ),
                bank_expansion_factor=runtime.weight.bank_expansion_factor,
                model_config=_independent_stack_config(runtime.weight.generator_stack),
            )
        bias_config = None
        if runtime.bias.enabled:
            bias_config = self._bias_config(
                runtime.bias.option,
                decay_schedule=runtime.bias.decay_schedule,
                decay_rate=runtime.bias.decay_rate,
                decay_warmup_batches=runtime.bias.decay_warmup_batches,
                bank_expansion_factor=runtime.bias.bank_expansion_factor,
                model_config=_independent_stack_config(runtime.bias.generator_stack),
            )
        diagonal_config = None
        if runtime.diagonal.enabled:
            diagonal_config = self._diagonal_config(
                runtime.diagonal.option,
                model_config=_independent_stack_config(
                    runtime.diagonal.generator_stack
                ),
            )
        mask_config = None
        if runtime.mask.enabled:
            mask_config = self._mask_config(
                runtime.mask.row_mask_option,
                mask_dimension_option=runtime.mask.dimension_option,
                mask_threshold=runtime.mask.threshold,
                mask_surrogate_scale=runtime.mask.surrogate_scale,
                mask_floor=runtime.mask.floor,
                mask_transition_width=runtime.mask.transition_width,
                model_config=_independent_stack_config(runtime.mask.generator_stack),
            )
        return AdaptiveParameterAugmentationConfig(
            weight_config=weight_config,
            bias_config=bias_config,
            diagonal_config=diagonal_config,
            mask_config=mask_config,
            model_config=_stack_config(runtime.adaptive_generator_stack),
        )

    def build_projection_config(
        self,
        options: AdaptiveProjectionOptions,
    ) -> AdaptiveParameterAugmentationConfig:
        return AdaptiveParameterAugmentationConfig(
            weight_config=self._weight_config(
                options.weight_option,
                generator_depth=options.weight_generator_depth,
                decay_schedule=options.weight_decay_schedule,
                decay_rate=options.weight_decay_rate,
                decay_warmup_batches=options.weight_decay_warmup_batches,
                normalization_option=options.weight_normalization_option,
                normalization_position_option=(
                    options.weight_normalization_position_option
                ),
                bank_expansion_factor=options.weight_bank_expansion_factor,
            ),
            bias_config=self._bias_config(
                options.bias_option,
                decay_schedule=options.bias_decay_schedule,
                decay_rate=options.bias_decay_rate,
                decay_warmup_batches=options.bias_decay_warmup_batches,
                bank_expansion_factor=options.bias_bank_expansion_factor,
            ),
            diagonal_config=self._diagonal_config(options.diagonal_option),
            mask_config=self._mask_config(
                options.row_mask_option,
                mask_dimension_option=options.mask_dimension_option,
                mask_threshold=options.mask_threshold,
                mask_surrogate_scale=options.mask_surrogate_scale,
                mask_floor=options.mask_floor,
                mask_transition_width=options.mask_transition_width,
            ),
            model_config=_stack_config(self._runtime.adaptive_generator_stack),
        )

    @staticmethod
    def _weight_config(
        option,
        *,
        generator_depth,
        decay_schedule,
        decay_rate,
        decay_warmup_batches,
        normalization_option,
        normalization_position_option,
        bank_expansion_factor,
        model_config=None,
    ) -> DynamicWeightConfig | None:
        if option is None:
            return None
        kwargs = {
            "generator_depth": generator_depth,
            "decay_schedule": decay_schedule,
            "decay_rate": decay_rate,
            "decay_warmup_batches": decay_warmup_batches,
            "model_config": model_config,
        }
        kwargs.update(
            _selected_kwargs(
                _WEIGHT_OPTION_FIELDS,
                option,
                {
                    "normalization_option": normalization_option,
                    "normalization_position_option": normalization_position_option,
                    "bank_expansion_factor": bank_expansion_factor,
                },
            )
        )
        return option(**kwargs)

    @staticmethod
    def _bias_config(
        option,
        *,
        decay_schedule,
        decay_rate,
        decay_warmup_batches,
        bank_expansion_factor,
        model_config=None,
    ) -> DynamicBiasConfig | None:
        if option is None:
            return None
        kwargs = {
            "decay_schedule": decay_schedule,
            "decay_rate": decay_rate,
            "decay_warmup_batches": decay_warmup_batches,
            "model_config": model_config,
        }
        kwargs.update(
            _selected_kwargs(
                _BIAS_OPTION_FIELDS,
                option,
                {"bank_expansion_factor": bank_expansion_factor},
            )
        )
        return option(**kwargs)

    @staticmethod
    def _diagonal_config(
        option,
        *,
        model_config=None,
    ) -> DynamicDiagonalConfig | None:
        return None if option is None else option(model_config=model_config)

    @staticmethod
    def _mask_config(
        option,
        *,
        mask_dimension_option,
        mask_threshold,
        mask_surrogate_scale,
        mask_floor,
        mask_transition_width,
        model_config=None,
    ) -> AxisMaskConfig | None:
        if option is None:
            return None
        kwargs = {
            "mask_threshold": mask_threshold,
            "mask_surrogate_scale": mask_surrogate_scale,
            "mask_floor": mask_floor,
            "model_config": model_config,
        }
        kwargs.update(
            _selected_kwargs(
                _MASK_OPTION_FIELDS,
                option,
                {
                    "mask_dimension_option": mask_dimension_option,
                    "mask_transition_width": mask_transition_width,
                },
            )
        )
        return option(**kwargs)

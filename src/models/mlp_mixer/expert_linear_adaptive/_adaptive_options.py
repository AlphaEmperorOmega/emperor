from __future__ import annotations

from dataclasses import dataclass

from emperor.augmentations.adaptive_parameters import (
    AxisMaskConfig,
    BankExpansionFactorOptions,
    DynamicBiasConfig,
    DynamicDepthOptions,
    DynamicDiagonalConfig,
    DynamicWeightConfig,
    MaskDimensionOptions,
    WeightDecayScheduleOptions,
    WeightNormalizationOptions,
    WeightNormalizationPositionOptions,
)
from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
    ResidualConfig,
)

from ._control_options import StackOptions
from .runtime_options import RuntimeOptions


@dataclass(frozen=True, slots=True)
class GeneratorStackSource:
    independent_flag: bool
    hidden_dim: int | None
    num_layers: int | None
    activation: ActivationOptions | None
    dropout_probability: float | None
    layer_norm_position: LayerNormPositionOptions | None
    residual_connection_option: type[ResidualConfig] | None
    residual_model_flag: bool
    last_layer_bias_option: LastLayerBiasOptions | None
    apply_output_postprocessing_flag: bool | None
    bias_flag: bool | None

    def resolve(self, defaults: StackOptions) -> StackOptions | None:
        if not self.independent_flag:
            return None
        return StackOptions(
            hidden_dim=(
                defaults.hidden_dim if self.hidden_dim is None else self.hidden_dim
            ),
            num_layers=(
                defaults.num_layers if self.num_layers is None else self.num_layers
            ),
            activation=(
                defaults.activation if self.activation is None else self.activation
            ),
            dropout_probability=(
                defaults.dropout_probability
                if self.dropout_probability is None
                else self.dropout_probability
            ),
            layer_norm_position=(
                defaults.layer_norm_position
                if self.layer_norm_position is None
                else self.layer_norm_position
            ),
            residual_connection_option=(
                defaults.residual_connection_option
                if self.residual_connection_option is None
                else self.residual_connection_option
            ),
            residual_model_flag=self.residual_model_flag,
            last_layer_bias_option=(
                defaults.last_layer_bias_option
                if self.last_layer_bias_option is None
                else self.last_layer_bias_option
            ),
            apply_output_postprocessing_flag=(
                defaults.apply_output_postprocessing_flag
                if self.apply_output_postprocessing_flag is None
                else self.apply_output_postprocessing_flag
            ),
            bias_flag=(
                defaults.bias_flag if self.bias_flag is None else self.bias_flag
            ),
        )


@dataclass(frozen=True, slots=True)
class WeightOptions:
    enabled: bool
    implementation: type[DynamicWeightConfig] | None
    generator_depth: DynamicDepthOptions
    decay_schedule: WeightDecayScheduleOptions
    decay_rate: float
    decay_warmup_batches: int
    normalization_option: WeightNormalizationOptions
    normalization_position_option: WeightNormalizationPositionOptions
    bank_expansion_factor: BankExpansionFactorOptions
    generator_stack: GeneratorStackSource


@dataclass(frozen=True, slots=True)
class BiasOptions:
    enabled: bool
    implementation: type[DynamicBiasConfig] | None
    decay_schedule: WeightDecayScheduleOptions
    decay_rate: float
    decay_warmup_batches: int
    bank_expansion_factor: BankExpansionFactorOptions
    generator_stack: GeneratorStackSource


@dataclass(frozen=True, slots=True)
class DiagonalOptions:
    enabled: bool
    implementation: type[DynamicDiagonalConfig] | None
    generator_stack: GeneratorStackSource


@dataclass(frozen=True, slots=True)
class MaskOptions:
    enabled: bool
    implementation: type[AxisMaskConfig] | None
    threshold: float
    surrogate_scale: float
    floor: float
    dimension_option: MaskDimensionOptions
    transition_width: float
    generator_stack: GeneratorStackSource


@dataclass(frozen=True, slots=True)
class AdaptiveOptions:
    generator_stack: StackOptions
    weight: WeightOptions
    bias: BiasOptions
    diagonal: DiagonalOptions
    mask: MaskOptions


def adaptive_options(runtime: RuntimeOptions) -> AdaptiveOptions:
    return AdaptiveOptions(
        generator_stack=StackOptions(
            hidden_dim=runtime.adaptive_generator_stack_hidden_dim,
            num_layers=runtime.adaptive_generator_stack_num_layers,
            activation=runtime.adaptive_generator_stack_activation,
            dropout_probability=runtime.adaptive_generator_stack_dropout_probability,
            layer_norm_position=runtime.adaptive_generator_stack_layer_norm_position,
            residual_connection_option=(
                runtime.adaptive_generator_stack_residual_connection_option
            ),
            residual_model_flag=(runtime.adaptive_generator_stack_residual_model_flag),
            last_layer_bias_option=(
                runtime.adaptive_generator_stack_last_layer_bias_option
            ),
            apply_output_postprocessing_flag=(
                runtime.adaptive_generator_stack_apply_output_postprocessing_flag
            ),
            bias_flag=runtime.adaptive_generator_stack_bias_flag,
        ),
        weight=WeightOptions(
            enabled=runtime.weight_option_flag,
            implementation=runtime.weight_option,
            generator_depth=runtime.generator_depth,
            decay_schedule=runtime.weight_decay_schedule,
            decay_rate=runtime.weight_decay_rate,
            decay_warmup_batches=runtime.weight_decay_warmup_batches,
            normalization_option=runtime.weight_normalization_option,
            normalization_position_option=(
                runtime.weight_normalization_position_option
            ),
            bank_expansion_factor=runtime.weight_bank_expansion_factor,
            generator_stack=_weight_generator_stack(runtime),
        ),
        bias=BiasOptions(
            enabled=runtime.bias_option_flag,
            implementation=runtime.bias_option,
            decay_schedule=runtime.bias_decay_schedule,
            decay_rate=runtime.bias_decay_rate,
            decay_warmup_batches=runtime.bias_decay_warmup_batches,
            bank_expansion_factor=runtime.bias_bank_expansion_factor,
            generator_stack=_bias_generator_stack(runtime),
        ),
        diagonal=DiagonalOptions(
            enabled=runtime.diagonal_option_flag,
            implementation=runtime.diagonal_option,
            generator_stack=_diagonal_generator_stack(runtime),
        ),
        mask=MaskOptions(
            enabled=runtime.mask_option_flag,
            implementation=runtime.row_mask_option,
            threshold=runtime.mask_threshold,
            surrogate_scale=runtime.mask_surrogate_scale,
            floor=runtime.mask_floor,
            dimension_option=runtime.mask_dimension_option,
            transition_width=runtime.mask_transition_width,
            generator_stack=_mask_generator_stack(runtime),
        ),
    )


def _weight_generator_stack(runtime: RuntimeOptions) -> GeneratorStackSource:
    return GeneratorStackSource(
        independent_flag=runtime.weight_generator_stack_independent_flag,
        hidden_dim=runtime.weight_generator_stack_hidden_dim,
        num_layers=runtime.weight_generator_stack_num_layers,
        activation=runtime.weight_generator_stack_activation,
        dropout_probability=runtime.weight_generator_stack_dropout_probability,
        layer_norm_position=runtime.weight_generator_stack_layer_norm_position,
        residual_connection_option=(
            runtime.weight_generator_stack_residual_connection_option
        ),
        residual_model_flag=runtime.weight_generator_stack_residual_model_flag,
        last_layer_bias_option=runtime.weight_generator_stack_last_layer_bias_option,
        apply_output_postprocessing_flag=(
            runtime.weight_generator_stack_apply_output_postprocessing_flag
        ),
        bias_flag=runtime.weight_generator_stack_bias_flag,
    )


def _bias_generator_stack(runtime: RuntimeOptions) -> GeneratorStackSource:
    return GeneratorStackSource(
        independent_flag=runtime.bias_generator_stack_independent_flag,
        hidden_dim=runtime.bias_generator_stack_hidden_dim,
        num_layers=runtime.bias_generator_stack_num_layers,
        activation=runtime.bias_generator_stack_activation,
        dropout_probability=runtime.bias_generator_stack_dropout_probability,
        layer_norm_position=runtime.bias_generator_stack_layer_norm_position,
        residual_connection_option=(
            runtime.bias_generator_stack_residual_connection_option
        ),
        residual_model_flag=runtime.bias_generator_stack_residual_model_flag,
        last_layer_bias_option=runtime.bias_generator_stack_last_layer_bias_option,
        apply_output_postprocessing_flag=(
            runtime.bias_generator_stack_apply_output_postprocessing_flag
        ),
        bias_flag=runtime.bias_generator_stack_bias_flag,
    )


def _diagonal_generator_stack(runtime: RuntimeOptions) -> GeneratorStackSource:
    return GeneratorStackSource(
        independent_flag=runtime.diagonal_generator_stack_independent_flag,
        hidden_dim=runtime.diagonal_generator_stack_hidden_dim,
        num_layers=runtime.diagonal_generator_stack_num_layers,
        activation=runtime.diagonal_generator_stack_activation,
        dropout_probability=runtime.diagonal_generator_stack_dropout_probability,
        layer_norm_position=runtime.diagonal_generator_stack_layer_norm_position,
        residual_connection_option=(
            runtime.diagonal_generator_stack_residual_connection_option
        ),
        residual_model_flag=runtime.diagonal_generator_stack_residual_model_flag,
        last_layer_bias_option=(
            runtime.diagonal_generator_stack_last_layer_bias_option
        ),
        apply_output_postprocessing_flag=(
            runtime.diagonal_generator_stack_apply_output_postprocessing_flag
        ),
        bias_flag=runtime.diagonal_generator_stack_bias_flag,
    )


def _mask_generator_stack(runtime: RuntimeOptions) -> GeneratorStackSource:
    return GeneratorStackSource(
        independent_flag=runtime.mask_generator_stack_independent_flag,
        hidden_dim=runtime.mask_generator_stack_hidden_dim,
        num_layers=runtime.mask_generator_stack_num_layers,
        activation=runtime.mask_generator_stack_activation,
        dropout_probability=runtime.mask_generator_stack_dropout_probability,
        layer_norm_position=runtime.mask_generator_stack_layer_norm_position,
        residual_connection_option=(
            runtime.mask_generator_stack_residual_connection_option
        ),
        residual_model_flag=runtime.mask_generator_stack_residual_model_flag,
        last_layer_bias_option=runtime.mask_generator_stack_last_layer_bias_option,
        apply_output_postprocessing_flag=(
            runtime.mask_generator_stack_apply_output_postprocessing_flag
        ),
        bias_flag=runtime.mask_generator_stack_bias_flag,
    )

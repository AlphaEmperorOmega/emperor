from emperor.augmentations.adaptive_parameters.core.bias import (
    AdditiveDynamicBiasConfig,
    AffineTransformDynamicBiasConfig,
    GeneratorDynamicBiasConfig,
    MultiplicativeDynamicBiasConfig,
    SigmoidGatedDynamicBiasConfig,
    TanhGatedDynamicBiasConfig,
    WeightedBankDynamicBiasConfig,
)
from emperor.augmentations.adaptive_parameters.core.diagonal import (
    AntiDynamicDiagonalConfig,
    CombinedDynamicDiagonalConfig,
    StandardDynamicDiagonalConfig,
)
from emperor.augmentations.adaptive_parameters.core.mask import (
    DiagonalAxisMaskConfig,
    OuterProductMaskConfig,
    PerAxisScoreMaskConfig,
    TopSliceAxisMaskConfig,
    WeightInformedScoreAxisMaskConfig,
)
from emperor.augmentations.adaptive_parameters.core.weight import (
    DualModelDynamicWeightConfig,
    HypernetworkDynamicWeightConfig,
    LayeredWeightedBankDynamicWeightConfig,
    LowRankDynamicWeightConfig,
    SingleModelDynamicWeightConfig,
    SoftWeightedBankDynamicWeightConfig,
)
from emperor.augmentations.adaptive_parameters.options import (
    DynamicDepthOptions,
    WeightDecayScheduleOptions,
    WeightNormalizationOptions,
)
from emperor.base.layer.residual import ResidualConnectionOptions
from emperor.base.options import BaseOptions, LayerNormPositionOptions
from model_runtime.packages import (
    BuilderBackedExperimentPresetsBase,
    ExperimentPresetsBase,
    PresetDefinition,
)
from model_runtime.runs import ExperimentBase

import models.linears.linear_adaptive.config as config
import models.linears.linear_adaptive.dataset_options as dataset_options
from models.linears.linear_adaptive.config_builder import LinearAdaptiveConfigBuilder
from models.linears.linear_adaptive.model import Model
from models.linears.linear_adaptive.runtime_defaults import runtime_from_flat


class ExperimentPreset(BaseOptions):
    BASELINE = 1
    GATING = 2
    HALTING = 3
    MEMORY = 4
    GATING_HALTING = 5
    GATING_MEMORY = 6
    HALTING_MEMORY = 7
    GATING_HALTING_MEMORY = 8
    RESIDUAL = 9
    POST_NORM = 10
    RESIDUAL_POST_NORM = 11
    RESIDUAL_GATING = 12
    RESIDUAL_HALTING = 13
    RESIDUAL_MEMORY = 14
    SINGLE_MODEL_WEIGHT = 15
    DUAL_MODEL_WEIGHT = 16
    LOW_RANK_WEIGHT = 17
    HYPERNETWORK_WEIGHT = 18
    LAYERED_WEIGHTED_BANK_WEIGHT = 19
    SOFT_WEIGHTED_BANK_WEIGHT = 20
    AFFINE_TRANSFORM_BIAS = 21
    ADDITIVE_BIAS = 22
    GENERATOR_BIAS = 23
    MULTIPLICATIVE_BIAS = 24
    SIGMOID_GATED_BIAS = 25
    TANH_GATED_BIAS = 26
    WEIGHTED_BANK_BIAS = 27
    STANDARD_DIAGONAL = 28
    ANTI_DIAGONAL = 29
    COMBINED_DIAGONAL = 30
    DIAGONAL_AXIS_MASK = 31
    OUTER_PRODUCT_MASK = 32
    PER_AXIS_SCORE_MASK = 33
    TOP_SLICE_AXIS_MASK = 34
    WEIGHT_INFORMED_SCORE_MASK = 35
    SINGLE_MODEL_WEIGHT_ADDITIVE_BIAS_COMBINED_DIAGONAL = 36
    DUAL_MODEL_WEIGHT_ADDITIVE_BIAS_COMBINED_DIAGONAL = 37
    LAYERED_WEIGHTED_BANK_WEIGHT_ADDITIVE_BIAS_COMBINED_DIAGONAL = 38
    LOW_RANK_WEIGHT_ADDITIVE_BIAS_COMBINED_DIAGONAL = 39
    SINGLE_MODEL_WEIGHT_ADDITIVE_BIAS_STANDARD_DIAGONAL = 40
    DUAL_MODEL_WEIGHT_ADDITIVE_BIAS_STANDARD_DIAGONAL = 41
    LAYERED_WEIGHTED_BANK_WEIGHT_ADDITIVE_BIAS_STANDARD_DIAGONAL = 42
    LOW_RANK_WEIGHT_ADDITIVE_BIAS_STANDARD_DIAGONAL = 43
    DECAY_EXPONENTIAL_WEIGHT = 44
    NORM_L2_WEIGHT = 45
    DEEP_GENERATOR = 46
    FULL_STACK = 47
    DUAL_WEIGHT_GATING = 48
    DUAL_WEIGHT_HALTING = 49
    DUAL_WEIGHT_GATING_HALTING = 50
    DUAL_WEIGHT_MEMORY = 51
    DUAL_WEIGHT_GATING_MEMORY = 52
    DUAL_WEIGHT_HALTING_MEMORY = 53
    FULL_STACK_GATING = 54
    FULL_STACK_HALTING = 55
    FULL_STACK_MEMORY = 56
    FULL_STACK_GATING_HALTING = 57
    FULL_STACK_RECURRENT = 58
    BANK_WEIGHT_MASK = 59
    LOW_RANK_POST_NORM = 60
    RECURRENT = 61
    RECURRENT_GATING = 62
    RECURRENT_HALTING = 63
    RECURRENT_MEMORY = 64
    RECURRENT_GATING_HALTING = 65
    RECURRENT_GATING_MEMORY = 66
    RECURRENT_HALTING_MEMORY = 67
    RECURRENT_GATING_HALTING_MEMORY = 68
    RECURRENT_RESIDUAL = 69
    RECURRENT_POST_NORM = 70


_ADAPTIVE_OPTION_FLAGS = {
    "weight_option": "weight_option_flag",
    "bias_option": "bias_option_flag",
    "diagonal_option": "diagonal_option_flag",
    "row_mask_option": "mask_option_flag",
}


def _with_adaptive_option_flags(overrides: dict[str, object]) -> dict[str, object]:
    option_flags = {
        flag: True
        for option, flag in _ADAPTIVE_OPTION_FLAGS.items()
        if option in overrides and overrides[option] is not None
    }
    return {**option_flags, **overrides}


_FULL_STACK_OVERRIDES = _with_adaptive_option_flags(
    {
        "weight_option": DualModelDynamicWeightConfig,
        "bias_option": AdditiveDynamicBiasConfig,
        "diagonal_option": CombinedDynamicDiagonalConfig,
        "row_mask_option": WeightInformedScoreAxisMaskConfig,
    }
)


_PRESET_DEFINITIONS = {
    ExperimentPreset.BASELINE: PresetDefinition(
        preset_values=_with_adaptive_option_flags({}),
        description="Default config: a GELU adaptive linear stack with pre-layer norm and "
        "dropout.",
    ),
    ExperimentPreset.GATING: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "stack_gate_flag": True,
            }
        ),
        description="Default adaptive config with per-layer gating enabled, so each hidden "
        "layer output is modulated by a learned sigmoid gate.",
    ),
    ExperimentPreset.HALTING: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "stack_halting_flag": True,
            }
        ),
        description="Default adaptive config with stack halting enabled, so examples can "
        "stop early as they move through the adaptive hidden stack.",
    ),
    ExperimentPreset.MEMORY: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "memory_flag": True,
            }
        ),
        description="Default adaptive config with shared stack memory enabled across the "
        "hidden layers.",
    ),
    ExperimentPreset.GATING_HALTING: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "stack_gate_flag": True,
                "stack_halting_flag": True,
            }
        ),
        description="Default adaptive config with both per-layer gating and stack halting "
        "enabled.",
    ),
    ExperimentPreset.GATING_MEMORY: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "stack_gate_flag": True,
                "memory_flag": True,
            }
        ),
        description="Default adaptive config with both per-layer gating and shared stack "
        "memory enabled.",
    ),
    ExperimentPreset.HALTING_MEMORY: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "stack_halting_flag": True,
                "memory_flag": True,
            }
        ),
        description="Default adaptive config with both stack halting and shared stack "
        "memory enabled.",
    ),
    ExperimentPreset.GATING_HALTING_MEMORY: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "stack_gate_flag": True,
                "stack_halting_flag": True,
                "memory_flag": True,
            }
        ),
        description="Default adaptive config with per-layer gating, stack halting, and "
        "shared stack memory enabled.",
    ),
    ExperimentPreset.RESIDUAL: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "stack_residual_connection_option": (
                    ResidualConnectionOptions.RESIDUAL
                ),
            }
        ),
        description="Default adaptive config with residual skip connections "
        "enabled between same-width hidden layers.",
    ),
    ExperimentPreset.POST_NORM: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "layer_norm_position": LayerNormPositionOptions.AFTER,
            }
        ),
        description="Default adaptive config with layer norm applied after each layer "
        "instead of before it.",
    ),
    ExperimentPreset.RESIDUAL_POST_NORM: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "stack_residual_connection_option": (
                    ResidualConnectionOptions.RESIDUAL
                ),
                "layer_norm_position": LayerNormPositionOptions.AFTER,
            }
        ),
        description="Default adaptive config with residual skip connections and post-layer "
        "normalization enabled.",
    ),
    ExperimentPreset.RESIDUAL_GATING: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "stack_residual_connection_option": (
                    ResidualConnectionOptions.RESIDUAL
                ),
                "stack_gate_flag": True,
            }
        ),
        description="Default adaptive config with residual skip connections and per-layer "
        "gating enabled.",
    ),
    ExperimentPreset.RESIDUAL_HALTING: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "stack_residual_connection_option": (
                    ResidualConnectionOptions.RESIDUAL
                ),
                "stack_halting_flag": True,
            }
        ),
        description="Default adaptive config with residual skip connections and stack "
        "halting enabled.",
    ),
    ExperimentPreset.RESIDUAL_MEMORY: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "stack_residual_connection_option": (
                    ResidualConnectionOptions.RESIDUAL
                ),
                "memory_flag": True,
            }
        ),
        description="Default adaptive config with residual skip connections and shared "
        "stack memory enabled.",
    ),
    ExperimentPreset.SINGLE_MODEL_WEIGHT: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "weight_option": SingleModelDynamicWeightConfig,
            }
        ),
        description="Default adaptive config with the single-model dynamic "
        "weight generator enabled.",
    ),
    ExperimentPreset.DUAL_MODEL_WEIGHT: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "weight_option": DualModelDynamicWeightConfig,
            }
        ),
        description="Default adaptive config with the dual-model dynamic weight generator "
        "enabled.",
    ),
    ExperimentPreset.LOW_RANK_WEIGHT: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "weight_option": LowRankDynamicWeightConfig,
            }
        ),
        description="Default adaptive config with the low-rank dynamic weight generator "
        "enabled.",
    ),
    ExperimentPreset.HYPERNETWORK_WEIGHT: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "weight_option": HypernetworkDynamicWeightConfig,
            }
        ),
        description="Default adaptive config with the hypernetwork dynamic "
        "weight generator enabled.",
    ),
    ExperimentPreset.LAYERED_WEIGHTED_BANK_WEIGHT: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "weight_option": LayeredWeightedBankDynamicWeightConfig,
            }
        ),
        description="Default adaptive config with the layered weighted-bank dynamic weight "
        "generator enabled.",
    ),
    ExperimentPreset.SOFT_WEIGHTED_BANK_WEIGHT: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "weight_option": SoftWeightedBankDynamicWeightConfig,
            }
        ),
        description="Default adaptive config with the soft weighted-bank dynamic weight "
        "generator enabled.",
    ),
    ExperimentPreset.AFFINE_TRANSFORM_BIAS: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "bias_option": AffineTransformDynamicBiasConfig,
            }
        ),
        description="Default adaptive config with affine-transform dynamic bias enabled.",
    ),
    ExperimentPreset.ADDITIVE_BIAS: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "bias_option": AdditiveDynamicBiasConfig,
            }
        ),
        description="Default adaptive config with additive dynamic bias enabled.",
    ),
    ExperimentPreset.GENERATOR_BIAS: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "bias_option": GeneratorDynamicBiasConfig,
            }
        ),
        description="Default adaptive config with generator-based dynamic bias enabled.",
    ),
    ExperimentPreset.MULTIPLICATIVE_BIAS: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "bias_option": MultiplicativeDynamicBiasConfig,
            }
        ),
        description="Default adaptive config with multiplicative dynamic bias enabled.",
    ),
    ExperimentPreset.SIGMOID_GATED_BIAS: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "bias_option": SigmoidGatedDynamicBiasConfig,
            }
        ),
        description="Default adaptive config with sigmoid-gated dynamic bias enabled.",
    ),
    ExperimentPreset.TANH_GATED_BIAS: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "bias_option": TanhGatedDynamicBiasConfig,
            }
        ),
        description="Default adaptive config with tanh-gated dynamic bias enabled.",
    ),
    ExperimentPreset.WEIGHTED_BANK_BIAS: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "bias_option": WeightedBankDynamicBiasConfig,
            }
        ),
        description="Default adaptive config with weighted-bank dynamic bias enabled.",
    ),
    ExperimentPreset.STANDARD_DIAGONAL: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "diagonal_option": StandardDynamicDiagonalConfig,
            }
        ),
        description="Default adaptive config with standard dynamic diagonal enabled.",
    ),
    ExperimentPreset.ANTI_DIAGONAL: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "diagonal_option": AntiDynamicDiagonalConfig,
            }
        ),
        description="Default adaptive config with anti dynamic diagonal enabled.",
    ),
    ExperimentPreset.COMBINED_DIAGONAL: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "diagonal_option": CombinedDynamicDiagonalConfig,
            }
        ),
        description="Default adaptive config with combined dynamic diagonal enabled.",
    ),
    ExperimentPreset.DIAGONAL_AXIS_MASK: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "row_mask_option": DiagonalAxisMaskConfig,
            }
        ),
        description="Default adaptive config with diagonal-axis row masking enabled.",
    ),
    ExperimentPreset.OUTER_PRODUCT_MASK: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "row_mask_option": OuterProductMaskConfig,
            }
        ),
        description="Default adaptive config with outer-product row masking enabled.",
    ),
    ExperimentPreset.PER_AXIS_SCORE_MASK: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "row_mask_option": PerAxisScoreMaskConfig,
            }
        ),
        description="Default adaptive config with per-axis score row masking enabled.",
    ),
    ExperimentPreset.TOP_SLICE_AXIS_MASK: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "row_mask_option": TopSliceAxisMaskConfig,
            }
        ),
        description="Default adaptive config with top-slice axis row masking enabled.",
    ),
    ExperimentPreset.WEIGHT_INFORMED_SCORE_MASK: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "row_mask_option": WeightInformedScoreAxisMaskConfig,
            }
        ),
        description="Default adaptive config with weight-informed score axis row masking "
        "enabled.",
    ),
    ExperimentPreset.SINGLE_MODEL_WEIGHT_ADDITIVE_BIAS_COMBINED_DIAGONAL: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "weight_option": SingleModelDynamicWeightConfig,
                "bias_option": AdditiveDynamicBiasConfig,
                "diagonal_option": CombinedDynamicDiagonalConfig,
            }
        ),
        description="Default adaptive config with single-model dynamic weights, additive "
        "bias, and combined dynamic diagonal enabled.",
    ),
    ExperimentPreset.DUAL_MODEL_WEIGHT_ADDITIVE_BIAS_COMBINED_DIAGONAL: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "weight_option": DualModelDynamicWeightConfig,
                "bias_option": AdditiveDynamicBiasConfig,
                "diagonal_option": CombinedDynamicDiagonalConfig,
            }
        ),
        description="Default adaptive config with dual-model dynamic weights, additive "
        "bias, and combined dynamic diagonal enabled.",
    ),
    ExperimentPreset.LAYERED_WEIGHTED_BANK_WEIGHT_ADDITIVE_BIAS_COMBINED_DIAGONAL: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "weight_option": LayeredWeightedBankDynamicWeightConfig,
                "bias_option": AdditiveDynamicBiasConfig,
                "diagonal_option": CombinedDynamicDiagonalConfig,
            }
        ),
        description="Default adaptive config with layered weighted-bank dynamic weights, "
        "additive bias, and combined dynamic diagonal enabled.",
    ),
    ExperimentPreset.LOW_RANK_WEIGHT_ADDITIVE_BIAS_COMBINED_DIAGONAL: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "weight_option": LowRankDynamicWeightConfig,
                "bias_option": AdditiveDynamicBiasConfig,
                "diagonal_option": CombinedDynamicDiagonalConfig,
            }
        ),
        description="Default adaptive config with low-rank dynamic weights, additive bias, "
        "and combined dynamic diagonal enabled.",
    ),
    ExperimentPreset.SINGLE_MODEL_WEIGHT_ADDITIVE_BIAS_STANDARD_DIAGONAL: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "weight_option": SingleModelDynamicWeightConfig,
                "bias_option": AdditiveDynamicBiasConfig,
                "diagonal_option": StandardDynamicDiagonalConfig,
            }
        ),
        description="Default adaptive config with single-model dynamic weights, additive "
        "bias, and standard dynamic diagonal enabled.",
    ),
    ExperimentPreset.DUAL_MODEL_WEIGHT_ADDITIVE_BIAS_STANDARD_DIAGONAL: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "weight_option": DualModelDynamicWeightConfig,
                "bias_option": AdditiveDynamicBiasConfig,
                "diagonal_option": StandardDynamicDiagonalConfig,
            }
        ),
        description="Default adaptive config with dual-model dynamic weights, additive "
        "bias, and standard dynamic diagonal enabled.",
    ),
    ExperimentPreset.LAYERED_WEIGHTED_BANK_WEIGHT_ADDITIVE_BIAS_STANDARD_DIAGONAL: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "weight_option": LayeredWeightedBankDynamicWeightConfig,
                "bias_option": AdditiveDynamicBiasConfig,
                "diagonal_option": StandardDynamicDiagonalConfig,
            }
        ),
        description="Default adaptive config with layered weighted-bank dynamic weights, "
        "additive bias, and standard dynamic diagonal enabled.",
    ),
    ExperimentPreset.LOW_RANK_WEIGHT_ADDITIVE_BIAS_STANDARD_DIAGONAL: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "weight_option": LowRankDynamicWeightConfig,
                "bias_option": AdditiveDynamicBiasConfig,
                "diagonal_option": StandardDynamicDiagonalConfig,
            }
        ),
        description="Default adaptive config with low-rank dynamic weights, additive bias, "
        "and standard dynamic diagonal enabled.",
    ),
    ExperimentPreset.DECAY_EXPONENTIAL_WEIGHT: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "weight_option": DualModelDynamicWeightConfig,
                "weight_decay_schedule": WeightDecayScheduleOptions.EXPONENTIAL,
                "weight_decay_rate": 1e-3,
                "weight_decay_warmup_batches": 500,
            }
        ),
        description="Default adaptive config with dual-model dynamic weights decaying "
        "exponentially toward a static linear layer.",
    ),
    ExperimentPreset.NORM_L2_WEIGHT: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "weight_option": DualModelDynamicWeightConfig,
                "weight_normalization_option": WeightNormalizationOptions.L2_SCALE,
            }
        ),
        description="Default adaptive config with dual-model dynamic weights and L2-scale "
        "weight normalization.",
    ),
    ExperimentPreset.DEEP_GENERATOR: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "weight_option": DualModelDynamicWeightConfig,
                "generator_depth": DynamicDepthOptions.DEPTH_OF_EIGHT,
            }
        ),
        description="Default adaptive config with dual-model dynamic weights produced by a "
        "depth-8 generator network.",
    ),
    ExperimentPreset.FULL_STACK: PresetDefinition(
        preset_values=_FULL_STACK_OVERRIDES,
        description="Default adaptive config with dual-model dynamic weights, additive "
        "bias, combined dynamic diagonal, and weight-informed row masking "
        "enabled.",
    ),
    ExperimentPreset.DUAL_WEIGHT_GATING: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "weight_option": DualModelDynamicWeightConfig,
                "stack_gate_flag": True,
            }
        ),
        description="Default adaptive config with dual-model dynamic weights and per-layer "
        "gating enabled.",
    ),
    ExperimentPreset.DUAL_WEIGHT_HALTING: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "weight_option": DualModelDynamicWeightConfig,
                "stack_halting_flag": True,
            }
        ),
        description="Default adaptive config with dual-model dynamic weights and stack "
        "halting enabled.",
    ),
    ExperimentPreset.DUAL_WEIGHT_GATING_HALTING: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "weight_option": DualModelDynamicWeightConfig,
                "stack_gate_flag": True,
                "stack_halting_flag": True,
            }
        ),
        description="Default adaptive config with dual-model dynamic weights, per-layer "
        "gating, and stack halting enabled.",
    ),
    ExperimentPreset.DUAL_WEIGHT_MEMORY: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "weight_option": DualModelDynamicWeightConfig,
                "memory_flag": True,
            }
        ),
        description="Default adaptive config with dual-model dynamic weights and shared "
        "stack memory enabled.",
    ),
    ExperimentPreset.DUAL_WEIGHT_GATING_MEMORY: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "weight_option": DualModelDynamicWeightConfig,
                "stack_gate_flag": True,
                "memory_flag": True,
            }
        ),
        description="Default adaptive config with dual-model dynamic weights, per-layer "
        "gating, and shared stack memory enabled.",
    ),
    ExperimentPreset.DUAL_WEIGHT_HALTING_MEMORY: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "weight_option": DualModelDynamicWeightConfig,
                "stack_halting_flag": True,
                "memory_flag": True,
            }
        ),
        description="Default adaptive config with dual-model dynamic weights, stack "
        "halting, and shared stack memory enabled.",
    ),
    ExperimentPreset.FULL_STACK_GATING: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                **_FULL_STACK_OVERRIDES,
                "stack_gate_flag": True,
            }
        ),
        description="Default adaptive config with full adaptive parameter controls and "
        "per-layer gating enabled.",
    ),
    ExperimentPreset.FULL_STACK_HALTING: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                **_FULL_STACK_OVERRIDES,
                "stack_halting_flag": True,
            }
        ),
        description="Default adaptive config with full adaptive parameter controls and "
        "stack halting enabled.",
    ),
    ExperimentPreset.FULL_STACK_MEMORY: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                **_FULL_STACK_OVERRIDES,
                "memory_flag": True,
            }
        ),
        description="Default adaptive config with full adaptive parameter controls and "
        "shared stack memory enabled.",
    ),
    ExperimentPreset.FULL_STACK_GATING_HALTING: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                **_FULL_STACK_OVERRIDES,
                "stack_gate_flag": True,
                "stack_halting_flag": True,
            }
        ),
        description="Default adaptive config with full adaptive parameter controls, "
        "per-layer gating, and stack halting enabled.",
    ),
    ExperimentPreset.FULL_STACK_RECURRENT: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                **_FULL_STACK_OVERRIDES,
                "recurrent_flag": True,
            }
        ),
        description="Default adaptive config with full adaptive parameter controls wrapped "
        "in fixed-step recurrence.",
    ),
    ExperimentPreset.BANK_WEIGHT_MASK: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "weight_option": LayeredWeightedBankDynamicWeightConfig,
                "row_mask_option": WeightInformedScoreAxisMaskConfig,
            }
        ),
        description="Default adaptive config with layered weighted-bank dynamic "
        "weights and weight-informed row masking enabled.",
    ),
    ExperimentPreset.LOW_RANK_POST_NORM: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "weight_option": LowRankDynamicWeightConfig,
                "layer_norm_position": LayerNormPositionOptions.AFTER,
            }
        ),
        description="Default adaptive config with low-rank dynamic weights and post-layer "
        "normalization enabled.",
    ),
    ExperimentPreset.RECURRENT: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "recurrent_flag": True,
            }
        ),
        description="Default adaptive config wrapped in fixed-step recurrence, reusing the "
        "adaptive linear stack for each recurrent step.",
    ),
    ExperimentPreset.RECURRENT_GATING: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "recurrent_flag": True,
                "recurrent_gate_flag": True,
            }
        ),
        description="Default recurrent adaptive config with step-level gating "
        "enabled after each recurrent update.",
    ),
    ExperimentPreset.RECURRENT_HALTING: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "recurrent_flag": True,
                "recurrent_halting_flag": True,
            }
        ),
        description="Default recurrent adaptive config with recurrent halting enabled, "
        "allowing early stopping before the max step count.",
    ),
    ExperimentPreset.RECURRENT_MEMORY: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "recurrent_flag": True,
                "memory_flag": True,
            }
        ),
        description="Default recurrent adaptive config whose reused hidden stack "
        "has shared memory enabled.",
    ),
    ExperimentPreset.RECURRENT_GATING_HALTING: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "recurrent_flag": True,
                "recurrent_gate_flag": True,
                "recurrent_halting_flag": True,
            }
        ),
        description="Default recurrent adaptive config with both step-level gating and "
        "recurrent halting enabled.",
    ),
    ExperimentPreset.RECURRENT_GATING_MEMORY: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "recurrent_flag": True,
                "recurrent_gate_flag": True,
                "memory_flag": True,
            }
        ),
        description="Default recurrent adaptive config with step-level gating and shared "
        "memory in the reused adaptive stack.",
    ),
    ExperimentPreset.RECURRENT_HALTING_MEMORY: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "recurrent_flag": True,
                "recurrent_halting_flag": True,
                "memory_flag": True,
            }
        ),
        description="Default recurrent adaptive config with recurrent halting and shared "
        "memory in the reused adaptive stack.",
    ),
    ExperimentPreset.RECURRENT_GATING_HALTING_MEMORY: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "recurrent_flag": True,
                "recurrent_gate_flag": True,
                "recurrent_halting_flag": True,
                "memory_flag": True,
            }
        ),
        description="Default recurrent adaptive config with step-level gating, recurrent "
        "halting, and shared memory in the reused adaptive stack.",
    ),
    ExperimentPreset.RECURRENT_RESIDUAL: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "recurrent_flag": True,
                "stack_residual_connection_option": (
                    ResidualConnectionOptions.RESIDUAL
                ),
            }
        ),
        description="Default recurrent adaptive config using a residual hidden stack at "
        "each recurrent step.",
    ),
    ExperimentPreset.RECURRENT_POST_NORM: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "recurrent_flag": True,
                "layer_norm_position": LayerNormPositionOptions.AFTER,
            }
        ),
        description="Default recurrent adaptive config using a post-normalized "
        "hidden stack at each recurrent step.",
    ),
}


class ExperimentPresets(BuilderBackedExperimentPresetsBase):
    def __init__(self) -> None:
        super().__init__(
            _PRESET_DEFINITIONS,
            builder_type=LinearAdaptiveConfigBuilder,
            default_preset=ExperimentPreset.BASELINE,
        )

    def _preset(self, **kwargs):
        return self._builder_type(runtime=runtime_from_flat(kwargs)).build()


class Experiment(ExperimentBase):
    def __init__(
        self,
        experiment_preset: ExperimentPreset | None = None,
        experiment_task=None,
        *,
        model_package=None,
        run_artifacts=None,
    ) -> None:
        super().__init__(
            experiment_preset,
            experiment_task=experiment_task,
            model_package=model_package,
            run_artifacts=run_artifacts,
        )

    def _num_epochs(self) -> int:
        return config.NUM_EPOCHS

    def _dataset_options(self) -> list:
        return dataset_options.DATASET_OPTIONS_BY_TASK[
            dataset_options.DEFAULT_EXPERIMENT_TASK
        ]

    def _model_type(self) -> type:
        return Model

    def _preset_generator_instance(self) -> ExperimentPresetsBase:
        return ExperimentPresets()

    def _experiment_preset_enum(self) -> type[BaseOptions]:
        return ExperimentPreset

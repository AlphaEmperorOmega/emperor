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
from emperor.experiments.base import (
    BuilderBackedExperimentPresetsBase,
    ExperimentBase,
    ExperimentPresetsBase,
    PresetDefinition,
)
from emperor.experts.core.options import (
    DroppedTokenOptions,
    ExpertWeightingPositionOptions,
    RoutingInitializationMode,
)

import models.experts.linear_adaptive.config as config
import models.experts.linear_adaptive.dataset_options as dataset_options
from models.experts.linear_adaptive.config_builder import (
    LinearAdaptiveConfigBuilder,
)
from models.experts.linear_adaptive.model import Model
from models.experts.linear_adaptive.runtime_defaults import runtime_from_flat


class ExperimentPreset(BaseOptions):
    BASELINE = 1
    GATING = 2
    HALTING = 3
    GATING_HALTING = 4
    RECURRENT = 5
    RECURRENT_GATING = 6
    RECURRENT_HALTING = 7
    RECURRENT_GATING_HALTING = 8
    ADAPTIVE_SHARED_ROUTER = 9
    ADAPTIVE_AFTER_WEIGHT = 10
    ADAPTIVE_TOP1_SWITCH = 11
    ADAPTIVE_FULL_SHARED = 12
    ADAPTIVE_FULL_CAPACITY = 13
    ADAPTIVE_BANK_ROUTER = 14
    MEMORY = 15
    GATING_MEMORY = 16
    HALTING_MEMORY = 17
    GATING_HALTING_MEMORY = 18
    RECURRENT_MEMORY = 19
    RECURRENT_GATING_MEMORY = 20
    RECURRENT_HALTING_MEMORY = 21
    RECURRENT_GATING_HALTING_MEMORY = 22
    RESIDUAL = 23
    POST_NORM = 24
    RESIDUAL_POST_NORM = 25
    RESIDUAL_GATING = 26
    RESIDUAL_HALTING = 27
    RESIDUAL_MEMORY = 28
    SINGLE_MODEL_WEIGHT = 29
    DUAL_MODEL_WEIGHT = 30
    LOW_RANK_WEIGHT = 31
    HYPERNETWORK_WEIGHT = 32
    LAYERED_WEIGHTED_BANK_WEIGHT = 33
    SOFT_WEIGHTED_BANK_WEIGHT = 34
    AFFINE_TRANSFORM_BIAS = 35
    ADDITIVE_BIAS = 36
    GENERATOR_BIAS = 37
    MULTIPLICATIVE_BIAS = 38
    SIGMOID_GATED_BIAS = 39
    TANH_GATED_BIAS = 40
    WEIGHTED_BANK_BIAS = 41
    STANDARD_DIAGONAL = 42
    ANTI_DIAGONAL = 43
    COMBINED_DIAGONAL = 44
    DIAGONAL_AXIS_MASK = 45
    OUTER_PRODUCT_MASK = 46
    PER_AXIS_SCORE_MASK = 47
    TOP_SLICE_AXIS_MASK = 48
    WEIGHT_INFORMED_SCORE_MASK = 49
    SINGLE_MODEL_WEIGHT_ADDITIVE_BIAS_COMBINED_DIAGONAL = 50
    DUAL_MODEL_WEIGHT_ADDITIVE_BIAS_COMBINED_DIAGONAL = 51
    LAYERED_WEIGHTED_BANK_WEIGHT_ADDITIVE_BIAS_COMBINED_DIAGONAL = 52
    LOW_RANK_WEIGHT_ADDITIVE_BIAS_COMBINED_DIAGONAL = 53
    SINGLE_MODEL_WEIGHT_ADDITIVE_BIAS_STANDARD_DIAGONAL = 54
    DUAL_MODEL_WEIGHT_ADDITIVE_BIAS_STANDARD_DIAGONAL = 55
    LAYERED_WEIGHTED_BANK_WEIGHT_ADDITIVE_BIAS_STANDARD_DIAGONAL = 56
    LOW_RANK_WEIGHT_ADDITIVE_BIAS_STANDARD_DIAGONAL = 57
    DECAY_EXPONENTIAL_WEIGHT = 58
    NORM_L2_WEIGHT = 59
    DEEP_GENERATOR = 60
    FULL_STACK = 61
    DUAL_WEIGHT_GATING = 62
    DUAL_WEIGHT_HALTING = 63
    DUAL_WEIGHT_GATING_HALTING = 64
    DUAL_WEIGHT_MEMORY = 65
    DUAL_WEIGHT_GATING_MEMORY = 66
    DUAL_WEIGHT_HALTING_MEMORY = 67
    FULL_STACK_GATING = 68
    FULL_STACK_HALTING = 69
    FULL_STACK_MEMORY = 70
    FULL_STACK_GATING_HALTING = 71
    FULL_STACK_RECURRENT = 72
    BANK_WEIGHT_MASK = 73
    LOW_RANK_POST_NORM = 74
    RECURRENT_RESIDUAL = 75
    RECURRENT_POST_NORM = 76


_ADAPTIVE_OPTION_FLAGS = {
    "weight_option": "weight_option_flag",
    "bias_option": "bias_option_flag",
    "diagonal_option": "diagonal_option_flag",
    "row_mask_option": "mask_option_flag",
    "router_weight_option": "router_weight_option_flag",
    "router_bias_option": "router_bias_option_flag",
    "router_diagonal_option": "router_diagonal_option_flag",
    "router_row_mask_option": "router_mask_option_flag",
}


def _with_adaptive_option_flags(overrides: dict[str, object]) -> dict[str, object]:
    option_flags = {
        flag: True
        for option, flag in _ADAPTIVE_OPTION_FLAGS.items()
        if option in overrides and overrides[option] is not None
    }
    return {**option_flags, **overrides}


def _full_adaptive_values() -> dict[str, object]:
    return {
        "weight_option": DualModelDynamicWeightConfig,
        "bias_option": AdditiveDynamicBiasConfig,
        "diagonal_option": CombinedDynamicDiagonalConfig,
        "row_mask_option": WeightInformedScoreAxisMaskConfig,
    }


_FULL_STACK_OVERRIDES = _with_adaptive_option_flags(_full_adaptive_values())


_PRESET_DEFINITIONS = {
    ExperimentPreset.BASELINE: PresetDefinition(
        preset_values={},
        description="Default config: a mixture-of-experts classifier with adaptive linear "
        "input, output, expert, and router stacks.",
    ),
    ExperimentPreset.GATING: PresetDefinition(
        preset_values={
            "stack_gate_flag": True,
        },
        description="Default config with per-layer gating enabled in the adaptive expert "
        "stack.",
    ),
    ExperimentPreset.HALTING: PresetDefinition(
        preset_values={
            "stack_halting_flag": True,
        },
        description="Default config with stack halting enabled in the adaptive expert "
        "stack.",
    ),
    ExperimentPreset.GATING_HALTING: PresetDefinition(
        preset_values={
            "stack_gate_flag": True,
            "stack_halting_flag": True,
        },
        description="Default config with both per-layer gating and stack halting "
        "enabled in the adaptive expert stack.",
    ),
    ExperimentPreset.MEMORY: PresetDefinition(
        preset_values={
            "memory_flag": True,
        },
        description="Default config with shared stack memory enabled in the adaptive "
        "expert stack.",
    ),
    ExperimentPreset.GATING_MEMORY: PresetDefinition(
        preset_values={
            "stack_gate_flag": True,
            "memory_flag": True,
        },
        description="Default config with both per-layer gating and shared stack "
        "memory enabled in the adaptive expert stack.",
    ),
    ExperimentPreset.HALTING_MEMORY: PresetDefinition(
        preset_values={
            "stack_halting_flag": True,
            "memory_flag": True,
        },
        description="Default config with both stack halting and shared stack memory "
        "enabled in the adaptive expert stack.",
    ),
    ExperimentPreset.GATING_HALTING_MEMORY: PresetDefinition(
        preset_values={
            "stack_gate_flag": True,
            "stack_halting_flag": True,
            "memory_flag": True,
        },
        description="Default config with per-layer gating, stack halting, and shared "
        "stack memory enabled in the adaptive expert stack.",
    ),
    ExperimentPreset.RESIDUAL: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "stack_residual_connection_option": (
                    ResidualConnectionOptions.RESIDUAL
                ),
            }
        ),
        description="Default adaptive expert config with residual skip connections.",
    ),
    ExperimentPreset.POST_NORM: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "layer_norm_position": LayerNormPositionOptions.AFTER,
            }
        ),
        description="Default adaptive expert config with post-layer normalization.",
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
        description="Default adaptive expert config with residual connections and "
        "post-layer normalization.",
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
        description="Default adaptive expert config with residual connections and "
        "per-layer gating.",
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
        description="Default adaptive expert config with residual connections and "
        "stack halting.",
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
        description="Default adaptive expert config with residual connections and "
        "shared stack memory.",
    ),
    ExperimentPreset.SINGLE_MODEL_WEIGHT: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "weight_option": SingleModelDynamicWeightConfig,
            }
        ),
        description="Default adaptive expert config with single-model dynamic weights.",
    ),
    ExperimentPreset.DUAL_MODEL_WEIGHT: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "weight_option": DualModelDynamicWeightConfig,
            }
        ),
        description="Default adaptive expert config with dual-model dynamic weights.",
    ),
    ExperimentPreset.LOW_RANK_WEIGHT: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "weight_option": LowRankDynamicWeightConfig,
            }
        ),
        description="Default adaptive expert config with low-rank dynamic weights.",
    ),
    ExperimentPreset.HYPERNETWORK_WEIGHT: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "weight_option": HypernetworkDynamicWeightConfig,
            }
        ),
        description="Default adaptive expert config with hypernetwork dynamic weights.",
    ),
    ExperimentPreset.LAYERED_WEIGHTED_BANK_WEIGHT: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "weight_option": LayeredWeightedBankDynamicWeightConfig,
            }
        ),
        description="Default adaptive expert config with layered weighted-bank dynamic "
        "weights.",
    ),
    ExperimentPreset.SOFT_WEIGHTED_BANK_WEIGHT: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "weight_option": SoftWeightedBankDynamicWeightConfig,
            }
        ),
        description="Default adaptive expert config with soft weighted-bank dynamic "
        "weights.",
    ),
    ExperimentPreset.AFFINE_TRANSFORM_BIAS: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "bias_option": AffineTransformDynamicBiasConfig,
            }
        ),
        description="Default adaptive expert config with affine-transform dynamic bias.",
    ),
    ExperimentPreset.ADDITIVE_BIAS: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "bias_option": AdditiveDynamicBiasConfig,
            }
        ),
        description="Default adaptive expert config with additive dynamic bias.",
    ),
    ExperimentPreset.GENERATOR_BIAS: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "bias_option": GeneratorDynamicBiasConfig,
            }
        ),
        description="Default adaptive expert config with generator dynamic bias.",
    ),
    ExperimentPreset.MULTIPLICATIVE_BIAS: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "bias_option": MultiplicativeDynamicBiasConfig,
            }
        ),
        description="Default adaptive expert config with multiplicative dynamic bias.",
    ),
    ExperimentPreset.SIGMOID_GATED_BIAS: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "bias_option": SigmoidGatedDynamicBiasConfig,
            }
        ),
        description="Default adaptive expert config with sigmoid-gated dynamic bias.",
    ),
    ExperimentPreset.TANH_GATED_BIAS: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "bias_option": TanhGatedDynamicBiasConfig,
            }
        ),
        description="Default adaptive expert config with tanh-gated dynamic bias.",
    ),
    ExperimentPreset.WEIGHTED_BANK_BIAS: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "bias_option": WeightedBankDynamicBiasConfig,
            }
        ),
        description="Default adaptive expert config with weighted-bank dynamic bias.",
    ),
    ExperimentPreset.STANDARD_DIAGONAL: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "diagonal_option": StandardDynamicDiagonalConfig,
            }
        ),
        description="Default adaptive expert config with standard dynamic diagonal.",
    ),
    ExperimentPreset.ANTI_DIAGONAL: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "diagonal_option": AntiDynamicDiagonalConfig,
            }
        ),
        description="Default adaptive expert config with anti dynamic diagonal.",
    ),
    ExperimentPreset.COMBINED_DIAGONAL: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "diagonal_option": CombinedDynamicDiagonalConfig,
            }
        ),
        description="Default adaptive expert config with combined dynamic diagonal.",
    ),
    ExperimentPreset.DIAGONAL_AXIS_MASK: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "row_mask_option": DiagonalAxisMaskConfig,
            }
        ),
        description="Default adaptive expert config with diagonal-axis row masking.",
    ),
    ExperimentPreset.OUTER_PRODUCT_MASK: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "row_mask_option": OuterProductMaskConfig,
            }
        ),
        description="Default adaptive expert config with outer-product row masking.",
    ),
    ExperimentPreset.PER_AXIS_SCORE_MASK: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "row_mask_option": PerAxisScoreMaskConfig,
            }
        ),
        description="Default adaptive expert config with per-axis score row masking.",
    ),
    ExperimentPreset.TOP_SLICE_AXIS_MASK: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "row_mask_option": TopSliceAxisMaskConfig,
            }
        ),
        description="Default adaptive expert config with top-slice axis row masking.",
    ),
    ExperimentPreset.WEIGHT_INFORMED_SCORE_MASK: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "row_mask_option": WeightInformedScoreAxisMaskConfig,
            }
        ),
        description="Default adaptive expert config with weight-informed score row "
        "masking.",
    ),
    ExperimentPreset.SINGLE_MODEL_WEIGHT_ADDITIVE_BIAS_COMBINED_DIAGONAL: (
        PresetDefinition(
            preset_values=_with_adaptive_option_flags(
                {
                    "weight_option": SingleModelDynamicWeightConfig,
                    "bias_option": AdditiveDynamicBiasConfig,
                    "diagonal_option": CombinedDynamicDiagonalConfig,
                }
            ),
            description="Default adaptive expert config with single-model weights, "
            "additive bias, and combined diagonal.",
        )
    ),
    ExperimentPreset.DUAL_MODEL_WEIGHT_ADDITIVE_BIAS_COMBINED_DIAGONAL: (
        PresetDefinition(
            preset_values=_with_adaptive_option_flags(
                {
                    "weight_option": DualModelDynamicWeightConfig,
                    "bias_option": AdditiveDynamicBiasConfig,
                    "diagonal_option": CombinedDynamicDiagonalConfig,
                }
            ),
            description="Default adaptive expert config with dual-model weights, "
            "additive bias, and combined diagonal.",
        )
    ),
    ExperimentPreset.LAYERED_WEIGHTED_BANK_WEIGHT_ADDITIVE_BIAS_COMBINED_DIAGONAL: (
        PresetDefinition(
            preset_values=_with_adaptive_option_flags(
                {
                    "weight_option": LayeredWeightedBankDynamicWeightConfig,
                    "bias_option": AdditiveDynamicBiasConfig,
                    "diagonal_option": CombinedDynamicDiagonalConfig,
                }
            ),
            description="Default adaptive expert config with layered weighted-bank "
            "weights, additive bias, and combined diagonal.",
        )
    ),
    ExperimentPreset.LOW_RANK_WEIGHT_ADDITIVE_BIAS_COMBINED_DIAGONAL: (
        PresetDefinition(
            preset_values=_with_adaptive_option_flags(
                {
                    "weight_option": LowRankDynamicWeightConfig,
                    "bias_option": AdditiveDynamicBiasConfig,
                    "diagonal_option": CombinedDynamicDiagonalConfig,
                }
            ),
            description="Default adaptive expert config with low-rank weights, "
            "additive bias, and combined diagonal.",
        )
    ),
    ExperimentPreset.SINGLE_MODEL_WEIGHT_ADDITIVE_BIAS_STANDARD_DIAGONAL: (
        PresetDefinition(
            preset_values=_with_adaptive_option_flags(
                {
                    "weight_option": SingleModelDynamicWeightConfig,
                    "bias_option": AdditiveDynamicBiasConfig,
                    "diagonal_option": StandardDynamicDiagonalConfig,
                }
            ),
            description="Default adaptive expert config with single-model weights, "
            "additive bias, and standard diagonal.",
        )
    ),
    ExperimentPreset.DUAL_MODEL_WEIGHT_ADDITIVE_BIAS_STANDARD_DIAGONAL: (
        PresetDefinition(
            preset_values=_with_adaptive_option_flags(
                {
                    "weight_option": DualModelDynamicWeightConfig,
                    "bias_option": AdditiveDynamicBiasConfig,
                    "diagonal_option": StandardDynamicDiagonalConfig,
                }
            ),
            description="Default adaptive expert config with dual-model weights, "
            "additive bias, and standard diagonal.",
        )
    ),
    ExperimentPreset.LAYERED_WEIGHTED_BANK_WEIGHT_ADDITIVE_BIAS_STANDARD_DIAGONAL: (
        PresetDefinition(
            preset_values=_with_adaptive_option_flags(
                {
                    "weight_option": LayeredWeightedBankDynamicWeightConfig,
                    "bias_option": AdditiveDynamicBiasConfig,
                    "diagonal_option": StandardDynamicDiagonalConfig,
                }
            ),
            description="Default adaptive expert config with layered weighted-bank "
            "weights, additive bias, and standard diagonal.",
        )
    ),
    ExperimentPreset.LOW_RANK_WEIGHT_ADDITIVE_BIAS_STANDARD_DIAGONAL: (
        PresetDefinition(
            preset_values=_with_adaptive_option_flags(
                {
                    "weight_option": LowRankDynamicWeightConfig,
                    "bias_option": AdditiveDynamicBiasConfig,
                    "diagonal_option": StandardDynamicDiagonalConfig,
                }
            ),
            description="Default adaptive expert config with low-rank weights, "
            "additive bias, and standard diagonal.",
        )
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
        description="Default adaptive expert config with exponentially decayed "
        "dual-model dynamic weights.",
    ),
    ExperimentPreset.NORM_L2_WEIGHT: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "weight_option": DualModelDynamicWeightConfig,
                "weight_normalization_option": WeightNormalizationOptions.L2_SCALE,
            }
        ),
        description="Default adaptive expert config with L2-scale normalized dynamic "
        "weights.",
    ),
    ExperimentPreset.DEEP_GENERATOR: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "weight_option": DualModelDynamicWeightConfig,
                "generator_depth": DynamicDepthOptions.DEPTH_OF_EIGHT,
            }
        ),
        description="Default adaptive expert config with a depth-8 weight generator.",
    ),
    ExperimentPreset.FULL_STACK: PresetDefinition(
        preset_values=_FULL_STACK_OVERRIDES,
        description="Default adaptive expert config with dynamic weights, bias, "
        "diagonal, and row masking.",
    ),
    ExperimentPreset.DUAL_WEIGHT_GATING: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "weight_option": DualModelDynamicWeightConfig,
                "stack_gate_flag": True,
            }
        ),
        description="Default adaptive expert config with dual-model weights and "
        "per-layer gating.",
    ),
    ExperimentPreset.DUAL_WEIGHT_HALTING: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "weight_option": DualModelDynamicWeightConfig,
                "stack_halting_flag": True,
            }
        ),
        description="Default adaptive expert config with dual-model weights and "
        "stack halting.",
    ),
    ExperimentPreset.DUAL_WEIGHT_GATING_HALTING: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "weight_option": DualModelDynamicWeightConfig,
                "stack_gate_flag": True,
                "stack_halting_flag": True,
            }
        ),
        description="Default adaptive expert config with dual-model weights, "
        "per-layer gating, and stack halting.",
    ),
    ExperimentPreset.DUAL_WEIGHT_MEMORY: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "weight_option": DualModelDynamicWeightConfig,
                "memory_flag": True,
            }
        ),
        description="Default adaptive expert config with dual-model weights and "
        "shared stack memory.",
    ),
    ExperimentPreset.DUAL_WEIGHT_GATING_MEMORY: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "weight_option": DualModelDynamicWeightConfig,
                "stack_gate_flag": True,
                "memory_flag": True,
            }
        ),
        description="Default adaptive expert config with dual-model weights, "
        "per-layer gating, and shared stack memory.",
    ),
    ExperimentPreset.DUAL_WEIGHT_HALTING_MEMORY: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "weight_option": DualModelDynamicWeightConfig,
                "stack_halting_flag": True,
                "memory_flag": True,
            }
        ),
        description="Default adaptive expert config with dual-model weights, "
        "stack halting, and shared stack memory.",
    ),
    ExperimentPreset.FULL_STACK_GATING: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                **_FULL_STACK_OVERRIDES,
                "stack_gate_flag": True,
            }
        ),
        description="Default full adaptive expert config with per-layer gating.",
    ),
    ExperimentPreset.FULL_STACK_HALTING: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                **_FULL_STACK_OVERRIDES,
                "stack_halting_flag": True,
            }
        ),
        description="Default full adaptive expert config with stack halting.",
    ),
    ExperimentPreset.FULL_STACK_MEMORY: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                **_FULL_STACK_OVERRIDES,
                "memory_flag": True,
            }
        ),
        description="Default full adaptive expert config with shared stack memory.",
    ),
    ExperimentPreset.FULL_STACK_GATING_HALTING: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                **_FULL_STACK_OVERRIDES,
                "stack_gate_flag": True,
                "stack_halting_flag": True,
            }
        ),
        description="Default full adaptive expert config with per-layer gating and "
        "stack halting.",
    ),
    ExperimentPreset.FULL_STACK_RECURRENT: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                **_FULL_STACK_OVERRIDES,
                "recurrent_flag": True,
            }
        ),
        description="Default full adaptive expert config wrapped in recurrence.",
    ),
    ExperimentPreset.BANK_WEIGHT_MASK: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "weight_option": LayeredWeightedBankDynamicWeightConfig,
                "row_mask_option": WeightInformedScoreAxisMaskConfig,
            }
        ),
        description="Default adaptive expert config with bank weights and "
        "weight-informed row masking.",
    ),
    ExperimentPreset.LOW_RANK_POST_NORM: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "weight_option": LowRankDynamicWeightConfig,
                "layer_norm_position": LayerNormPositionOptions.AFTER,
            }
        ),
        description="Default adaptive expert config with low-rank weights and "
        "post-layer normalization.",
    ),
    ExperimentPreset.RECURRENT: PresetDefinition(
        preset_values={
            "recurrent_flag": True,
        },
        description="Default config wrapped in fixed-step recurrence, reusing the adaptive "
        "expert stack for each recurrent step.",
    ),
    ExperimentPreset.RECURRENT_GATING: PresetDefinition(
        preset_values={
            "recurrent_flag": True,
            "recurrent_gate_flag": True,
        },
        description="Default recurrent config with step-level gating enabled after each "
        "recurrent update.",
    ),
    ExperimentPreset.RECURRENT_HALTING: PresetDefinition(
        preset_values={
            "recurrent_flag": True,
            "recurrent_halting_flag": True,
        },
        description="Default recurrent config with recurrent halting enabled, allowing "
        "early stopping before the max step count.",
    ),
    ExperimentPreset.RECURRENT_GATING_HALTING: PresetDefinition(
        preset_values={
            "recurrent_flag": True,
            "recurrent_gate_flag": True,
            "recurrent_halting_flag": True,
        },
        description="Default recurrent config with both step-level gating and recurrent "
        "halting enabled.",
    ),
    ExperimentPreset.RECURRENT_MEMORY: PresetDefinition(
        preset_values={
            "recurrent_flag": True,
            "memory_flag": True,
        },
        description="Default recurrent config whose reused adaptive expert stack has "
        "shared memory enabled.",
    ),
    ExperimentPreset.RECURRENT_GATING_MEMORY: PresetDefinition(
        preset_values={
            "recurrent_flag": True,
            "recurrent_gate_flag": True,
            "memory_flag": True,
        },
        description="Default recurrent config with step-level gating and shared memory "
        "in the reused adaptive expert stack.",
    ),
    ExperimentPreset.RECURRENT_HALTING_MEMORY: PresetDefinition(
        preset_values={
            "recurrent_flag": True,
            "recurrent_halting_flag": True,
            "memory_flag": True,
        },
        description="Default recurrent config with recurrent halting and shared memory "
        "in the reused adaptive expert stack.",
    ),
    ExperimentPreset.RECURRENT_GATING_HALTING_MEMORY: PresetDefinition(
        preset_values={
            "recurrent_flag": True,
            "recurrent_gate_flag": True,
            "recurrent_halting_flag": True,
            "memory_flag": True,
        },
        description="Default recurrent config with step-level gating, recurrent "
        "halting, and shared memory in the reused adaptive expert stack.",
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
        description="Default recurrent adaptive expert config using a residual expert "
        "stack at each recurrent step.",
    ),
    ExperimentPreset.RECURRENT_POST_NORM: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "recurrent_flag": True,
                "layer_norm_position": LayerNormPositionOptions.AFTER,
            }
        ),
        description="Default recurrent adaptive expert config using a post-normalized "
        "expert stack at each recurrent step.",
    ),
    ExperimentPreset.ADAPTIVE_SHARED_ROUTER: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "weight_option": DualModelDynamicWeightConfig,
                "routing_initialization_mode": RoutingInitializationMode.SHARED,
            }
        ),
        description="Default adaptive config with dual-model dynamic weights and shared "
        "expert routing.",
    ),
    ExperimentPreset.ADAPTIVE_AFTER_WEIGHT: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "weight_option": DualModelDynamicWeightConfig,
                "weighting_position_option": (
                    ExpertWeightingPositionOptions.AFTER_EXPERTS
                ),
            }
        ),
        description="Default adaptive config with dual-model dynamic weights and expert "
        "weighting after expert outputs.",
    ),
    ExperimentPreset.ADAPTIVE_TOP1_SWITCH: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "weight_option": DualModelDynamicWeightConfig,
                "top_k": 1,
                "sampler_normalize_probabilities_flag": False,
                "sampler_switch_loss_weight": 0.1,
            }
        ),
        description="Default adaptive config with dual-model dynamic weights, top-1 switch "
        "routing, and switch auxiliary loss enabled.",
    ),
    ExperimentPreset.ADAPTIVE_FULL_SHARED: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                **_full_adaptive_values(),
                "routing_initialization_mode": RoutingInitializationMode.SHARED,
            }
        ),
        description="Default adaptive config with dynamic weights, bias, diagonal, row "
        "mask, and shared expert routing.",
    ),
    ExperimentPreset.ADAPTIVE_FULL_CAPACITY: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                **_full_adaptive_values(),
                "top_k": 1,
                "capacity_factor": 1.0,
                "dropped_token_behavior": DroppedTokenOptions.ZEROS,
                "sampler_normalize_probabilities_flag": False,
            }
        ),
        description="Default adaptive config with dynamic weights, bias, diagonal, row "
        "mask, top-1 capacity limiting, and dropped tokens zeroed.",
    ),
    ExperimentPreset.ADAPTIVE_BANK_ROUTER: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "router_weight_option": LayeredWeightedBankDynamicWeightConfig,
                "routing_initialization_mode": RoutingInitializationMode.SHARED,
            }
        ),
        description="Default adaptive config with layered weighted-bank dynamic "
        "weights and shared expert routing.",
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
        return self._builder_type(runtime=runtime_from_flat(kwargs, config)).build()


class Experiment(ExperimentBase):
    def __init__(
        self,
        experiment_preset: ExperimentPreset | None = None,
        experiment_task=None,
    ) -> None:
        super().__init__(experiment_preset, experiment_task=experiment_task)

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

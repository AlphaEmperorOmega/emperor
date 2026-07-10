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
    SoftWeightedBankDynamicWeightConfig,
)
from emperor.augmentations.adaptive_parameters.options import (
    DynamicDepthOptions,
    WeightDecayScheduleOptions,
    WeightNormalizationOptions,
)
from emperor.base.layer.residual import ResidualConnectionOptions
from emperor.base.options import BaseOptions, LayerNormPositionOptions
from emperor.datasets.text.language_modeling import WikiText2
from emperor.embedding.absolute.core.config import (
    TextSinusoidalPositionalEmbeddingConfig,
)
from emperor.experiments.base import (
    BuilderBackedExperimentPresetsBase,
    ExperimentBase,
    ExperimentPresetsBase,
    PresetDefinition,
)

import models.gpt.linear_adaptive.config as config
import models.gpt.linear_adaptive.dataset_options as dataset_options
from models.gpt.linear_adaptive._builder_adapter import (
    linear_adaptive_builder_kwargs_from_flat,
)
from models.gpt.linear_adaptive.config_builder import (
    GptLinearAdaptiveConfigBuilder,
)
from models.gpt.linear_adaptive.model import Model


class ExperimentPreset(BaseOptions):
    BASELINE = 1
    PRE_NORM = 2
    POST_NORM = 3
    SINUSOIDAL = 4
    ATTENTION_BIAS = 5
    GATING = 6
    HALTING = 7
    GATING_HALTING = 8
    MEMORY = 9
    GATING_MEMORY = 10
    HALTING_MEMORY = 11
    GATING_HALTING_MEMORY = 12
    RECURRENT = 13
    RECURRENT_GATING = 14
    RECURRENT_HALTING = 15
    RECURRENT_MEMORY = 16
    RECURRENT_GATING_HALTING = 17
    RECURRENT_GATING_MEMORY = 18
    RECURRENT_HALTING_MEMORY = 19
    RECURRENT_GATING_HALTING_MEMORY = 20
    RESIDUAL = 21
    RESIDUAL_POST_NORM = 22
    RESIDUAL_GATING = 23
    RESIDUAL_HALTING = 24
    RESIDUAL_MEMORY = 25
    RECURRENT_RESIDUAL = 26
    RECURRENT_POST_NORM = 27
    SINGLE_MODEL_WEIGHT = 28
    DUAL_MODEL_WEIGHT = 29
    LOW_RANK_WEIGHT = 30
    HYPERNETWORK_WEIGHT = 31
    LAYERED_WEIGHTED_BANK_WEIGHT = 32
    SOFT_WEIGHTED_BANK_WEIGHT = 33
    AFFINE_TRANSFORM_BIAS = 34
    ADDITIVE_BIAS = 35
    GENERATOR_BIAS = 36
    MULTIPLICATIVE_BIAS = 37
    SIGMOID_GATED_BIAS = 38
    TANH_GATED_BIAS = 39
    WEIGHTED_BANK_BIAS = 40
    STANDARD_DIAGONAL = 41
    ANTI_DIAGONAL = 42
    COMBINED_DIAGONAL = 43
    DIAGONAL_AXIS_MASK = 44
    OUTER_PRODUCT_MASK = 45
    PER_AXIS_SCORE_MASK = 46
    TOP_SLICE_AXIS_MASK = 47
    WEIGHT_INFORMED_SCORE_MASK = 48
    SINGLE_MODEL_WEIGHT_ADDITIVE_BIAS_COMBINED_DIAGONAL = 49
    DUAL_MODEL_WEIGHT_ADDITIVE_BIAS_COMBINED_DIAGONAL = 50
    LAYERED_WEIGHTED_BANK_WEIGHT_ADDITIVE_BIAS_COMBINED_DIAGONAL = 51
    LOW_RANK_WEIGHT_ADDITIVE_BIAS_COMBINED_DIAGONAL = 52
    SINGLE_MODEL_WEIGHT_ADDITIVE_BIAS_STANDARD_DIAGONAL = 53
    DUAL_MODEL_WEIGHT_ADDITIVE_BIAS_STANDARD_DIAGONAL = 54
    LAYERED_WEIGHTED_BANK_WEIGHT_ADDITIVE_BIAS_STANDARD_DIAGONAL = 55
    LOW_RANK_WEIGHT_ADDITIVE_BIAS_STANDARD_DIAGONAL = 56
    DECAY_EXPONENTIAL_WEIGHT = 57
    NORM_L2_WEIGHT = 58
    DEEP_GENERATOR = 59
    FULL_STACK = 60
    DUAL_WEIGHT_GATING = 61
    DUAL_WEIGHT_HALTING = 62
    DUAL_WEIGHT_GATING_HALTING = 63
    DUAL_WEIGHT_MEMORY = 64
    DUAL_WEIGHT_GATING_MEMORY = 65
    DUAL_WEIGHT_HALTING_MEMORY = 66
    FULL_STACK_GATING = 67
    FULL_STACK_HALTING = 68
    FULL_STACK_MEMORY = 69
    FULL_STACK_GATING_HALTING = 70
    FULL_STACK_RECURRENT = 71
    BANK_WEIGHT_MASK = 72
    LOW_RANK_POST_NORM = 73


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
        description=(
            "Default config: a GPT-style causal language-modeling token decoder "
            "with adaptive-capable attention and feed-forward linear stacks, "
            "learned positional embeddings, and causal attention."
        ),
    ),
    ExperimentPreset.PRE_NORM: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {"layer_norm_position": LayerNormPositionOptions.BEFORE}
        ),
        description="Default config with layer normalization applied before each "
        "decoder sub-block.",
    ),
    ExperimentPreset.POST_NORM: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {"layer_norm_position": LayerNormPositionOptions.AFTER}
        ),
        description="Default config with layer normalization applied after each "
        "decoder sub-block.",
    ),
    ExperimentPreset.SINUSOIDAL: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {"positional_embedding_option": (TextSinusoidalPositionalEmbeddingConfig)}
        ),
        description="Default config with fixed sinusoidal token positional embeddings.",
    ),
    ExperimentPreset.ATTENTION_BIAS: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "attn_bias_flag": True,
                "attn_add_key_value_bias_flag": True,
            }
        ),
        description="Default config with attention projection bias and key/value bias "
        "enabled.",
    ),
    ExperimentPreset.GATING: PresetDefinition(
        preset_values=_with_adaptive_option_flags({"stack_gate_flag": True}),
        description="Default config with per-decoder-block gating enabled.",
    ),
    ExperimentPreset.HALTING: PresetDefinition(
        preset_values=_with_adaptive_option_flags({"stack_halting_flag": True}),
        description="Default config with decoder-block stack halting enabled.",
    ),
    ExperimentPreset.GATING_HALTING: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "stack_gate_flag": True,
                "stack_halting_flag": True,
            }
        ),
        description="Default config with both decoder-block gating and halting "
        "enabled.",
    ),
    ExperimentPreset.MEMORY: PresetDefinition(
        preset_values=_with_adaptive_option_flags({"memory_flag": True}),
        description="Default config with shared decoder-stack memory enabled.",
    ),
    ExperimentPreset.GATING_MEMORY: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {"stack_gate_flag": True, "memory_flag": True}
        ),
        description="Default config with decoder-block gating and shared memory "
        "enabled.",
    ),
    ExperimentPreset.HALTING_MEMORY: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {"stack_halting_flag": True, "memory_flag": True}
        ),
        description="Default config with decoder-block halting and shared memory "
        "enabled.",
    ),
    ExperimentPreset.GATING_HALTING_MEMORY: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "stack_gate_flag": True,
                "stack_halting_flag": True,
                "memory_flag": True,
            }
        ),
        description="Default config with decoder-block gating, halting, and shared "
        "memory.",
    ),
    ExperimentPreset.RECURRENT: PresetDefinition(
        preset_values=_with_adaptive_option_flags({"recurrent_flag": True}),
        description="Default token decoder stack wrapped in fixed-step recurrence.",
    ),
    ExperimentPreset.RECURRENT_GATING: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {"recurrent_flag": True, "recurrent_gate_flag": True}
        ),
        description="Default recurrent token decoder with step-level gating enabled.",
    ),
    ExperimentPreset.RECURRENT_HALTING: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {"recurrent_flag": True, "recurrent_halting_flag": True}
        ),
        description="Default recurrent token decoder with recurrent halting enabled.",
    ),
    ExperimentPreset.RECURRENT_MEMORY: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {"recurrent_flag": True, "memory_flag": True}
        ),
        description="Default recurrent token decoder whose reused stack has shared "
        "memory.",
    ),
    ExperimentPreset.RECURRENT_GATING_HALTING: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "recurrent_flag": True,
                "recurrent_gate_flag": True,
                "recurrent_halting_flag": True,
            }
        ),
        description="Default recurrent token decoder with step-level gating and "
        "halting.",
    ),
    ExperimentPreset.RECURRENT_GATING_MEMORY: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "recurrent_flag": True,
                "recurrent_gate_flag": True,
                "memory_flag": True,
            }
        ),
        description="Default recurrent token decoder with step-level gating and shared "
        "memory.",
    ),
    ExperimentPreset.RECURRENT_HALTING_MEMORY: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "recurrent_flag": True,
                "recurrent_halting_flag": True,
                "memory_flag": True,
            }
        ),
        description="Default recurrent token decoder with recurrent halting and shared "
        "memory.",
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
        description="Default recurrent token decoder with step-level gating, recurrent "
        "halting, and shared memory.",
    ),
    ExperimentPreset.RESIDUAL: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {"stack_residual_connection_option": (ResidualConnectionOptions.RESIDUAL)}
        ),
        description="Default config with residual skip connections between same-width "
        "decoder stack layers.",
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
        description="Default config with residual skip connections and post-layer "
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
        description="Default config with residual skip connections and per-layer "
        "decoder gating enabled.",
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
        description="Default config with residual skip connections and decoder-stack "
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
        description="Default config with residual skip connections and shared decoder "
        "memory enabled.",
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
        description="Default recurrent config using a residual token decoder stack at "
        "each step.",
    ),
    ExperimentPreset.RECURRENT_POST_NORM: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "recurrent_flag": True,
                "layer_norm_position": LayerNormPositionOptions.AFTER,
            }
        ),
        description="Default recurrent config with post-layer normalization inside the "
        "reused token decoder stack.",
    ),
    ExperimentPreset.SINGLE_MODEL_WEIGHT: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {"weight_option": DualModelDynamicWeightConfig}
        ),
        description="GPT-compatible dynamic weights for the single-model preset "
        "family on eligible decoder attention and feed-forward linear stacks.",
    ),
    ExperimentPreset.DUAL_MODEL_WEIGHT: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {"weight_option": DualModelDynamicWeightConfig}
        ),
        description="Default config with dual-model dynamic weights on eligible GPT "
        "decoder linear stacks.",
    ),
    ExperimentPreset.LOW_RANK_WEIGHT: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {"weight_option": LowRankDynamicWeightConfig}
        ),
        description="Default config with low-rank dynamic weights on eligible GPT "
        "decoder linear stacks.",
    ),
    ExperimentPreset.HYPERNETWORK_WEIGHT: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {"weight_option": HypernetworkDynamicWeightConfig}
        ),
        description="Default config with hypernetwork dynamic weights on eligible GPT "
        "decoder linear stacks.",
    ),
    ExperimentPreset.LAYERED_WEIGHTED_BANK_WEIGHT: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {"weight_option": LayeredWeightedBankDynamicWeightConfig}
        ),
        description="Default config with layered weighted-bank dynamic weights on "
        "eligible GPT decoder linear stacks.",
    ),
    ExperimentPreset.SOFT_WEIGHTED_BANK_WEIGHT: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {"weight_option": SoftWeightedBankDynamicWeightConfig}
        ),
        description="Default config with soft weighted-bank dynamic weights on "
        "eligible GPT decoder linear stacks.",
    ),
    ExperimentPreset.AFFINE_TRANSFORM_BIAS: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {"bias_option": AffineTransformDynamicBiasConfig}
        ),
        description="Default config with affine-transform dynamic bias on eligible "
        "GPT decoder linear stacks.",
    ),
    ExperimentPreset.ADDITIVE_BIAS: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {"bias_option": AdditiveDynamicBiasConfig}
        ),
        description="Default config with additive dynamic bias on eligible GPT "
        "decoder linear stacks.",
    ),
    ExperimentPreset.GENERATOR_BIAS: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {"bias_option": GeneratorDynamicBiasConfig}
        ),
        description="Default config with generator-based dynamic bias on eligible GPT "
        "decoder linear stacks.",
    ),
    ExperimentPreset.MULTIPLICATIVE_BIAS: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {"bias_option": MultiplicativeDynamicBiasConfig}
        ),
        description="Default config with multiplicative dynamic bias on eligible GPT "
        "decoder linear stacks.",
    ),
    ExperimentPreset.SIGMOID_GATED_BIAS: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {"bias_option": SigmoidGatedDynamicBiasConfig}
        ),
        description="Default config with sigmoid-gated dynamic bias on eligible GPT "
        "decoder linear stacks.",
    ),
    ExperimentPreset.TANH_GATED_BIAS: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {"bias_option": TanhGatedDynamicBiasConfig}
        ),
        description="Default config with tanh-gated dynamic bias on eligible GPT "
        "decoder linear stacks.",
    ),
    ExperimentPreset.WEIGHTED_BANK_BIAS: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {"bias_option": WeightedBankDynamicBiasConfig}
        ),
        description="Default config with weighted-bank dynamic bias on eligible GPT "
        "decoder linear stacks.",
    ),
    ExperimentPreset.STANDARD_DIAGONAL: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {"diagonal_option": StandardDynamicDiagonalConfig}
        ),
        description="Default config with a standard dynamic diagonal on eligible GPT "
        "decoder linear stacks.",
    ),
    ExperimentPreset.ANTI_DIAGONAL: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {"diagonal_option": AntiDynamicDiagonalConfig}
        ),
        description="Default config with an anti dynamic diagonal on eligible GPT "
        "decoder linear stacks.",
    ),
    ExperimentPreset.COMBINED_DIAGONAL: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {"diagonal_option": CombinedDynamicDiagonalConfig}
        ),
        description="Default config with a combined dynamic diagonal on eligible GPT "
        "decoder linear stacks.",
    ),
    ExperimentPreset.DIAGONAL_AXIS_MASK: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {"row_mask_option": DiagonalAxisMaskConfig}
        ),
        description="Default config with diagonal-axis row masking on eligible GPT "
        "decoder linear stacks.",
    ),
    ExperimentPreset.OUTER_PRODUCT_MASK: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {"row_mask_option": OuterProductMaskConfig}
        ),
        description="Default config with outer-product row masking on eligible GPT "
        "decoder linear stacks.",
    ),
    ExperimentPreset.PER_AXIS_SCORE_MASK: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {"row_mask_option": PerAxisScoreMaskConfig}
        ),
        description="Default config with per-axis score row masking on eligible GPT "
        "decoder linear stacks.",
    ),
    ExperimentPreset.TOP_SLICE_AXIS_MASK: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {"row_mask_option": TopSliceAxisMaskConfig}
        ),
        description="Default config with top-slice axis row masking on eligible GPT "
        "decoder linear stacks.",
    ),
    ExperimentPreset.WEIGHT_INFORMED_SCORE_MASK: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {"row_mask_option": WeightInformedScoreAxisMaskConfig}
        ),
        description="Default config with weight-informed score row masking on eligible "
        "GPT decoder linear stacks.",
    ),
    (
        ExperimentPreset.SINGLE_MODEL_WEIGHT_ADDITIVE_BIAS_COMBINED_DIAGONAL
    ): PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "weight_option": DualModelDynamicWeightConfig,
                "bias_option": AdditiveDynamicBiasConfig,
                "diagonal_option": CombinedDynamicDiagonalConfig,
            }
        ),
        description="GPT-compatible dynamic weights, additive bias, and a combined "
        "dynamic diagonal for the single-model preset family.",
    ),
    (
        ExperimentPreset.DUAL_MODEL_WEIGHT_ADDITIVE_BIAS_COMBINED_DIAGONAL
    ): PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "weight_option": DualModelDynamicWeightConfig,
                "bias_option": AdditiveDynamicBiasConfig,
                "diagonal_option": CombinedDynamicDiagonalConfig,
            }
        ),
        description="Default config with dual-model dynamic weights, additive bias, "
        "and a combined dynamic diagonal on eligible GPT decoder stacks.",
    ),
    (
        ExperimentPreset.LAYERED_WEIGHTED_BANK_WEIGHT_ADDITIVE_BIAS_COMBINED_DIAGONAL
    ): PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "weight_option": LayeredWeightedBankDynamicWeightConfig,
                "bias_option": AdditiveDynamicBiasConfig,
                "diagonal_option": CombinedDynamicDiagonalConfig,
            }
        ),
        description="Default config with layered weighted-bank weights, additive bias, "
        "and a combined diagonal on eligible GPT decoder stacks.",
    ),
    ExperimentPreset.LOW_RANK_WEIGHT_ADDITIVE_BIAS_COMBINED_DIAGONAL: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "weight_option": LowRankDynamicWeightConfig,
                "bias_option": AdditiveDynamicBiasConfig,
                "diagonal_option": CombinedDynamicDiagonalConfig,
            }
        ),
        description="Default config with low-rank weights, additive bias, and a "
        "combined diagonal on eligible GPT decoder stacks.",
    ),
    (
        ExperimentPreset.SINGLE_MODEL_WEIGHT_ADDITIVE_BIAS_STANDARD_DIAGONAL
    ): PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "weight_option": DualModelDynamicWeightConfig,
                "bias_option": AdditiveDynamicBiasConfig,
                "diagonal_option": StandardDynamicDiagonalConfig,
            }
        ),
        description="GPT-compatible dynamic weights, additive bias, and a standard "
        "dynamic diagonal for the single-model preset family.",
    ),
    (
        ExperimentPreset.DUAL_MODEL_WEIGHT_ADDITIVE_BIAS_STANDARD_DIAGONAL
    ): PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "weight_option": DualModelDynamicWeightConfig,
                "bias_option": AdditiveDynamicBiasConfig,
                "diagonal_option": StandardDynamicDiagonalConfig,
            }
        ),
        description="Default config with dual-model dynamic weights, additive bias, "
        "and a standard dynamic diagonal on eligible GPT decoder stacks.",
    ),
    (
        ExperimentPreset.LAYERED_WEIGHTED_BANK_WEIGHT_ADDITIVE_BIAS_STANDARD_DIAGONAL
    ): PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "weight_option": LayeredWeightedBankDynamicWeightConfig,
                "bias_option": AdditiveDynamicBiasConfig,
                "diagonal_option": StandardDynamicDiagonalConfig,
            }
        ),
        description="Default config with layered weighted-bank weights, additive bias, "
        "and a standard diagonal on eligible GPT decoder stacks.",
    ),
    ExperimentPreset.LOW_RANK_WEIGHT_ADDITIVE_BIAS_STANDARD_DIAGONAL: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "weight_option": LowRankDynamicWeightConfig,
                "bias_option": AdditiveDynamicBiasConfig,
                "diagonal_option": StandardDynamicDiagonalConfig,
            }
        ),
        description="Default config with low-rank weights, additive bias, and a "
        "standard diagonal on eligible GPT decoder stacks.",
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
        description="Default config with dual-model dynamic weights decaying "
        "exponentially toward static GPT decoder linear layers.",
    ),
    ExperimentPreset.NORM_L2_WEIGHT: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "weight_option": DualModelDynamicWeightConfig,
                "weight_normalization_option": WeightNormalizationOptions.L2_SCALE,
            }
        ),
        description="Default config with dual-model dynamic weights and L2-scale "
        "normalization on eligible GPT decoder linear stacks.",
    ),
    ExperimentPreset.DEEP_GENERATOR: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "weight_option": DualModelDynamicWeightConfig,
                "generator_depth": DynamicDepthOptions.DEPTH_OF_EIGHT,
            }
        ),
        description="Default config with dual-model dynamic weights produced by a "
        "depth-8 generator network for GPT decoder layers.",
    ),
    ExperimentPreset.FULL_STACK: PresetDefinition(
        preset_values=_FULL_STACK_OVERRIDES,
        description="Default GPT token decoder with dynamic weights, additive bias, "
        "combined diagonal, and weight-informed row masking on eligible stacks.",
    ),
    ExperimentPreset.DUAL_WEIGHT_GATING: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "weight_option": DualModelDynamicWeightConfig,
                "stack_gate_flag": True,
            }
        ),
        description="Default GPT decoder with dual-model dynamic weights and "
        "per-layer gating enabled.",
    ),
    ExperimentPreset.DUAL_WEIGHT_HALTING: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "weight_option": DualModelDynamicWeightConfig,
                "stack_halting_flag": True,
            }
        ),
        description="Default GPT decoder with dual-model dynamic weights and stack "
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
        description="Default GPT decoder with dual-model dynamic weights, per-layer "
        "gating, and stack halting enabled.",
    ),
    ExperimentPreset.DUAL_WEIGHT_MEMORY: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "weight_option": DualModelDynamicWeightConfig,
                "memory_flag": True,
            }
        ),
        description="Default GPT decoder with dual-model dynamic weights and shared "
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
        description="Default GPT decoder with dual-model weights, per-layer gating, "
        "and shared memory enabled.",
    ),
    ExperimentPreset.DUAL_WEIGHT_HALTING_MEMORY: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "weight_option": DualModelDynamicWeightConfig,
                "stack_halting_flag": True,
                "memory_flag": True,
            }
        ),
        description="Default GPT decoder with dual-model weights, stack halting, and "
        "shared memory enabled.",
    ),
    ExperimentPreset.FULL_STACK_GATING: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {**_FULL_STACK_OVERRIDES, "stack_gate_flag": True}
        ),
        description="Full adaptive GPT decoder parameter controls with per-layer "
        "gating enabled.",
    ),
    ExperimentPreset.FULL_STACK_HALTING: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {**_FULL_STACK_OVERRIDES, "stack_halting_flag": True}
        ),
        description="Full adaptive GPT decoder parameter controls with stack halting "
        "enabled.",
    ),
    ExperimentPreset.FULL_STACK_MEMORY: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {**_FULL_STACK_OVERRIDES, "memory_flag": True}
        ),
        description="Full adaptive GPT decoder parameter controls with shared stack "
        "memory enabled.",
    ),
    ExperimentPreset.FULL_STACK_GATING_HALTING: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                **_FULL_STACK_OVERRIDES,
                "stack_gate_flag": True,
                "stack_halting_flag": True,
            }
        ),
        description="Full adaptive GPT decoder controls with per-layer gating and "
        "stack halting enabled.",
    ),
    ExperimentPreset.FULL_STACK_RECURRENT: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {**_FULL_STACK_OVERRIDES, "recurrent_flag": True}
        ),
        description="Full adaptive GPT decoder parameter controls wrapped in "
        "fixed-step recurrence.",
    ),
    ExperimentPreset.BANK_WEIGHT_MASK: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "weight_option": LayeredWeightedBankDynamicWeightConfig,
                "row_mask_option": WeightInformedScoreAxisMaskConfig,
            }
        ),
        description="Default GPT decoder with layered weighted-bank dynamic weights "
        "and weight-informed row masking.",
    ),
    ExperimentPreset.LOW_RANK_POST_NORM: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "weight_option": LowRankDynamicWeightConfig,
                "layer_norm_position": LayerNormPositionOptions.AFTER,
            }
        ),
        description="Default GPT decoder with low-rank dynamic weights and post-layer "
        "normalization enabled.",
    ),
}


class ExperimentPresets(BuilderBackedExperimentPresetsBase):
    def __init__(self) -> None:
        super().__init__(
            _PRESET_DEFINITIONS,
            builder_type=GptLinearAdaptiveConfigBuilder,
            default_preset=ExperimentPreset.BASELINE,
            default_dataset=WikiText2,
        )

    def _dataset_config(self, dataset: type) -> dict:
        return {
            **super()._dataset_config(dataset),
            "sequence_length": dataset.sequence_length,
        }

    def _preset(self, **kwargs):
        return self._builder_type(
            **linear_adaptive_builder_kwargs_from_flat(kwargs, config)
        ).build()


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

# ruff: noqa: E501

from dataclasses import replace

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
from emperor.embedding.absolute.core.config import (
    ImageSinusoidalPositionalEmbeddingConfig,
)
from emperor.experiments.base import (
    BuilderBackedExperimentPresetsBase,
    ExperimentBase,
    ExperimentPresetsBase,
    PresetDefinition,
)

import models.vit.linear_adaptive.config as config
import models.vit.linear_adaptive.dataset_options as dataset_options
from models.vit.linear_adaptive._builder_adapter import (
    linear_adaptive_builder_kwargs_from_flat,
)
from models.vit.linear_adaptive._config_defaults import vit_patch_options
from models.vit.linear_adaptive.config_builder import VitLinearAdaptiveConfigBuilder
from models.vit.linear_adaptive.model import Model


def default_patch_size_for_dataset(dataset: type) -> int:
    image_height = dataset.default_height
    if (
        image_height >= config.IMAGE_HEIGHT
        and image_height % config.IMAGE_PATCH_SIZE == 0
    ):
        return config.IMAGE_PATCH_SIZE
    for patch_size in (4, 2, 1):
        if image_height % patch_size == 0:
            return patch_size
    return 1


class ExperimentPreset(BaseOptions):
    BASELINE = 1
    POST_NORM = 2
    SINUSOIDAL = 3
    ATTENTION_BIAS = 4
    GATING = 5
    HALTING = 6
    MEMORY = 7
    GATING_HALTING = 8
    GATING_MEMORY = 9
    HALTING_MEMORY = 10
    GATING_HALTING_MEMORY = 11
    RESIDUAL = 12
    RESIDUAL_POST_NORM = 13
    RESIDUAL_GATING = 14
    RESIDUAL_HALTING = 15
    RESIDUAL_MEMORY = 16
    SINGLE_MODEL_WEIGHT = 17
    DUAL_MODEL_WEIGHT = 18
    LOW_RANK_WEIGHT = 19
    HYPERNETWORK_WEIGHT = 20
    LAYERED_WEIGHTED_BANK_WEIGHT = 21
    SOFT_WEIGHTED_BANK_WEIGHT = 22
    AFFINE_TRANSFORM_BIAS = 23
    ADDITIVE_BIAS = 24
    GENERATOR_BIAS = 25
    MULTIPLICATIVE_BIAS = 26
    SIGMOID_GATED_BIAS = 27
    TANH_GATED_BIAS = 28
    WEIGHTED_BANK_BIAS = 29
    STANDARD_DIAGONAL = 30
    ANTI_DIAGONAL = 31
    COMBINED_DIAGONAL = 32
    DIAGONAL_AXIS_MASK = 33
    OUTER_PRODUCT_MASK = 34
    PER_AXIS_SCORE_MASK = 35
    TOP_SLICE_AXIS_MASK = 36
    WEIGHT_INFORMED_SCORE_MASK = 37
    SINGLE_MODEL_WEIGHT_ADDITIVE_BIAS_COMBINED_DIAGONAL = 38
    DUAL_MODEL_WEIGHT_ADDITIVE_BIAS_COMBINED_DIAGONAL = 39
    LAYERED_WEIGHTED_BANK_WEIGHT_ADDITIVE_BIAS_COMBINED_DIAGONAL = 40
    LOW_RANK_WEIGHT_ADDITIVE_BIAS_COMBINED_DIAGONAL = 41
    SINGLE_MODEL_WEIGHT_ADDITIVE_BIAS_STANDARD_DIAGONAL = 42
    DUAL_MODEL_WEIGHT_ADDITIVE_BIAS_STANDARD_DIAGONAL = 43
    LAYERED_WEIGHTED_BANK_WEIGHT_ADDITIVE_BIAS_STANDARD_DIAGONAL = 44
    LOW_RANK_WEIGHT_ADDITIVE_BIAS_STANDARD_DIAGONAL = 45
    DECAY_EXPONENTIAL_WEIGHT = 46
    NORM_L2_WEIGHT = 47
    DEEP_GENERATOR = 48
    FULL_STACK = 49
    DUAL_WEIGHT_GATING = 50
    DUAL_WEIGHT_HALTING = 51
    DUAL_WEIGHT_GATING_HALTING = 52
    DUAL_WEIGHT_MEMORY = 53
    DUAL_WEIGHT_GATING_MEMORY = 54
    DUAL_WEIGHT_HALTING_MEMORY = 55
    FULL_STACK_GATING = 56
    FULL_STACK_HALTING = 57
    FULL_STACK_MEMORY = 58
    FULL_STACK_GATING_HALTING = 59
    FULL_STACK_RECURRENT = 60
    BANK_WEIGHT_MASK = 61
    LOW_RANK_POST_NORM = 62
    RECURRENT = 63
    RECURRENT_GATING = 64
    RECURRENT_HALTING = 65
    RECURRENT_MEMORY = 66
    RECURRENT_GATING_HALTING = 67
    RECURRENT_GATING_MEMORY = 68
    RECURRENT_HALTING_MEMORY = 69
    RECURRENT_GATING_HALTING_MEMORY = 70
    RECURRENT_RESIDUAL = 71
    RECURRENT_POST_NORM = 72


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
        description="Default config: a Vision Transformer classifier with adaptive "
        "linear patch embeddings, a trainable class token, learned image positions, "
        "and a pre-norm bidirectional encoder.",
    ),
    ExperimentPreset.POST_NORM: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "layer_norm_position": LayerNormPositionOptions.AFTER,
            }
        ),
        description="Default config with layer normalization after each encoder "
        "sub-block.",
    ),
    ExperimentPreset.SINUSOIDAL: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "positional_embedding_option": ImageSinusoidalPositionalEmbeddingConfig,
            }
        ),
        description="Default config with fixed sinusoidal image positional embeddings.",
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
        preset_values=_with_adaptive_option_flags(
            {
                "stack_gate_flag": True,
            }
        ),
        description="Default config with per-layer gating enabled on encoder blocks.",
    ),
    ExperimentPreset.HALTING: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "stack_halting_flag": True,
            }
        ),
        description="Default config with encoder stack halting enabled, so examples can "
        "stop early as they move through the encoder.",
    ),
    ExperimentPreset.MEMORY: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "memory_flag": True,
            }
        ),
        description="Default config with shared encoder stack memory enabled.",
    ),
    ExperimentPreset.GATING_HALTING: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "stack_gate_flag": True,
                "stack_halting_flag": True,
            }
        ),
        description="Default config with both per-layer gating and encoder stack "
        "halting enabled.",
    ),
    ExperimentPreset.GATING_MEMORY: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "stack_gate_flag": True,
                "memory_flag": True,
            }
        ),
        description="Default config with both per-layer gating and shared encoder stack "
        "memory enabled.",
    ),
    ExperimentPreset.HALTING_MEMORY: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "stack_halting_flag": True,
                "memory_flag": True,
            }
        ),
        description="Default config with both encoder stack halting and shared memory "
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
        description="Default config with per-layer gating, encoder stack halting, and "
        "shared memory enabled.",
    ),
    ExperimentPreset.RESIDUAL: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "stack_residual_connection_option": (
                    ResidualConnectionOptions.RESIDUAL
                ),
            }
        ),
        description="Default config with residual skip connections enabled between "
        "same-width encoder stack layers.",
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
        description="Default config with residual skip connections and per-layer gating "
        "enabled.",
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
        description="Default config with residual skip connections and encoder stack "
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
        description="Default config with residual skip connections and shared encoder "
        "stack memory enabled.",
    ),
    ExperimentPreset.SINGLE_MODEL_WEIGHT: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "weight_option": DualModelDynamicWeightConfig,
            }
        ),
        description="Default config with VIT-compatible dynamic weights for the "
        "single-model preset family on eligible encoder attention and feed-forward "
        "linear stacks.",
    ),
    ExperimentPreset.DUAL_MODEL_WEIGHT: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "weight_option": DualModelDynamicWeightConfig,
            }
        ),
        description="Default config with dual-model dynamic weights on eligible encoder "
        "attention and feed-forward linear stacks.",
    ),
    ExperimentPreset.LOW_RANK_WEIGHT: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "weight_option": LowRankDynamicWeightConfig,
            }
        ),
        description="Default config with adaptive low-rank dynamic weights on eligible "
        "encoder attention and feed-forward linear stacks.",
    ),
    ExperimentPreset.HYPERNETWORK_WEIGHT: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "weight_option": HypernetworkDynamicWeightConfig,
            }
        ),
        description="Default config with hypernetwork dynamic weights on eligible "
        "encoder attention and feed-forward linear stacks.",
    ),
    ExperimentPreset.LAYERED_WEIGHTED_BANK_WEIGHT: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "weight_option": LayeredWeightedBankDynamicWeightConfig,
            }
        ),
        description="Default config with layered weighted-bank dynamic weights on "
        "eligible encoder attention and feed-forward linear stacks.",
    ),
    ExperimentPreset.SOFT_WEIGHTED_BANK_WEIGHT: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "weight_option": SoftWeightedBankDynamicWeightConfig,
            }
        ),
        description="Default config with soft weighted-bank dynamic weights on eligible "
        "encoder attention and feed-forward linear stacks.",
    ),
    ExperimentPreset.AFFINE_TRANSFORM_BIAS: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "bias_option": AffineTransformDynamicBiasConfig,
            }
        ),
        description="Default config with affine-transform dynamic bias on eligible "
        "encoder attention and feed-forward linear stacks.",
    ),
    ExperimentPreset.ADDITIVE_BIAS: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "bias_option": AdditiveDynamicBiasConfig,
            }
        ),
        description="Default config with additive dynamic bias on eligible encoder "
        "attention and feed-forward linear stacks.",
    ),
    ExperimentPreset.GENERATOR_BIAS: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "bias_option": GeneratorDynamicBiasConfig,
            }
        ),
        description="Default config with generator-based dynamic bias on eligible "
        "encoder attention and feed-forward linear stacks.",
    ),
    ExperimentPreset.MULTIPLICATIVE_BIAS: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "bias_option": MultiplicativeDynamicBiasConfig,
            }
        ),
        description="Default config with multiplicative dynamic bias on eligible encoder "
        "attention and feed-forward linear stacks.",
    ),
    ExperimentPreset.SIGMOID_GATED_BIAS: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "bias_option": SigmoidGatedDynamicBiasConfig,
            }
        ),
        description="Default config with sigmoid-gated dynamic bias on eligible encoder "
        "attention and feed-forward linear stacks.",
    ),
    ExperimentPreset.TANH_GATED_BIAS: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "bias_option": TanhGatedDynamicBiasConfig,
            }
        ),
        description="Default config with tanh-gated dynamic bias on eligible encoder "
        "attention and feed-forward linear stacks.",
    ),
    ExperimentPreset.WEIGHTED_BANK_BIAS: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "bias_option": WeightedBankDynamicBiasConfig,
            }
        ),
        description="Default config with weighted-bank dynamic bias on eligible encoder "
        "attention and feed-forward linear stacks.",
    ),
    ExperimentPreset.STANDARD_DIAGONAL: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "diagonal_option": StandardDynamicDiagonalConfig,
            }
        ),
        description="Default config with standard dynamic diagonal on eligible encoder "
        "attention and feed-forward linear stacks.",
    ),
    ExperimentPreset.ANTI_DIAGONAL: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "diagonal_option": AntiDynamicDiagonalConfig,
            }
        ),
        description="Default config with anti dynamic diagonal on eligible encoder "
        "attention and feed-forward linear stacks.",
    ),
    ExperimentPreset.COMBINED_DIAGONAL: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "diagonal_option": CombinedDynamicDiagonalConfig,
            }
        ),
        description="Default config with combined dynamic diagonal on eligible encoder "
        "attention and feed-forward linear stacks.",
    ),
    ExperimentPreset.DIAGONAL_AXIS_MASK: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "row_mask_option": DiagonalAxisMaskConfig,
            }
        ),
        description="Default config with diagonal-axis row masking on eligible encoder "
        "attention and feed-forward linear stacks.",
    ),
    ExperimentPreset.OUTER_PRODUCT_MASK: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "row_mask_option": OuterProductMaskConfig,
            }
        ),
        description="Default config with outer-product row masking on eligible encoder "
        "attention and feed-forward linear stacks.",
    ),
    ExperimentPreset.PER_AXIS_SCORE_MASK: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "row_mask_option": PerAxisScoreMaskConfig,
            }
        ),
        description="Default config with per-axis score row masking on eligible encoder "
        "attention and feed-forward linear stacks.",
    ),
    ExperimentPreset.TOP_SLICE_AXIS_MASK: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "row_mask_option": TopSliceAxisMaskConfig,
            }
        ),
        description="Default config with top-slice axis row masking on eligible encoder "
        "attention and feed-forward linear stacks.",
    ),
    ExperimentPreset.WEIGHT_INFORMED_SCORE_MASK: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "row_mask_option": WeightInformedScoreAxisMaskConfig,
            }
        ),
        description="Default config with weight-informed score axis row masking on "
        "eligible encoder attention and feed-forward linear stacks.",
    ),
    ExperimentPreset.SINGLE_MODEL_WEIGHT_ADDITIVE_BIAS_COMBINED_DIAGONAL: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "weight_option": DualModelDynamicWeightConfig,
                "bias_option": AdditiveDynamicBiasConfig,
                "diagonal_option": CombinedDynamicDiagonalConfig,
            }
        ),
        description="Default config with VIT-compatible dynamic weights, additive bias, "
        "and combined dynamic diagonal for the single-model preset family on eligible "
        "encoder linear stacks.",
    ),
    ExperimentPreset.DUAL_MODEL_WEIGHT_ADDITIVE_BIAS_COMBINED_DIAGONAL: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "weight_option": DualModelDynamicWeightConfig,
                "bias_option": AdditiveDynamicBiasConfig,
                "diagonal_option": CombinedDynamicDiagonalConfig,
            }
        ),
        description="Default config with dual-model dynamic weights, additive bias, and "
        "combined dynamic diagonal on eligible encoder linear stacks.",
    ),
    ExperimentPreset.LAYERED_WEIGHTED_BANK_WEIGHT_ADDITIVE_BIAS_COMBINED_DIAGONAL: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "weight_option": LayeredWeightedBankDynamicWeightConfig,
                "bias_option": AdditiveDynamicBiasConfig,
                "diagonal_option": CombinedDynamicDiagonalConfig,
            }
        ),
        description="Default config with layered weighted-bank dynamic weights, additive "
        "bias, and combined dynamic diagonal on eligible encoder linear stacks.",
    ),
    ExperimentPreset.LOW_RANK_WEIGHT_ADDITIVE_BIAS_COMBINED_DIAGONAL: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "weight_option": LowRankDynamicWeightConfig,
                "bias_option": AdditiveDynamicBiasConfig,
                "diagonal_option": CombinedDynamicDiagonalConfig,
            }
        ),
        description="Default config with low-rank dynamic weights, additive bias, and "
        "combined dynamic diagonal on eligible encoder linear stacks.",
    ),
    ExperimentPreset.SINGLE_MODEL_WEIGHT_ADDITIVE_BIAS_STANDARD_DIAGONAL: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "weight_option": DualModelDynamicWeightConfig,
                "bias_option": AdditiveDynamicBiasConfig,
                "diagonal_option": StandardDynamicDiagonalConfig,
            }
        ),
        description="Default config with VIT-compatible dynamic weights, additive bias, "
        "and standard dynamic diagonal for the single-model preset family on eligible "
        "encoder linear stacks.",
    ),
    ExperimentPreset.DUAL_MODEL_WEIGHT_ADDITIVE_BIAS_STANDARD_DIAGONAL: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "weight_option": DualModelDynamicWeightConfig,
                "bias_option": AdditiveDynamicBiasConfig,
                "diagonal_option": StandardDynamicDiagonalConfig,
            }
        ),
        description="Default config with dual-model dynamic weights, additive bias, and "
        "standard dynamic diagonal on eligible encoder linear stacks.",
    ),
    ExperimentPreset.LAYERED_WEIGHTED_BANK_WEIGHT_ADDITIVE_BIAS_STANDARD_DIAGONAL: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "weight_option": LayeredWeightedBankDynamicWeightConfig,
                "bias_option": AdditiveDynamicBiasConfig,
                "diagonal_option": StandardDynamicDiagonalConfig,
            }
        ),
        description="Default config with layered weighted-bank dynamic weights, additive "
        "bias, and standard dynamic diagonal on eligible encoder linear stacks.",
    ),
    ExperimentPreset.LOW_RANK_WEIGHT_ADDITIVE_BIAS_STANDARD_DIAGONAL: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "weight_option": LowRankDynamicWeightConfig,
                "bias_option": AdditiveDynamicBiasConfig,
                "diagonal_option": StandardDynamicDiagonalConfig,
            }
        ),
        description="Default config with low-rank dynamic weights, additive bias, and "
        "standard dynamic diagonal on eligible encoder linear stacks.",
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
        "exponentially toward static encoder linear layers.",
    ),
    ExperimentPreset.NORM_L2_WEIGHT: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "weight_option": DualModelDynamicWeightConfig,
                "weight_normalization_option": WeightNormalizationOptions.L2_SCALE,
            }
        ),
        description="Default config with dual-model dynamic weights and L2-scale weight "
        "normalization on eligible encoder linear stacks.",
    ),
    ExperimentPreset.DEEP_GENERATOR: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "weight_option": DualModelDynamicWeightConfig,
                "generator_depth": DynamicDepthOptions.DEPTH_OF_EIGHT,
            }
        ),
        description="Default config with dual-model dynamic weights produced by a "
        "depth-8 generator network.",
    ),
    ExperimentPreset.FULL_STACK: PresetDefinition(
        preset_values=_FULL_STACK_OVERRIDES,
        description="Default config with dual-model dynamic weights, additive bias, "
        "combined dynamic diagonal, and weight-informed row masking on eligible "
        "encoder linear stacks.",
    ),
    ExperimentPreset.DUAL_WEIGHT_GATING: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "weight_option": DualModelDynamicWeightConfig,
                "stack_gate_flag": True,
            }
        ),
        description="Default config with dual-model dynamic weights and per-layer "
        "encoder gating enabled.",
    ),
    ExperimentPreset.DUAL_WEIGHT_HALTING: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "weight_option": DualModelDynamicWeightConfig,
                "stack_halting_flag": True,
            }
        ),
        description="Default config with dual-model dynamic weights and encoder stack "
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
        description="Default config with dual-model dynamic weights, per-layer encoder "
        "gating, and encoder stack halting enabled.",
    ),
    ExperimentPreset.DUAL_WEIGHT_MEMORY: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "weight_option": DualModelDynamicWeightConfig,
                "memory_flag": True,
            }
        ),
        description="Default config with dual-model dynamic weights and shared encoder "
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
        description="Default config with dual-model dynamic weights, per-layer encoder "
        "gating, and shared encoder stack memory enabled.",
    ),
    ExperimentPreset.DUAL_WEIGHT_HALTING_MEMORY: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "weight_option": DualModelDynamicWeightConfig,
                "stack_halting_flag": True,
                "memory_flag": True,
            }
        ),
        description="Default config with dual-model dynamic weights, encoder stack "
        "halting, and shared memory enabled.",
    ),
    ExperimentPreset.FULL_STACK_GATING: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                **_FULL_STACK_OVERRIDES,
                "stack_gate_flag": True,
            }
        ),
        description="Default config with full adaptive parameter controls and per-layer "
        "encoder gating enabled.",
    ),
    ExperimentPreset.FULL_STACK_HALTING: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                **_FULL_STACK_OVERRIDES,
                "stack_halting_flag": True,
            }
        ),
        description="Default config with full adaptive parameter controls and encoder "
        "stack halting enabled.",
    ),
    ExperimentPreset.FULL_STACK_MEMORY: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                **_FULL_STACK_OVERRIDES,
                "memory_flag": True,
            }
        ),
        description="Default config with full adaptive parameter controls and shared "
        "encoder stack memory enabled.",
    ),
    ExperimentPreset.FULL_STACK_GATING_HALTING: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                **_FULL_STACK_OVERRIDES,
                "stack_gate_flag": True,
                "stack_halting_flag": True,
            }
        ),
        description="Default config with full adaptive parameter controls, per-layer "
        "encoder gating, and encoder stack halting enabled.",
    ),
    ExperimentPreset.FULL_STACK_RECURRENT: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                **_FULL_STACK_OVERRIDES,
                "recurrent_flag": True,
            }
        ),
        description="Default config with full adaptive parameter controls wrapped in "
        "fixed-step encoder recurrence.",
    ),
    ExperimentPreset.BANK_WEIGHT_MASK: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "weight_option": LayeredWeightedBankDynamicWeightConfig,
                "row_mask_option": WeightInformedScoreAxisMaskConfig,
            }
        ),
        description="Default config with layered weighted-bank dynamic weights and "
        "weight-informed row masking on eligible encoder linear stacks.",
    ),
    ExperimentPreset.LOW_RANK_POST_NORM: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "weight_option": LowRankDynamicWeightConfig,
                "layer_norm_position": LayerNormPositionOptions.AFTER,
            }
        ),
        description="Default config with low-rank dynamic weights and post-layer "
        "normalization enabled.",
    ),
    ExperimentPreset.RECURRENT: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "recurrent_flag": True,
            }
        ),
        description="Default config wrapped in fixed-step recurrence, reusing the "
        "encoder stack for each recurrent step.",
    ),
    ExperimentPreset.RECURRENT_GATING: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "recurrent_flag": True,
                "recurrent_gate_flag": True,
            }
        ),
        description="Default recurrent config with step-level gating enabled after each "
        "recurrent encoder update.",
    ),
    ExperimentPreset.RECURRENT_HALTING: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "recurrent_flag": True,
                "recurrent_halting_flag": True,
            }
        ),
        description="Default recurrent config with recurrent halting enabled, allowing "
        "early stopping before the max step count.",
    ),
    ExperimentPreset.RECURRENT_MEMORY: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "recurrent_flag": True,
                "memory_flag": True,
            }
        ),
        description="Default recurrent config whose reused encoder stack has shared "
        "memory enabled.",
    ),
    ExperimentPreset.RECURRENT_GATING_HALTING: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "recurrent_flag": True,
                "recurrent_gate_flag": True,
                "recurrent_halting_flag": True,
            }
        ),
        description="Default recurrent config with both step-level gating and recurrent "
        "halting enabled.",
    ),
    ExperimentPreset.RECURRENT_GATING_MEMORY: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "recurrent_flag": True,
                "recurrent_gate_flag": True,
                "memory_flag": True,
            }
        ),
        description="Default recurrent config with step-level gating and shared memory "
        "in the reused encoder stack.",
    ),
    ExperimentPreset.RECURRENT_HALTING_MEMORY: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "recurrent_flag": True,
                "recurrent_halting_flag": True,
                "memory_flag": True,
            }
        ),
        description="Default recurrent config with recurrent halting and shared memory "
        "in the reused encoder stack.",
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
        description="Default recurrent config with step-level gating, recurrent halting, "
        "and shared memory in the reused encoder stack.",
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
        description="Default recurrent config using a residual encoder stack at each "
        "recurrent step.",
    ),
    ExperimentPreset.RECURRENT_POST_NORM: PresetDefinition(
        preset_values=_with_adaptive_option_flags(
            {
                "recurrent_flag": True,
                "layer_norm_position": LayerNormPositionOptions.AFTER,
            }
        ),
        description="Default recurrent config using a post-normalized encoder stack at "
        "each recurrent step.",
    ),
}


class ExperimentPresets(BuilderBackedExperimentPresetsBase):
    def __init__(self) -> None:
        super().__init__(
            _PRESET_DEFINITIONS,
            builder_type=VitLinearAdaptiveConfigBuilder,
            default_preset=ExperimentPreset.BASELINE,
        )

    def _dataset_config(self, dataset: type) -> dict:
        dataset_config = super()._dataset_config(dataset)
        patch_options = vit_patch_options(config)
        dataset_patch_size = default_patch_size_for_dataset(dataset)
        dataset_patch_options = replace(
            patch_options,
            patch_size=dataset_patch_size,
            input_channels=dataset.num_channels,
            image_height=dataset.default_height,
        )

        return {
            **dataset_config,
            "patch_options": dataset_patch_options,
            "output_dim": dataset.num_classes,
        }

    def _preset(self, **kwargs):
        builder_kwargs = linear_adaptive_builder_kwargs_from_flat(kwargs, config)
        return self._builder_type(**builder_kwargs).build()


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

import models.linears.linear_adaptive.config as config

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
from emperor.base.options import BaseOptions, LayerNormPositionOptions
from emperor.datasets.image.classification.mnist import Mnist
from emperor.experiments.base import (
    ExperimentBase,
    ExperimentPresetsBase,
    PresetLock,
    SearchMode,
)
from models.linears.linear_adaptive.config_builder import LinearAdaptiveConfigBuilder
from models.linears.linear_adaptive.model import Model

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class ExperimentPreset(BaseOptions):
    BASELINE = (
        "[BASELINE] Baseline adaptive linear stack preset; supports search-space flags."
    )
    SINGLE_MODEL_WEIGHT = (
        "[WEIGHT] Adaptive linear stack with the single-model dynamic weight generator."
    )
    DUAL_MODEL_WEIGHT = (
        "[WEIGHT] Adaptive linear stack with the dual-model dynamic weight generator."
    )
    LOW_RANK_WEIGHT = (
        "[WEIGHT] Adaptive linear stack with the low-rank dynamic weight generator."
    )
    HYPERNETWORK_WEIGHT = (
        "[WEIGHT] Adaptive linear stack with the hypernetwork dynamic weight generator."
    )
    LAYERED_WEIGHTED_BANK_WEIGHT = "[WEIGHT] Adaptive linear stack with the layered weighted bank dynamic weight generator."
    SOFT_WEIGHTED_BANK_WEIGHT = "[WEIGHT] Adaptive linear stack with the soft weighted bank dynamic weight generator."
    AFFINE_TRANSFORM_BIAS = (
        "[BIAS] Adaptive linear stack with the affine-transform dynamic bias generator."
    )
    ADDITIVE_BIAS = (
        "[BIAS] Adaptive linear stack with the additive dynamic bias generator."
    )
    GENERATOR_BIAS = (
        "[BIAS] Adaptive linear stack with the generator-based dynamic bias."
    )
    MULTIPLICATIVE_BIAS = (
        "[BIAS] Adaptive linear stack with the multiplicative dynamic bias generator."
    )
    SIGMOID_GATED_BIAS = (
        "[BIAS] Adaptive linear stack with the sigmoid-gated dynamic bias generator."
    )
    TANH_GATED_BIAS = (
        "[BIAS] Adaptive linear stack with the tanh-gated dynamic bias generator."
    )
    WEIGHTED_BANK_BIAS = (
        "[BIAS] Adaptive linear stack with the weighted bank dynamic bias generator."
    )
    STANDARD_DIAGONAL = (
        "[DIAGONAL] Adaptive linear stack with the standard dynamic diagonal generator."
    )
    ANTI_DIAGONAL = (
        "[DIAGONAL] Adaptive linear stack with the anti dynamic diagonal generator."
    )
    COMBINED_DIAGONAL = (
        "[DIAGONAL] Adaptive linear stack with the combined dynamic diagonal generator."
    )
    DIAGONAL_AXIS_MASK = (
        "[MASK] Adaptive linear stack with the diagonal-axis row mask generator."
    )
    OUTER_PRODUCT_MASK = (
        "[MASK] Adaptive linear stack with the outer-product row mask generator."
    )
    PER_AXIS_SCORE_MASK = (
        "[MASK] Adaptive linear stack with the per-axis score row mask generator."
    )
    TOP_SLICE_AXIS_MASK = (
        "[MASK] Adaptive linear stack with the top-slice axis row mask generator."
    )
    WEIGHT_INFORMED_SCORE_MASK = "[MASK] Adaptive linear stack with the weight-informed score axis row mask generator."
    SINGLE_MODEL_WEIGHT_ADDITIVE_BIAS_COMBINED_DIAGONAL = "[WEIGHT+BIAS+DIAGONAL] Single-model weight + additive bias + combined diagonal."
    DUAL_MODEL_WEIGHT_ADDITIVE_BIAS_COMBINED_DIAGONAL = (
        "[WEIGHT+BIAS+DIAGONAL] Dual-model weight + additive bias + combined diagonal."
    )
    LAYERED_WEIGHTED_BANK_WEIGHT_ADDITIVE_BIAS_COMBINED_DIAGONAL = "[WEIGHT+BIAS+DIAGONAL] Layered weighted bank weight + additive bias + combined diagonal."
    LOW_RANK_WEIGHT_ADDITIVE_BIAS_COMBINED_DIAGONAL = (
        "[WEIGHT+BIAS+DIAGONAL] Low-rank weight + additive bias + combined diagonal."
    )
    SINGLE_MODEL_WEIGHT_ADDITIVE_BIAS_STANDARD_DIAGONAL = "[WEIGHT+BIAS+DIAGONAL] Single-model weight + additive bias + standard diagonal."
    DUAL_MODEL_WEIGHT_ADDITIVE_BIAS_STANDARD_DIAGONAL = (
        "[WEIGHT+BIAS+DIAGONAL] Dual-model weight + additive bias + standard diagonal."
    )
    LAYERED_WEIGHTED_BANK_WEIGHT_ADDITIVE_BIAS_STANDARD_DIAGONAL = "[WEIGHT+BIAS+DIAGONAL] Layered weighted bank weight + additive bias + standard diagonal."
    LOW_RANK_WEIGHT_ADDITIVE_BIAS_STANDARD_DIAGONAL = (
        "[WEIGHT+BIAS+DIAGONAL] Low-rank weight + additive bias + standard diagonal."
    )
    DECAY_EXPONENTIAL_WEIGHT = "[DECAY] Dual-model weight that decays exponentially toward a static linear layer."
    NORM_L2_WEIGHT = "[NORM] Dual-model weight with L2-scale weight normalization."
    DEEP_GENERATOR = (
        "[CAPACITY] Dual-model weight produced by a depth-8 generator network."
    )
    FULL_STACK = "[WEIGHT+BIAS+DIAGONAL+MASK] Dual-model weight + additive bias + combined diagonal + weight-informed mask."
    ADAPTIVE_HALTING = (
        "[ADAPTIVE+ACT] Dual-model weight with adaptive computation halting enabled."
    )
    DUAL_WEIGHT_GATING = (
        "[WEIGHT+GATE] Dual-model dynamic weight with learned stack gating."
    )
    DUAL_WEIGHT_HALTING = (
        "[WEIGHT+HALT] Dual-model dynamic weight with adaptive computation halting."
    )
    FULL_STACK_GATING = (
        "[WEIGHT+BIAS+DIAGONAL+MASK+GATE] Full adaptive stack with learned gating."
    )
    FULL_STACK_RECURRENT = (
        "[WEIGHT+BIAS+DIAGONAL+MASK+RECURRENT] Full adaptive stack applied recurrently."
    )
    BANK_WEIGHT_MASK = (
        "[WEIGHT+MASK] Layered weighted-bank dynamic weight with weight-informed mask."
    )
    LOW_RANK_POST_NORM = (
        "[WEIGHT+NORM] Low-rank dynamic weight with post-layer normalization."
    )
    RECURRENT = "[RECURRENT] Adaptive linear stack applied recurrently for a fixed number of steps."
    RECURRENT_GATING = "[RECURRENT+GATE] Adaptive linear stack applied recurrently with a learned recurrent gate."
    RECURRENT_HALTING = "[RECURRENT+HALT] Adaptive linear stack applied recurrently with adaptive recurrent halting."
    RECURRENT_GATING_HALTING = "[RECURRENT+GATE+HALT] Adaptive linear stack applied recurrently with both learned recurrent gating and adaptive recurrent halting."


def _lock(preset, value, field: str) -> PresetLock:
    label = field.replace("_", " ")
    return PresetLock(
        value=value,
        reason=(
            f"Locked by the {preset.name} preset because this preset sets {label}."
        ),
    )


def _preset_locks(
    preset_overrides: dict["ExperimentPreset", dict[str, object]],
) -> dict["ExperimentPreset", dict[str, PresetLock]]:
    return {
        preset: {
            field: _lock(preset, value, field) for field, value in overrides.items()
        }
        for preset, overrides in preset_overrides.items()
        if overrides
    }


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

_PRESET_OVERRIDES = {
    ExperimentPreset.BASELINE: _with_adaptive_option_flags({}),
    ExperimentPreset.SINGLE_MODEL_WEIGHT: _with_adaptive_option_flags(
        {
            "weight_option": SingleModelDynamicWeightConfig,
        }
    ),
    ExperimentPreset.DUAL_MODEL_WEIGHT: _with_adaptive_option_flags(
        {
            "weight_option": DualModelDynamicWeightConfig,
        }
    ),
    ExperimentPreset.LOW_RANK_WEIGHT: _with_adaptive_option_flags(
        {
            "weight_option": LowRankDynamicWeightConfig,
        }
    ),
    ExperimentPreset.HYPERNETWORK_WEIGHT: _with_adaptive_option_flags(
        {
            "weight_option": HypernetworkDynamicWeightConfig,
        }
    ),
    ExperimentPreset.LAYERED_WEIGHTED_BANK_WEIGHT: _with_adaptive_option_flags(
        {
            "weight_option": LayeredWeightedBankDynamicWeightConfig,
        }
    ),
    ExperimentPreset.SOFT_WEIGHTED_BANK_WEIGHT: _with_adaptive_option_flags(
        {
            "weight_option": SoftWeightedBankDynamicWeightConfig,
        }
    ),
    ExperimentPreset.AFFINE_TRANSFORM_BIAS: _with_adaptive_option_flags(
        {
            "bias_option": AffineTransformDynamicBiasConfig,
        }
    ),
    ExperimentPreset.ADDITIVE_BIAS: _with_adaptive_option_flags(
        {
            "bias_option": AdditiveDynamicBiasConfig,
        }
    ),
    ExperimentPreset.GENERATOR_BIAS: _with_adaptive_option_flags(
        {
            "bias_option": GeneratorDynamicBiasConfig,
        }
    ),
    ExperimentPreset.MULTIPLICATIVE_BIAS: _with_adaptive_option_flags(
        {
            "bias_option": MultiplicativeDynamicBiasConfig,
        }
    ),
    ExperimentPreset.SIGMOID_GATED_BIAS: _with_adaptive_option_flags(
        {
            "bias_option": SigmoidGatedDynamicBiasConfig,
        }
    ),
    ExperimentPreset.TANH_GATED_BIAS: _with_adaptive_option_flags(
        {
            "bias_option": TanhGatedDynamicBiasConfig,
        }
    ),
    ExperimentPreset.WEIGHTED_BANK_BIAS: _with_adaptive_option_flags(
        {
            "bias_option": WeightedBankDynamicBiasConfig,
        }
    ),
    ExperimentPreset.STANDARD_DIAGONAL: _with_adaptive_option_flags(
        {
            "diagonal_option": StandardDynamicDiagonalConfig,
        }
    ),
    ExperimentPreset.ANTI_DIAGONAL: _with_adaptive_option_flags(
        {
            "diagonal_option": AntiDynamicDiagonalConfig,
        }
    ),
    ExperimentPreset.COMBINED_DIAGONAL: _with_adaptive_option_flags(
        {
            "diagonal_option": CombinedDynamicDiagonalConfig,
        }
    ),
    ExperimentPreset.DIAGONAL_AXIS_MASK: _with_adaptive_option_flags(
        {
            "row_mask_option": DiagonalAxisMaskConfig,
        }
    ),
    ExperimentPreset.OUTER_PRODUCT_MASK: _with_adaptive_option_flags(
        {
            "row_mask_option": OuterProductMaskConfig,
        }
    ),
    ExperimentPreset.PER_AXIS_SCORE_MASK: _with_adaptive_option_flags(
        {
            "row_mask_option": PerAxisScoreMaskConfig,
        }
    ),
    ExperimentPreset.TOP_SLICE_AXIS_MASK: _with_adaptive_option_flags(
        {
            "row_mask_option": TopSliceAxisMaskConfig,
        }
    ),
    ExperimentPreset.WEIGHT_INFORMED_SCORE_MASK: _with_adaptive_option_flags(
        {
            "row_mask_option": WeightInformedScoreAxisMaskConfig,
        }
    ),
    ExperimentPreset.SINGLE_MODEL_WEIGHT_ADDITIVE_BIAS_COMBINED_DIAGONAL: _with_adaptive_option_flags(
        {
            "weight_option": SingleModelDynamicWeightConfig,
            "bias_option": AdditiveDynamicBiasConfig,
            "diagonal_option": CombinedDynamicDiagonalConfig,
        }
    ),
    ExperimentPreset.DUAL_MODEL_WEIGHT_ADDITIVE_BIAS_COMBINED_DIAGONAL: _with_adaptive_option_flags(
        {
            "weight_option": DualModelDynamicWeightConfig,
            "bias_option": AdditiveDynamicBiasConfig,
            "diagonal_option": CombinedDynamicDiagonalConfig,
        }
    ),
    ExperimentPreset.LAYERED_WEIGHTED_BANK_WEIGHT_ADDITIVE_BIAS_COMBINED_DIAGONAL: _with_adaptive_option_flags(
        {
            "weight_option": LayeredWeightedBankDynamicWeightConfig,
            "bias_option": AdditiveDynamicBiasConfig,
            "diagonal_option": CombinedDynamicDiagonalConfig,
        }
    ),
    ExperimentPreset.LOW_RANK_WEIGHT_ADDITIVE_BIAS_COMBINED_DIAGONAL: _with_adaptive_option_flags(
        {
            "weight_option": LowRankDynamicWeightConfig,
            "bias_option": AdditiveDynamicBiasConfig,
            "diagonal_option": CombinedDynamicDiagonalConfig,
        }
    ),
    ExperimentPreset.SINGLE_MODEL_WEIGHT_ADDITIVE_BIAS_STANDARD_DIAGONAL: _with_adaptive_option_flags(
        {
            "weight_option": SingleModelDynamicWeightConfig,
            "bias_option": AdditiveDynamicBiasConfig,
            "diagonal_option": StandardDynamicDiagonalConfig,
        }
    ),
    ExperimentPreset.DUAL_MODEL_WEIGHT_ADDITIVE_BIAS_STANDARD_DIAGONAL: _with_adaptive_option_flags(
        {
            "weight_option": DualModelDynamicWeightConfig,
            "bias_option": AdditiveDynamicBiasConfig,
            "diagonal_option": StandardDynamicDiagonalConfig,
        }
    ),
    ExperimentPreset.LAYERED_WEIGHTED_BANK_WEIGHT_ADDITIVE_BIAS_STANDARD_DIAGONAL: _with_adaptive_option_flags(
        {
            "weight_option": LayeredWeightedBankDynamicWeightConfig,
            "bias_option": AdditiveDynamicBiasConfig,
            "diagonal_option": StandardDynamicDiagonalConfig,
        }
    ),
    ExperimentPreset.LOW_RANK_WEIGHT_ADDITIVE_BIAS_STANDARD_DIAGONAL: _with_adaptive_option_flags(
        {
            "weight_option": LowRankDynamicWeightConfig,
            "bias_option": AdditiveDynamicBiasConfig,
            "diagonal_option": StandardDynamicDiagonalConfig,
        }
    ),
    ExperimentPreset.DECAY_EXPONENTIAL_WEIGHT: _with_adaptive_option_flags(
        {
            "weight_option": DualModelDynamicWeightConfig,
            "weight_decay_schedule": WeightDecayScheduleOptions.EXPONENTIAL,
            "weight_decay_rate": 1e-3,
            "weight_decay_warmup_batches": 500,
        }
    ),
    ExperimentPreset.NORM_L2_WEIGHT: _with_adaptive_option_flags(
        {
            "weight_option": DualModelDynamicWeightConfig,
            "weight_normalization_option": WeightNormalizationOptions.L2_SCALE,
        }
    ),
    ExperimentPreset.DEEP_GENERATOR: _with_adaptive_option_flags(
        {
            "weight_option": DualModelDynamicWeightConfig,
            "generator_depth": DynamicDepthOptions.DEPTH_OF_EIGHT,
        }
    ),
    ExperimentPreset.FULL_STACK: _FULL_STACK_OVERRIDES,
    ExperimentPreset.ADAPTIVE_HALTING: _with_adaptive_option_flags(
        {
            "weight_option": DualModelDynamicWeightConfig,
            "stack_halting_flag": True,
        }
    ),
    ExperimentPreset.DUAL_WEIGHT_GATING: _with_adaptive_option_flags(
        {
            "weight_option": DualModelDynamicWeightConfig,
            "stack_gate_flag": True,
        }
    ),
    ExperimentPreset.DUAL_WEIGHT_HALTING: _with_adaptive_option_flags(
        {
            "weight_option": DualModelDynamicWeightConfig,
            "stack_halting_flag": True,
        }
    ),
    ExperimentPreset.FULL_STACK_GATING: _with_adaptive_option_flags(
        {
            **_FULL_STACK_OVERRIDES,
            "stack_gate_flag": True,
        }
    ),
    ExperimentPreset.FULL_STACK_RECURRENT: _with_adaptive_option_flags(
        {
            **_FULL_STACK_OVERRIDES,
            "recurrent_flag": True,
        }
    ),
    ExperimentPreset.BANK_WEIGHT_MASK: _with_adaptive_option_flags(
        {
            "weight_option": LayeredWeightedBankDynamicWeightConfig,
            "row_mask_option": WeightInformedScoreAxisMaskConfig,
        }
    ),
    ExperimentPreset.LOW_RANK_POST_NORM: _with_adaptive_option_flags(
        {
            "weight_option": LowRankDynamicWeightConfig,
            "layer_norm_position": LayerNormPositionOptions.AFTER,
        }
    ),
    ExperimentPreset.RECURRENT: _with_adaptive_option_flags(
        {
            "recurrent_flag": True,
        }
    ),
    ExperimentPreset.RECURRENT_GATING: _with_adaptive_option_flags(
        {
            "recurrent_flag": True,
            "recurrent_gate_flag": True,
        }
    ),
    ExperimentPreset.RECURRENT_HALTING: _with_adaptive_option_flags(
        {
            "recurrent_flag": True,
            "recurrent_halting_flag": True,
        }
    ),
    ExperimentPreset.RECURRENT_GATING_HALTING: _with_adaptive_option_flags(
        {
            "recurrent_flag": True,
            "recurrent_gate_flag": True,
            "recurrent_halting_flag": True,
        }
    ),
}


class ExperimentPresets(ExperimentPresetsBase):
    PRESET_OVERRIDES = _PRESET_OVERRIDES
    PRESET_LOCKS = _preset_locks(PRESET_OVERRIDES)

    def __init__(self) -> None:
        super().__init__()

    def get_config(
        self,
        model_config_preset: ExperimentPreset = ExperimentPreset.BASELINE,
        dataset: type = Mnist,
        search_mode: SearchMode = None,
        log_folder: str | None = None,
        search_keys: list[str] | None = None,
        config_overrides: dict | None = None,
        search_overrides: dict | None = None,
    ) -> list["ModelConfig"]:
        preset_callback = self._preset_callback_for_preset(model_config_preset)
        return self._create_preset_search_space_configs(
            dataset,
            search_mode,
            preset_callback,
            search_keys,
            config_overrides=config_overrides,
            search_overrides=search_overrides,
            model_config_preset=model_config_preset,
        )

    def _preset_callback_for_preset(self, preset: ExperimentPreset):
        if preset not in self.PRESET_OVERRIDES:
            raise ValueError(
                "The specified preset is not supported. Please choose a valid `ExperimentPreset`."
            )
        return lambda **kwargs: self._preset_for_preset(preset, **kwargs)

    def _preset_for_preset(
        self,
        preset: ExperimentPreset,
        **kwargs,
    ) -> "ModelConfig":
        preset_overrides = self.PRESET_OVERRIDES[preset]
        return self._preset(**{**kwargs, **preset_overrides})

    def _preset(self, **kwargs) -> "ModelConfig":
        return LinearAdaptiveConfigBuilder(**kwargs).build()


class Experiment(ExperimentBase):
    def __init__(
        self,
        experiment_preset: ExperimentPreset | None = None,
    ) -> None:
        super().__init__(experiment_preset)

    def _num_epochs(self) -> int:
        return config.NUM_EPOCHS

    def _dataset_options(self) -> list:
        return config.DATASET_OPTIONS

    def _model_type(self) -> type:
        return Model

    def _preset_generator_instance(self) -> ExperimentPresetsBase:
        return ExperimentPresets()

    def _experiment_preset_enum(self) -> type[BaseOptions]:
        return ExperimentPreset

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
from models.linears.linear_adaptive.model import Model

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.config import ModelConfig


class ExperimentOptions(BaseOptions):
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
    DEEP_GENERATOR = "[CAPACITY] Dual-model weight produced by a depth-8 generator network."
    FULL_STACK = "[WEIGHT+BIAS+DIAGONAL+MASK] Dual-model weight + additive bias + combined diagonal + weight-informed mask."
    ADAPTIVE_HALTING = "[ADAPTIVE+ACT] Dual-model weight with adaptive computation halting enabled."
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
    RECURRENT_GATING = (
        "[RECURRENT+GATE] Adaptive linear stack applied recurrently with a learned recurrent gate."
    )
    RECURRENT_HALTING = (
        "[RECURRENT+HALT] Adaptive linear stack applied recurrently with adaptive recurrent halting."
    )
    RECURRENT_GATING_HALTING = "[RECURRENT+GATE+HALT] Adaptive linear stack applied recurrently with both learned recurrent gating and adaptive recurrent halting."


def _lock(option, value, field: str) -> PresetLock:
    label = field.replace("_", " ")
    return PresetLock(
        value=value,
        reason=(
            f"Locked by the {option.name} preset because this preset sets "
            f"{label}."
        ),
    )


def _locks(option, **fields) -> dict[str, PresetLock]:
    return {key: _lock(option, value, key) for key, value in fields.items()}


class ExperimentPresets(ExperimentPresetsBase):
    PRESET_LOCKS = {
        ExperimentOptions.SINGLE_MODEL_WEIGHT: _locks(
            ExperimentOptions.SINGLE_MODEL_WEIGHT,
            weight_option=SingleModelDynamicWeightConfig,
        ),
        ExperimentOptions.DUAL_MODEL_WEIGHT: _locks(
            ExperimentOptions.DUAL_MODEL_WEIGHT,
            weight_option=DualModelDynamicWeightConfig,
        ),
        ExperimentOptions.LOW_RANK_WEIGHT: _locks(
            ExperimentOptions.LOW_RANK_WEIGHT,
            weight_option=LowRankDynamicWeightConfig,
        ),
        ExperimentOptions.HYPERNETWORK_WEIGHT: _locks(
            ExperimentOptions.HYPERNETWORK_WEIGHT,
            weight_option=HypernetworkDynamicWeightConfig,
        ),
        ExperimentOptions.LAYERED_WEIGHTED_BANK_WEIGHT: _locks(
            ExperimentOptions.LAYERED_WEIGHTED_BANK_WEIGHT,
            weight_option=LayeredWeightedBankDynamicWeightConfig,
        ),
        ExperimentOptions.SOFT_WEIGHTED_BANK_WEIGHT: _locks(
            ExperimentOptions.SOFT_WEIGHTED_BANK_WEIGHT,
            weight_option=SoftWeightedBankDynamicWeightConfig,
        ),
        ExperimentOptions.AFFINE_TRANSFORM_BIAS: _locks(
            ExperimentOptions.AFFINE_TRANSFORM_BIAS,
            bias_option=AffineTransformDynamicBiasConfig,
        ),
        ExperimentOptions.ADDITIVE_BIAS: _locks(
            ExperimentOptions.ADDITIVE_BIAS,
            bias_option=AdditiveDynamicBiasConfig,
        ),
        ExperimentOptions.GENERATOR_BIAS: _locks(
            ExperimentOptions.GENERATOR_BIAS,
            bias_option=GeneratorDynamicBiasConfig,
        ),
        ExperimentOptions.MULTIPLICATIVE_BIAS: _locks(
            ExperimentOptions.MULTIPLICATIVE_BIAS,
            bias_option=MultiplicativeDynamicBiasConfig,
        ),
        ExperimentOptions.SIGMOID_GATED_BIAS: _locks(
            ExperimentOptions.SIGMOID_GATED_BIAS,
            bias_option=SigmoidGatedDynamicBiasConfig,
        ),
        ExperimentOptions.TANH_GATED_BIAS: _locks(
            ExperimentOptions.TANH_GATED_BIAS,
            bias_option=TanhGatedDynamicBiasConfig,
        ),
        ExperimentOptions.WEIGHTED_BANK_BIAS: _locks(
            ExperimentOptions.WEIGHTED_BANK_BIAS,
            bias_option=WeightedBankDynamicBiasConfig,
        ),
        ExperimentOptions.STANDARD_DIAGONAL: _locks(
            ExperimentOptions.STANDARD_DIAGONAL,
            diagonal_option=StandardDynamicDiagonalConfig,
        ),
        ExperimentOptions.ANTI_DIAGONAL: _locks(
            ExperimentOptions.ANTI_DIAGONAL,
            diagonal_option=AntiDynamicDiagonalConfig,
        ),
        ExperimentOptions.COMBINED_DIAGONAL: _locks(
            ExperimentOptions.COMBINED_DIAGONAL,
            diagonal_option=CombinedDynamicDiagonalConfig,
        ),
        ExperimentOptions.DIAGONAL_AXIS_MASK: _locks(
            ExperimentOptions.DIAGONAL_AXIS_MASK,
            row_mask_option=DiagonalAxisMaskConfig,
        ),
        ExperimentOptions.OUTER_PRODUCT_MASK: _locks(
            ExperimentOptions.OUTER_PRODUCT_MASK,
            row_mask_option=OuterProductMaskConfig,
        ),
        ExperimentOptions.PER_AXIS_SCORE_MASK: _locks(
            ExperimentOptions.PER_AXIS_SCORE_MASK,
            row_mask_option=PerAxisScoreMaskConfig,
        ),
        ExperimentOptions.TOP_SLICE_AXIS_MASK: _locks(
            ExperimentOptions.TOP_SLICE_AXIS_MASK,
            row_mask_option=TopSliceAxisMaskConfig,
        ),
        ExperimentOptions.WEIGHT_INFORMED_SCORE_MASK: _locks(
            ExperimentOptions.WEIGHT_INFORMED_SCORE_MASK,
            row_mask_option=WeightInformedScoreAxisMaskConfig,
        ),
        ExperimentOptions.SINGLE_MODEL_WEIGHT_ADDITIVE_BIAS_COMBINED_DIAGONAL: _locks(
            ExperimentOptions.SINGLE_MODEL_WEIGHT_ADDITIVE_BIAS_COMBINED_DIAGONAL,
            weight_option=SingleModelDynamicWeightConfig,
            bias_option=AdditiveDynamicBiasConfig,
            diagonal_option=CombinedDynamicDiagonalConfig,
        ),
        ExperimentOptions.DUAL_MODEL_WEIGHT_ADDITIVE_BIAS_COMBINED_DIAGONAL: _locks(
            ExperimentOptions.DUAL_MODEL_WEIGHT_ADDITIVE_BIAS_COMBINED_DIAGONAL,
            weight_option=DualModelDynamicWeightConfig,
            bias_option=AdditiveDynamicBiasConfig,
            diagonal_option=CombinedDynamicDiagonalConfig,
        ),
        ExperimentOptions.LAYERED_WEIGHTED_BANK_WEIGHT_ADDITIVE_BIAS_COMBINED_DIAGONAL: _locks(
            ExperimentOptions.LAYERED_WEIGHTED_BANK_WEIGHT_ADDITIVE_BIAS_COMBINED_DIAGONAL,
            weight_option=LayeredWeightedBankDynamicWeightConfig,
            bias_option=AdditiveDynamicBiasConfig,
            diagonal_option=CombinedDynamicDiagonalConfig,
        ),
        ExperimentOptions.LOW_RANK_WEIGHT_ADDITIVE_BIAS_COMBINED_DIAGONAL: _locks(
            ExperimentOptions.LOW_RANK_WEIGHT_ADDITIVE_BIAS_COMBINED_DIAGONAL,
            weight_option=LowRankDynamicWeightConfig,
            bias_option=AdditiveDynamicBiasConfig,
            diagonal_option=CombinedDynamicDiagonalConfig,
        ),
        ExperimentOptions.SINGLE_MODEL_WEIGHT_ADDITIVE_BIAS_STANDARD_DIAGONAL: _locks(
            ExperimentOptions.SINGLE_MODEL_WEIGHT_ADDITIVE_BIAS_STANDARD_DIAGONAL,
            weight_option=SingleModelDynamicWeightConfig,
            bias_option=AdditiveDynamicBiasConfig,
            diagonal_option=StandardDynamicDiagonalConfig,
        ),
        ExperimentOptions.DUAL_MODEL_WEIGHT_ADDITIVE_BIAS_STANDARD_DIAGONAL: _locks(
            ExperimentOptions.DUAL_MODEL_WEIGHT_ADDITIVE_BIAS_STANDARD_DIAGONAL,
            weight_option=DualModelDynamicWeightConfig,
            bias_option=AdditiveDynamicBiasConfig,
            diagonal_option=StandardDynamicDiagonalConfig,
        ),
        ExperimentOptions.LAYERED_WEIGHTED_BANK_WEIGHT_ADDITIVE_BIAS_STANDARD_DIAGONAL: _locks(
            ExperimentOptions.LAYERED_WEIGHTED_BANK_WEIGHT_ADDITIVE_BIAS_STANDARD_DIAGONAL,
            weight_option=LayeredWeightedBankDynamicWeightConfig,
            bias_option=AdditiveDynamicBiasConfig,
            diagonal_option=StandardDynamicDiagonalConfig,
        ),
        ExperimentOptions.LOW_RANK_WEIGHT_ADDITIVE_BIAS_STANDARD_DIAGONAL: _locks(
            ExperimentOptions.LOW_RANK_WEIGHT_ADDITIVE_BIAS_STANDARD_DIAGONAL,
            weight_option=LowRankDynamicWeightConfig,
            bias_option=AdditiveDynamicBiasConfig,
            diagonal_option=StandardDynamicDiagonalConfig,
        ),
        ExperimentOptions.DECAY_EXPONENTIAL_WEIGHT: _locks(
            ExperimentOptions.DECAY_EXPONENTIAL_WEIGHT,
            weight_option=DualModelDynamicWeightConfig,
            weight_decay_schedule=WeightDecayScheduleOptions.EXPONENTIAL,
            weight_decay_rate=1e-3,
            weight_decay_warmup_batches=500,
        ),
        ExperimentOptions.NORM_L2_WEIGHT: _locks(
            ExperimentOptions.NORM_L2_WEIGHT,
            weight_option=DualModelDynamicWeightConfig,
            weight_normalization_option=WeightNormalizationOptions.L2_SCALE,
        ),
        ExperimentOptions.DEEP_GENERATOR: _locks(
            ExperimentOptions.DEEP_GENERATOR,
            weight_option=DualModelDynamicWeightConfig,
            generator_depth=DynamicDepthOptions.DEPTH_OF_EIGHT,
        ),
        ExperimentOptions.FULL_STACK: _locks(
            ExperimentOptions.FULL_STACK,
            weight_option=DualModelDynamicWeightConfig,
            bias_option=AdditiveDynamicBiasConfig,
            diagonal_option=CombinedDynamicDiagonalConfig,
            row_mask_option=WeightInformedScoreAxisMaskConfig,
        ),
        ExperimentOptions.ADAPTIVE_HALTING: _locks(
            ExperimentOptions.ADAPTIVE_HALTING,
            weight_option=DualModelDynamicWeightConfig,
            stack_halting_flag=True,
        ),
        ExperimentOptions.DUAL_WEIGHT_GATING: _locks(
            ExperimentOptions.DUAL_WEIGHT_GATING,
            weight_option=DualModelDynamicWeightConfig,
            stack_gate_flag=True,
        ),
        ExperimentOptions.DUAL_WEIGHT_HALTING: _locks(
            ExperimentOptions.DUAL_WEIGHT_HALTING,
            weight_option=DualModelDynamicWeightConfig,
            stack_halting_flag=True,
        ),
        ExperimentOptions.FULL_STACK_GATING: _locks(
            ExperimentOptions.FULL_STACK_GATING,
            weight_option=DualModelDynamicWeightConfig,
            bias_option=AdditiveDynamicBiasConfig,
            diagonal_option=CombinedDynamicDiagonalConfig,
            row_mask_option=WeightInformedScoreAxisMaskConfig,
            stack_gate_flag=True,
        ),
        ExperimentOptions.FULL_STACK_RECURRENT: _locks(
            ExperimentOptions.FULL_STACK_RECURRENT,
            weight_option=DualModelDynamicWeightConfig,
            bias_option=AdditiveDynamicBiasConfig,
            diagonal_option=CombinedDynamicDiagonalConfig,
            row_mask_option=WeightInformedScoreAxisMaskConfig,
            recurrent_flag=True,
        ),
        ExperimentOptions.BANK_WEIGHT_MASK: _locks(
            ExperimentOptions.BANK_WEIGHT_MASK,
            weight_option=LayeredWeightedBankDynamicWeightConfig,
            row_mask_option=WeightInformedScoreAxisMaskConfig,
        ),
        ExperimentOptions.LOW_RANK_POST_NORM: _locks(
            ExperimentOptions.LOW_RANK_POST_NORM,
            weight_option=LowRankDynamicWeightConfig,
            layer_norm_position=LayerNormPositionOptions.AFTER,
        ),
        ExperimentOptions.RECURRENT: _locks(
            ExperimentOptions.RECURRENT,
            recurrent_flag=True,
        ),
        ExperimentOptions.RECURRENT_GATING: _locks(
            ExperimentOptions.RECURRENT_GATING,
            recurrent_flag=True,
            recurrent_gate_flag=True,
        ),
        ExperimentOptions.RECURRENT_HALTING: _locks(
            ExperimentOptions.RECURRENT_HALTING,
            recurrent_flag=True,
            recurrent_halting_flag=True,
        ),
        ExperimentOptions.RECURRENT_GATING_HALTING: _locks(
            ExperimentOptions.RECURRENT_GATING_HALTING,
            recurrent_flag=True,
            recurrent_gate_flag=True,
            recurrent_halting_flag=True,
        ),
    }

    def __init__(self) -> None:
        super().__init__()

    def get_config(
        self,
        model_config_options: ExperimentOptions = ExperimentOptions.BASELINE,
        dataset: type = Mnist,
        search_mode: SearchMode = None,
        log_folder: str | None = None,
        search_keys: list[str] | None = None,
        config_overrides: dict | None = None,
        search_overrides: dict | None = None,
    ) -> list["ModelConfig"]:
        preset_callback = self._preset_callback_for_option(model_config_options)
        return self._create_preset_search_space_configs(
            dataset,
            search_mode,
            preset_callback,
            search_keys,
            config_overrides=config_overrides,
            search_overrides=search_overrides,
        )

    def _preset_callback_for_option(self, option: ExperimentOptions):
        callbacks = {
            ExperimentOptions.BASELINE: self._baseline_preset,
            ExperimentOptions.SINGLE_MODEL_WEIGHT: self._single_model_weight_preset,
            ExperimentOptions.DUAL_MODEL_WEIGHT: self._dual_model_weight_preset,
            ExperimentOptions.LOW_RANK_WEIGHT: self._low_rank_weight_preset,
            ExperimentOptions.HYPERNETWORK_WEIGHT: self._hypernetwork_weight_preset,
            ExperimentOptions.LAYERED_WEIGHTED_BANK_WEIGHT: self._layered_weighted_bank_weight_preset,
            ExperimentOptions.SOFT_WEIGHTED_BANK_WEIGHT: self._soft_weighted_bank_weight_preset,
            ExperimentOptions.AFFINE_TRANSFORM_BIAS: self._affine_transform_bias_preset,
            ExperimentOptions.ADDITIVE_BIAS: self._additive_bias_preset,
            ExperimentOptions.GENERATOR_BIAS: self._generator_bias_preset,
            ExperimentOptions.MULTIPLICATIVE_BIAS: self._multiplicative_bias_preset,
            ExperimentOptions.SIGMOID_GATED_BIAS: self._sigmoid_gated_bias_preset,
            ExperimentOptions.TANH_GATED_BIAS: self._tanh_gated_bias_preset,
            ExperimentOptions.WEIGHTED_BANK_BIAS: self._weighted_bank_bias_preset,
            ExperimentOptions.STANDARD_DIAGONAL: self._standard_diagonal_preset,
            ExperimentOptions.ANTI_DIAGONAL: self._anti_diagonal_preset,
            ExperimentOptions.COMBINED_DIAGONAL: self._combined_diagonal_preset,
            ExperimentOptions.DIAGONAL_AXIS_MASK: self._diagonal_axis_mask_preset,
            ExperimentOptions.OUTER_PRODUCT_MASK: self._outer_product_mask_preset,
            ExperimentOptions.PER_AXIS_SCORE_MASK: self._per_axis_score_mask_preset,
            ExperimentOptions.TOP_SLICE_AXIS_MASK: self._top_slice_axis_mask_preset,
            ExperimentOptions.WEIGHT_INFORMED_SCORE_MASK: self._weight_informed_score_mask_preset,
            ExperimentOptions.SINGLE_MODEL_WEIGHT_ADDITIVE_BIAS_COMBINED_DIAGONAL: self._single_model_weight_additive_bias_combined_diagonal_preset,
            ExperimentOptions.DUAL_MODEL_WEIGHT_ADDITIVE_BIAS_COMBINED_DIAGONAL: self._dual_model_weight_additive_bias_combined_diagonal_preset,
            ExperimentOptions.LAYERED_WEIGHTED_BANK_WEIGHT_ADDITIVE_BIAS_COMBINED_DIAGONAL: self._layered_weighted_bank_weight_additive_bias_combined_diagonal_preset,
            ExperimentOptions.LOW_RANK_WEIGHT_ADDITIVE_BIAS_COMBINED_DIAGONAL: self._low_rank_weight_additive_bias_combined_diagonal_preset,
            ExperimentOptions.SINGLE_MODEL_WEIGHT_ADDITIVE_BIAS_STANDARD_DIAGONAL: self._single_model_weight_additive_bias_standard_diagonal_preset,
            ExperimentOptions.DUAL_MODEL_WEIGHT_ADDITIVE_BIAS_STANDARD_DIAGONAL: self._dual_model_weight_additive_bias_standard_diagonal_preset,
            ExperimentOptions.LAYERED_WEIGHTED_BANK_WEIGHT_ADDITIVE_BIAS_STANDARD_DIAGONAL: self._layered_weighted_bank_weight_additive_bias_standard_diagonal_preset,
            ExperimentOptions.LOW_RANK_WEIGHT_ADDITIVE_BIAS_STANDARD_DIAGONAL: self._low_rank_weight_additive_bias_standard_diagonal_preset,
            ExperimentOptions.DECAY_EXPONENTIAL_WEIGHT: self._decay_exponential_weight_preset,
            ExperimentOptions.NORM_L2_WEIGHT: self._norm_l2_weight_preset,
            ExperimentOptions.DEEP_GENERATOR: self._deep_generator_preset,
            ExperimentOptions.FULL_STACK: self._full_stack_preset,
            ExperimentOptions.ADAPTIVE_HALTING: self._adaptive_halting_preset,
            ExperimentOptions.DUAL_WEIGHT_GATING: self._dual_weight_gating_preset,
            ExperimentOptions.DUAL_WEIGHT_HALTING: self._dual_weight_halting_preset,
            ExperimentOptions.FULL_STACK_GATING: self._full_stack_gating_preset,
            ExperimentOptions.FULL_STACK_RECURRENT: self._full_stack_recurrent_preset,
            ExperimentOptions.BANK_WEIGHT_MASK: self._bank_weight_mask_preset,
            ExperimentOptions.LOW_RANK_POST_NORM: self._low_rank_post_norm_preset,
            ExperimentOptions.RECURRENT: self._recurrent_preset,
            ExperimentOptions.RECURRENT_GATING: self._recurrent_gating_preset,
            ExperimentOptions.RECURRENT_HALTING: self._recurrent_halting_preset,
            ExperimentOptions.RECURRENT_GATING_HALTING: self._recurrent_gating_halting_preset,
        }
        if option not in callbacks:
            raise ValueError(
                "The specified option is not supported. Please choose a valid `ExperimentOptions`."
            )
        return callbacks[option]

    def _baseline_preset(self, **kwargs) -> "ModelConfig":
        return self._preset(**kwargs)

    def _single_model_weight_preset(self, **kwargs) -> "ModelConfig":
        return self._preset(**{"weight_option": SingleModelDynamicWeightConfig, **kwargs})

    def _dual_model_weight_preset(self, **kwargs) -> "ModelConfig":
        return self._preset(**{"weight_option": DualModelDynamicWeightConfig, **kwargs})

    def _low_rank_weight_preset(self, **kwargs) -> "ModelConfig":
        return self._preset(**{"weight_option": LowRankDynamicWeightConfig, **kwargs})

    def _hypernetwork_weight_preset(self, **kwargs) -> "ModelConfig":
        return self._preset(**{"weight_option": HypernetworkDynamicWeightConfig, **kwargs})

    def _layered_weighted_bank_weight_preset(self, **kwargs) -> "ModelConfig":
        return self._preset(
            **{"weight_option": LayeredWeightedBankDynamicWeightConfig, **kwargs}
        )

    def _soft_weighted_bank_weight_preset(self, **kwargs) -> "ModelConfig":
        return self._preset(**{"weight_option": SoftWeightedBankDynamicWeightConfig, **kwargs})

    def _affine_transform_bias_preset(self, **kwargs) -> "ModelConfig":
        return self._preset(**{"bias_option": AffineTransformDynamicBiasConfig, **kwargs})

    def _additive_bias_preset(self, **kwargs) -> "ModelConfig":
        return self._preset(**{"bias_option": AdditiveDynamicBiasConfig, **kwargs})

    def _generator_bias_preset(self, **kwargs) -> "ModelConfig":
        return self._preset(**{"bias_option": GeneratorDynamicBiasConfig, **kwargs})

    def _multiplicative_bias_preset(self, **kwargs) -> "ModelConfig":
        return self._preset(**{"bias_option": MultiplicativeDynamicBiasConfig, **kwargs})

    def _sigmoid_gated_bias_preset(self, **kwargs) -> "ModelConfig":
        return self._preset(**{"bias_option": SigmoidGatedDynamicBiasConfig, **kwargs})

    def _tanh_gated_bias_preset(self, **kwargs) -> "ModelConfig":
        return self._preset(**{"bias_option": TanhGatedDynamicBiasConfig, **kwargs})

    def _weighted_bank_bias_preset(self, **kwargs) -> "ModelConfig":
        return self._preset(**{"bias_option": WeightedBankDynamicBiasConfig, **kwargs})

    def _standard_diagonal_preset(self, **kwargs) -> "ModelConfig":
        return self._preset(**{"diagonal_option": StandardDynamicDiagonalConfig, **kwargs})

    def _anti_diagonal_preset(self, **kwargs) -> "ModelConfig":
        return self._preset(**{"diagonal_option": AntiDynamicDiagonalConfig, **kwargs})

    def _combined_diagonal_preset(self, **kwargs) -> "ModelConfig":
        return self._preset(**{"diagonal_option": CombinedDynamicDiagonalConfig, **kwargs})

    def _diagonal_axis_mask_preset(self, **kwargs) -> "ModelConfig":
        return self._preset(**{"row_mask_option": DiagonalAxisMaskConfig, **kwargs})

    def _outer_product_mask_preset(self, **kwargs) -> "ModelConfig":
        return self._preset(**{"row_mask_option": OuterProductMaskConfig, **kwargs})

    def _per_axis_score_mask_preset(self, **kwargs) -> "ModelConfig":
        return self._preset(**{"row_mask_option": PerAxisScoreMaskConfig, **kwargs})

    def _top_slice_axis_mask_preset(self, **kwargs) -> "ModelConfig":
        return self._preset(**{"row_mask_option": TopSliceAxisMaskConfig, **kwargs})

    def _weight_informed_score_mask_preset(self, **kwargs) -> "ModelConfig":
        return self._preset(**{"row_mask_option": WeightInformedScoreAxisMaskConfig, **kwargs})

    def _single_model_weight_additive_bias_combined_diagonal_preset(
        self, **kwargs
    ) -> "ModelConfig":
        return self._preset(
            **{
                "weight_option": SingleModelDynamicWeightConfig,
                "bias_option": AdditiveDynamicBiasConfig,
                "diagonal_option": CombinedDynamicDiagonalConfig,
                **kwargs,
            },
        )

    def _dual_model_weight_additive_bias_combined_diagonal_preset(
        self, **kwargs
    ) -> "ModelConfig":
        return self._preset(
            **{
                "weight_option": DualModelDynamicWeightConfig,
                "bias_option": AdditiveDynamicBiasConfig,
                "diagonal_option": CombinedDynamicDiagonalConfig,
                **kwargs,
            },
        )

    def _layered_weighted_bank_weight_additive_bias_combined_diagonal_preset(
        self, **kwargs
    ) -> "ModelConfig":
        return self._preset(
            **{
                "weight_option": LayeredWeightedBankDynamicWeightConfig,
                "bias_option": AdditiveDynamicBiasConfig,
                "diagonal_option": CombinedDynamicDiagonalConfig,
                **kwargs,
            },
        )

    def _low_rank_weight_additive_bias_combined_diagonal_preset(
        self, **kwargs
    ) -> "ModelConfig":
        return self._preset(
            **{
                "weight_option": LowRankDynamicWeightConfig,
                "bias_option": AdditiveDynamicBiasConfig,
                "diagonal_option": CombinedDynamicDiagonalConfig,
                **kwargs,
            },
        )

    def _single_model_weight_additive_bias_standard_diagonal_preset(
        self, **kwargs
    ) -> "ModelConfig":
        return self._preset(
            **{
                "weight_option": SingleModelDynamicWeightConfig,
                "bias_option": AdditiveDynamicBiasConfig,
                "diagonal_option": StandardDynamicDiagonalConfig,
                **kwargs,
            },
        )

    def _dual_model_weight_additive_bias_standard_diagonal_preset(
        self, **kwargs
    ) -> "ModelConfig":
        return self._preset(
            **{
                "weight_option": DualModelDynamicWeightConfig,
                "bias_option": AdditiveDynamicBiasConfig,
                "diagonal_option": StandardDynamicDiagonalConfig,
                **kwargs,
            },
        )

    def _layered_weighted_bank_weight_additive_bias_standard_diagonal_preset(
        self, **kwargs
    ) -> "ModelConfig":
        return self._preset(
            **{
                "weight_option": LayeredWeightedBankDynamicWeightConfig,
                "bias_option": AdditiveDynamicBiasConfig,
                "diagonal_option": StandardDynamicDiagonalConfig,
                **kwargs,
            },
        )

    def _low_rank_weight_additive_bias_standard_diagonal_preset(
        self, **kwargs
    ) -> "ModelConfig":
        return self._preset(
            **{
                "weight_option": LowRankDynamicWeightConfig,
                "bias_option": AdditiveDynamicBiasConfig,
                "diagonal_option": StandardDynamicDiagonalConfig,
                **kwargs,
            },
        )

    def _decay_exponential_weight_preset(self, **kwargs) -> "ModelConfig":
        return self._preset(
            **{
                "weight_option": DualModelDynamicWeightConfig,
                "weight_decay_schedule": WeightDecayScheduleOptions.EXPONENTIAL,
                "weight_decay_rate": 1e-3,
                "weight_decay_warmup_batches": 500,
                **kwargs,
            },
        )

    def _norm_l2_weight_preset(self, **kwargs) -> "ModelConfig":
        return self._preset(
            **{
                "weight_option": DualModelDynamicWeightConfig,
                "weight_normalization_option": WeightNormalizationOptions.L2_SCALE,
                **kwargs,
            },
        )

    def _deep_generator_preset(self, **kwargs) -> "ModelConfig":
        return self._preset(
            **{
                "weight_option": DualModelDynamicWeightConfig,
                "generator_depth": DynamicDepthOptions.DEPTH_OF_EIGHT,
                **kwargs,
            },
        )

    def _full_stack_kwargs(self) -> dict:
        return {
            "weight_option": DualModelDynamicWeightConfig,
            "bias_option": AdditiveDynamicBiasConfig,
            "diagonal_option": CombinedDynamicDiagonalConfig,
            "row_mask_option": WeightInformedScoreAxisMaskConfig,
        }

    def _full_stack_preset(self, **kwargs) -> "ModelConfig":
        return self._preset(**{**self._full_stack_kwargs(), **kwargs})

    def _adaptive_halting_preset(self, **kwargs) -> "ModelConfig":
        return self._preset(
            **{
                "weight_option": DualModelDynamicWeightConfig,
                "stack_halting_flag": True,
                **kwargs,
            },
        )

    def _dual_weight_gating_preset(self, **kwargs) -> "ModelConfig":
        return self._preset(
            **{
                "weight_option": DualModelDynamicWeightConfig,
                "stack_gate_flag": True,
                **kwargs,
            },
        )

    def _dual_weight_halting_preset(self, **kwargs) -> "ModelConfig":
        return self._preset(
            **{
                "weight_option": DualModelDynamicWeightConfig,
                "stack_halting_flag": True,
                **kwargs,
            },
        )

    def _full_stack_gating_preset(self, **kwargs) -> "ModelConfig":
        return self._preset(
            **{
                **self._full_stack_kwargs(),
                "stack_gate_flag": True,
                **kwargs,
            },
        )

    def _full_stack_recurrent_preset(self, **kwargs) -> "ModelConfig":
        return self._preset(
            **{
                **self._full_stack_kwargs(),
                "recurrent_flag": True,
                **kwargs,
            },
        )

    def _bank_weight_mask_preset(self, **kwargs) -> "ModelConfig":
        return self._preset(
            **{
                "weight_option": LayeredWeightedBankDynamicWeightConfig,
                "row_mask_option": WeightInformedScoreAxisMaskConfig,
                **kwargs,
            },
        )

    def _low_rank_post_norm_preset(self, **kwargs) -> "ModelConfig":
        return self._preset(
            **{
                "weight_option": LowRankDynamicWeightConfig,
                "layer_norm_position": LayerNormPositionOptions.AFTER,
                **kwargs,
            },
        )

    def _recurrent_preset(self, **kwargs) -> "ModelConfig":
        return self._preset(**{"recurrent_flag": True, **kwargs})

    def _recurrent_gating_preset(self, **kwargs) -> "ModelConfig":
        return self._preset(
            **{
                "recurrent_flag": True,
                "recurrent_gate_flag": True,
                **kwargs,
            },
        )

    def _recurrent_halting_preset(self, **kwargs) -> "ModelConfig":
        return self._preset(
            **{
                "recurrent_flag": True,
                "recurrent_halting_flag": True,
                **kwargs,
            },
        )

    def _recurrent_gating_halting_preset(self, **kwargs) -> "ModelConfig":
        return self._preset(
            **{
                "recurrent_flag": True,
                "recurrent_gate_flag": True,
                "recurrent_halting_flag": True,
                **kwargs,
            },
        )

    def _preset(self, **kwargs) -> "ModelConfig":
        from models.linears.linear_adaptive.config_builder import LinearAdaptiveConfigBuilder

        return LinearAdaptiveConfigBuilder(**kwargs).build()


class Experiment(ExperimentBase):
    def __init__(
        self,
        experiment_option: ExperimentOptions | None = None,
    ) -> None:
        super().__init__(experiment_option)

    def _num_epochs(self) -> int:
        return config.NUM_EPOCHS

    def _dataset_options(self) -> list:
        return config.DATASET_OPTIONS

    def _model_type(self) -> type:
        return Model

    def _preset_generator_instance(self) -> ExperimentPresetsBase:
        return ExperimentPresets()

    def _experiment_enumeration(self) -> type[BaseOptions]:
        return ExperimentOptions

import models.linear_adaptive.config as config

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
from emperor.base.options import BaseOptions
from emperor.datasets.image.classification.mnist import Mnist
from emperor.experiments.base import (
    ExperimentBase,
    ExperimentPresetsBase,
    SearchMode,
)
from models.linear_adaptive.model import Model

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
    COMBO_1 = "[WEIGHT+BIAS+DIAGONAL] Single-model weight + additive bias + combined diagonal."
    COMBO_2 = (
        "[WEIGHT+BIAS+DIAGONAL] Dual-model weight + additive bias + combined diagonal."
    )
    COMBO_3 = "[WEIGHT+BIAS+DIAGONAL] Layered weighted bank weight + additive bias + combined diagonal."
    COMBO_4 = (
        "[WEIGHT+BIAS+DIAGONAL] Low-rank weight + additive bias + combined diagonal."
    )
    COMBO_5 = "[WEIGHT+BIAS+DIAGONAL] Single-model weight + additive bias + standard diagonal."
    COMBO_6 = (
        "[WEIGHT+BIAS+DIAGONAL] Dual-model weight + additive bias + standard diagonal."
    )
    COMBO_7 = "[WEIGHT+BIAS+DIAGONAL] Layered weighted bank weight + additive bias + standard diagonal."
    COMBO_8 = (
        "[WEIGHT+BIAS+DIAGONAL] Low-rank weight + additive bias + standard diagonal."
    )
    DECAY_EXPONENTIAL_WEIGHT = "[DECAY] Dual-model weight that decays exponentially toward a static linear layer."
    NORM_L2_WEIGHT = "[NORM] Dual-model weight with L2-scale weight normalization."
    DEEP_GENERATOR = "[CAPACITY] Dual-model weight produced by a depth-8 generator network."


class ExperimentPresets(ExperimentPresetsBase):
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
            ExperimentOptions.COMBO_1: self._single_model_weight_additive_bias_combined_diagonal_preset,
            ExperimentOptions.COMBO_2: self._dual_model_weight_additive_bias_combined_diagonal_preset,
            ExperimentOptions.COMBO_3: self._layered_weighted_bank_weight_additive_bias_combined_diagonal_preset,
            ExperimentOptions.COMBO_4: self._low_rank_weight_additive_bias_combined_diagonal_preset,
            ExperimentOptions.COMBO_5: self._single_model_weight_additive_bias_standard_diagonal_preset,
            ExperimentOptions.COMBO_6: self._dual_model_weight_additive_bias_standard_diagonal_preset,
            ExperimentOptions.COMBO_7: self._layered_weighted_bank_weight_additive_bias_standard_diagonal_preset,
            ExperimentOptions.COMBO_8: self._low_rank_weight_additive_bias_standard_diagonal_preset,
            ExperimentOptions.DECAY_EXPONENTIAL_WEIGHT: self._decay_exponential_weight_preset,
            ExperimentOptions.NORM_L2_WEIGHT: self._norm_l2_weight_preset,
            ExperimentOptions.DEEP_GENERATOR: self._deep_generator_preset,
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

    def _preset(self, **kwargs) -> "ModelConfig":
        from models.linear_adaptive.config_builder import LinearAdaptiveConfigBuilder

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

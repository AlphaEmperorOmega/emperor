"""Public Interface for input-adaptive parameter configuration."""

from emperor.augmentations.adaptive_parameters._biases.config import (
    AdditiveDynamicBiasConfig,
    AffineTransformDynamicBiasConfig,
    DynamicBiasConfig,
    GeneratorDynamicBiasConfig,
    MultiplicativeDynamicBiasConfig,
    SigmoidGatedDynamicBiasConfig,
    TanhGatedDynamicBiasConfig,
    WeightedBankDynamicBiasConfig,
)
from emperor.augmentations.adaptive_parameters._config import (
    AdaptiveLinearLayerConfig,
    AdaptiveParameterAugmentationConfig,
)
from emperor.augmentations.adaptive_parameters._decay import DecayPolicy
from emperor.augmentations.adaptive_parameters._diagonals.config import (
    AntiDynamicDiagonalConfig,
    CombinedDynamicDiagonalConfig,
    DynamicDiagonalConfig,
    StandardDynamicDiagonalConfig,
)
from emperor.augmentations.adaptive_parameters._grouping.config import (
    AttentionGroupingConfig,
    GroupingConfig,
    MeanGroupingConfig,
    MeanStdGroupingConfig,
    RMSGroupingConfig,
    SumGroupingConfig,
)
from emperor.augmentations.adaptive_parameters._grouping.options import (
    SummaryNormalizationOptions,
)
from emperor.augmentations.adaptive_parameters._masks.config import (
    AxisMaskConfig,
    DiagonalAxisMaskConfig,
    OuterProductMaskConfig,
    PerAxisScoreMaskConfig,
    TopSliceAxisMaskConfig,
    WeightInformedScoreAxisMaskConfig,
)
from emperor.augmentations.adaptive_parameters._options import (
    AdaptiveParameterGroupingScopeOptions,
    AdaptiveParameterInputOrderOptions,
    BankExpansionFactorOptions,
    DynamicDepthOptions,
    MaskDimensionOptions,
    WeightDecayScheduleOptions,
    WeightNormalizationOptions,
    WeightNormalizationPositionOptions,
)
from emperor.augmentations.adaptive_parameters._weights.config import (
    DualModelDynamicWeightConfig,
    DynamicWeightConfig,
    HypernetworkDynamicWeightConfig,
    LayeredWeightedBankDynamicWeightConfig,
    LowRankDynamicWeightConfig,
    MatrixWeightsMixtureConfig,
    SingleModelDynamicWeightConfig,
    SoftWeightedBankDynamicWeightConfig,
)
from emperor.augmentations.adaptive_parameters._weights.normalization import (
    WeightNormalizationPolicy,
)
from emperor.augmentations.adaptive_parameters._weights.variants.matrix_mixture import (
    MatrixWeightsMixture,
)
from emperor.augmentations.adaptive_parameters.monitoring import (
    AdaptiveParameterMonitorCallback,
    WeightBankUtilizationMonitorCallback,
)

__all__ = (
    "DecayPolicy",
    "WeightNormalizationPolicy",
    "MatrixWeightsMixtureConfig",
    "MatrixWeightsMixture",
    "AttentionGroupingConfig",
    "GroupingConfig",
    "MeanGroupingConfig",
    "MeanStdGroupingConfig",
    "RMSGroupingConfig",
    "SumGroupingConfig",
    "SummaryNormalizationOptions",
    "AdaptiveParameterAugmentationConfig",
    "AdaptiveParameterInputOrderOptions",
    "AdaptiveLinearLayerConfig",
    "DynamicWeightConfig",
    "SingleModelDynamicWeightConfig",
    "DualModelDynamicWeightConfig",
    "LowRankDynamicWeightConfig",
    "HypernetworkDynamicWeightConfig",
    "LayeredWeightedBankDynamicWeightConfig",
    "SoftWeightedBankDynamicWeightConfig",
    "DynamicBiasConfig",
    "AffineTransformDynamicBiasConfig",
    "AdditiveDynamicBiasConfig",
    "MultiplicativeDynamicBiasConfig",
    "SigmoidGatedDynamicBiasConfig",
    "TanhGatedDynamicBiasConfig",
    "GeneratorDynamicBiasConfig",
    "WeightedBankDynamicBiasConfig",
    "DynamicDiagonalConfig",
    "StandardDynamicDiagonalConfig",
    "AntiDynamicDiagonalConfig",
    "CombinedDynamicDiagonalConfig",
    "AxisMaskConfig",
    "WeightInformedScoreAxisMaskConfig",
    "PerAxisScoreMaskConfig",
    "TopSliceAxisMaskConfig",
    "OuterProductMaskConfig",
    "DiagonalAxisMaskConfig",
    "BankExpansionFactorOptions",
    "DynamicDepthOptions",
    "MaskDimensionOptions",
    "WeightDecayScheduleOptions",
    "WeightNormalizationOptions",
    "WeightNormalizationPositionOptions",
    "AdaptiveParameterGroupingScopeOptions",
    "AdaptiveParameterMonitorCallback",
    "WeightBankUtilizationMonitorCallback",
)

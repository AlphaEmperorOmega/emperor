"""Public Interface for input-adaptive parameter generation."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from emperor.augmentations.adaptive_parameters._augmentation import (
        AdaptiveParameterAugmentation,
    )
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
    from emperor.augmentations.adaptive_parameters._diagonals.config import (
        AntiDynamicDiagonalConfig,
        CombinedDynamicDiagonalConfig,
        DynamicDiagonalConfig,
        StandardDynamicDiagonalConfig,
    )
    from emperor.augmentations.adaptive_parameters._linear_adapter import (
        AdaptiveLinearLayer,
    )
    from emperor.augmentations.adaptive_parameters._masks.config import (
        AxisMaskConfig,
        DiagonalAxisMaskConfig,
        OuterProductMaskConfig,
        PerAxisScoreMaskConfig,
        TopSliceAxisMaskConfig,
        WeightInformedScoreAxisMaskConfig,
    )
    from emperor.augmentations.adaptive_parameters._monitoring.weight_banks import (
        WeightBankUtilizationMonitorCallback,
    )
    from emperor.augmentations.adaptive_parameters._options import (
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
        SingleModelDynamicWeightConfig,
        SoftWeightedBankDynamicWeightConfig,
    )

    from ._monitoring.adaptive_parameters import (
        AdaptiveParameterMonitorCallback,
    )

__all__ = (
    "AdaptiveParameterAugmentation",
    "AdaptiveParameterAugmentationConfig",
    "AdaptiveLinearLayer",
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
    "AdaptiveParameterMonitorCallback",
    "WeightBankUtilizationMonitorCallback",
)

_LAZY_EXPORTS = {
    "AdaptiveParameterAugmentation": (
        "emperor.augmentations.adaptive_parameters._augmentation",
        "AdaptiveParameterAugmentation",
    ),
    "AdaptiveParameterAugmentationConfig": (
        "emperor.augmentations.adaptive_parameters._config",
        "AdaptiveParameterAugmentationConfig",
    ),
    "AdaptiveLinearLayer": (
        "emperor.augmentations.adaptive_parameters._linear_adapter",
        "AdaptiveLinearLayer",
    ),
    "AdaptiveLinearLayerConfig": (
        "emperor.augmentations.adaptive_parameters._config",
        "AdaptiveLinearLayerConfig",
    ),
    "DynamicWeightConfig": (
        "emperor.augmentations.adaptive_parameters._weights.config",
        "DynamicWeightConfig",
    ),
    "SingleModelDynamicWeightConfig": (
        "emperor.augmentations.adaptive_parameters._weights.config",
        "SingleModelDynamicWeightConfig",
    ),
    "DualModelDynamicWeightConfig": (
        "emperor.augmentations.adaptive_parameters._weights.config",
        "DualModelDynamicWeightConfig",
    ),
    "LowRankDynamicWeightConfig": (
        "emperor.augmentations.adaptive_parameters._weights.config",
        "LowRankDynamicWeightConfig",
    ),
    "HypernetworkDynamicWeightConfig": (
        "emperor.augmentations.adaptive_parameters._weights.config",
        "HypernetworkDynamicWeightConfig",
    ),
    "LayeredWeightedBankDynamicWeightConfig": (
        "emperor.augmentations.adaptive_parameters._weights.config",
        "LayeredWeightedBankDynamicWeightConfig",
    ),
    "SoftWeightedBankDynamicWeightConfig": (
        "emperor.augmentations.adaptive_parameters._weights.config",
        "SoftWeightedBankDynamicWeightConfig",
    ),
    "DynamicBiasConfig": (
        "emperor.augmentations.adaptive_parameters._biases.config",
        "DynamicBiasConfig",
    ),
    "AffineTransformDynamicBiasConfig": (
        "emperor.augmentations.adaptive_parameters._biases.config",
        "AffineTransformDynamicBiasConfig",
    ),
    "AdditiveDynamicBiasConfig": (
        "emperor.augmentations.adaptive_parameters._biases.config",
        "AdditiveDynamicBiasConfig",
    ),
    "MultiplicativeDynamicBiasConfig": (
        "emperor.augmentations.adaptive_parameters._biases.config",
        "MultiplicativeDynamicBiasConfig",
    ),
    "SigmoidGatedDynamicBiasConfig": (
        "emperor.augmentations.adaptive_parameters._biases.config",
        "SigmoidGatedDynamicBiasConfig",
    ),
    "TanhGatedDynamicBiasConfig": (
        "emperor.augmentations.adaptive_parameters._biases.config",
        "TanhGatedDynamicBiasConfig",
    ),
    "GeneratorDynamicBiasConfig": (
        "emperor.augmentations.adaptive_parameters._biases.config",
        "GeneratorDynamicBiasConfig",
    ),
    "WeightedBankDynamicBiasConfig": (
        "emperor.augmentations.adaptive_parameters._biases.config",
        "WeightedBankDynamicBiasConfig",
    ),
    "DynamicDiagonalConfig": (
        "emperor.augmentations.adaptive_parameters._diagonals.config",
        "DynamicDiagonalConfig",
    ),
    "StandardDynamicDiagonalConfig": (
        "emperor.augmentations.adaptive_parameters._diagonals.config",
        "StandardDynamicDiagonalConfig",
    ),
    "AntiDynamicDiagonalConfig": (
        "emperor.augmentations.adaptive_parameters._diagonals.config",
        "AntiDynamicDiagonalConfig",
    ),
    "CombinedDynamicDiagonalConfig": (
        "emperor.augmentations.adaptive_parameters._diagonals.config",
        "CombinedDynamicDiagonalConfig",
    ),
    "AxisMaskConfig": (
        "emperor.augmentations.adaptive_parameters._masks.config",
        "AxisMaskConfig",
    ),
    "WeightInformedScoreAxisMaskConfig": (
        "emperor.augmentations.adaptive_parameters._masks.config",
        "WeightInformedScoreAxisMaskConfig",
    ),
    "PerAxisScoreMaskConfig": (
        "emperor.augmentations.adaptive_parameters._masks.config",
        "PerAxisScoreMaskConfig",
    ),
    "TopSliceAxisMaskConfig": (
        "emperor.augmentations.adaptive_parameters._masks.config",
        "TopSliceAxisMaskConfig",
    ),
    "OuterProductMaskConfig": (
        "emperor.augmentations.adaptive_parameters._masks.config",
        "OuterProductMaskConfig",
    ),
    "DiagonalAxisMaskConfig": (
        "emperor.augmentations.adaptive_parameters._masks.config",
        "DiagonalAxisMaskConfig",
    ),
    "BankExpansionFactorOptions": (
        "emperor.augmentations.adaptive_parameters._options",
        "BankExpansionFactorOptions",
    ),
    "DynamicDepthOptions": (
        "emperor.augmentations.adaptive_parameters._options",
        "DynamicDepthOptions",
    ),
    "MaskDimensionOptions": (
        "emperor.augmentations.adaptive_parameters._options",
        "MaskDimensionOptions",
    ),
    "WeightDecayScheduleOptions": (
        "emperor.augmentations.adaptive_parameters._options",
        "WeightDecayScheduleOptions",
    ),
    "WeightNormalizationOptions": (
        "emperor.augmentations.adaptive_parameters._options",
        "WeightNormalizationOptions",
    ),
    "WeightNormalizationPositionOptions": (
        "emperor.augmentations.adaptive_parameters._options",
        "WeightNormalizationPositionOptions",
    ),
    "AdaptiveParameterMonitorCallback": (
        "emperor.augmentations.adaptive_parameters._monitoring.adaptive_parameters",
        "AdaptiveParameterMonitorCallback",
    ),
    "WeightBankUtilizationMonitorCallback": (
        "emperor.augmentations.adaptive_parameters._monitoring.weight_banks",
        "WeightBankUtilizationMonitorCallback",
    ),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as error:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message) from error

    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value

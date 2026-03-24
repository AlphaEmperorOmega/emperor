from emperor.base.utils import Module
from emperor.augmentations.adaptive_parameters.options import (
    DynamicBiasOptions,
    DynamicDiagonalOptions,
    DynamicWeightOptions,
    LinearMemoryOptions,
    RowMaskOptions,
)
from emperor.augmentations.adaptive_parameters.utils.handlers.mask import (
    MaskHandlerAbstract,
    PerRowMaskHandler,
    RowMaskHandler,
    TopSliceMaskHandler,
)
from emperor.augmentations.adaptive_parameters.utils.handlers.bias import (
    AffineBiasTransformHandler,
    BiasGeneratorHandler,
    BiasHandlerAbstract,
    ElementwiseBiasHandler,
    GatedBiasHandler,
)
from emperor.augmentations.adaptive_parameters.utils.handlers.diagonal import (
    AntiDiagonalHandler,
    DiagonalAndAntiDiagonalHandler,
    DiagonalHandler,
    DiagonalHandlerAbstract,
)
from emperor.augmentations.adaptive_parameters.utils.handlers.memory import (
    MemoryFusionHandler,
    MemoryHandlerAbstract,
    WeightedMemoryHandler,
)
from emperor.augmentations.adaptive_parameters.utils.handlers.weight import (
    DualModelWeightHandler,
    HypernetworkWeightHandler,
    LowRankWeightHandler,
    SingleModelWeightHandler,
    WeightHandlerAbstract,
    WeightMaskHandler,
)

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from emperor.augmentations.adaptive_parameters.config import AdaptiveParameterAugmentationConfig


class DynamicWeightFactory(Module):
    def __init__(
        self,
        cfg: "AdaptiveParameterAugmentationConfig",
        overrides: "AdaptiveParameterAugmentationConfig | None" = None,
    ):
        super().__init__()
        self.cfg: "AdaptiveParameterAugmentationConfig" = self._overwrite_config(
            cfg, overrides
        )
        self.weight_option = self.cfg.weight_option

    def build(self) -> WeightHandlerAbstract:
        match self.weight_option:
            case DynamicWeightOptions.DUAL_MODEL:
                return DualModelWeightHandler(self.cfg)
            case DynamicWeightOptions.SINGLE_MODEL:
                return SingleModelWeightHandler(self.cfg)
            case DynamicWeightOptions.LOW_RANK:
                return LowRankWeightHandler(self.cfg)
            case DynamicWeightOptions.WEIGHT_MASK:
                return WeightMaskHandler(self.cfg)
            case DynamicWeightOptions.HYPERNETWORK:
                return HypernetworkWeightHandler(self.cfg)
            case DynamicWeightOptions.DISABLED:
                raise ValueError(
                    "If the `weight_option` is set to `DISABLED`, this class should not be initialized"
                )


# TODO: Add option for a kernel to take the context
# of every token into account when computing the dynamic parameters
class DynamicDiagonalFactory(Module):
    def __init__(
        self,
        cfg: "AdaptiveParameterAugmentationConfig",
        overrides: "AdaptiveParameterAugmentationConfig | None" = None,
    ):
        super().__init__()
        config = getattr(cfg, "linear_layer_config", cfg)
        self.cfg: "AdaptiveParameterAugmentationConfig" = self._overwrite_config(
            config, overrides
        )
        self.diagonal_option = self.cfg.diagonal_option

    def build(self) -> DiagonalHandlerAbstract:
        match self.diagonal_option:
            case DynamicDiagonalOptions.DIAGONAL:
                return DiagonalHandler(self.cfg)
            case DynamicDiagonalOptions.ANTI_DIAGONAL:
                return AntiDiagonalHandler(self.cfg)
            case DynamicDiagonalOptions.DIAGONAL_AND_ANTI_DIAGONAL:
                return DiagonalAndAntiDiagonalHandler(self.cfg)
            case DynamicDiagonalOptions.DISABLED:
                raise ValueError(
                    "If the `diagonal_option` is set to `DISABLED`, this class should not be initialized"
                )


class DynamicBiasFactory(Module):
    def __init__(
        self,
        cfg: "AdaptiveParameterAugmentationConfig",
        overrides: "AdaptiveParameterAugmentationConfig | None" = None,
    ):
        super().__init__()
        self.cfg: "AdaptiveParameterAugmentationConfig" = self._overwrite_config(
            cfg, overrides
        )
        self.bias_option = self.cfg.bias_option

    def build(self) -> BiasHandlerAbstract:
        match self.bias_option:
            case DynamicBiasOptions.SCALE_AND_OFFSET:
                return AffineBiasTransformHandler(self.cfg)
            case DynamicBiasOptions.ELEMENT_WISE_OFFSET:
                return ElementwiseBiasHandler(self.cfg)
            case DynamicBiasOptions.DYNAMIC_PARAMETERS:
                return BiasGeneratorHandler(self.cfg)
            case DynamicBiasOptions.GATED:
                return GatedBiasHandler(self.cfg)
            case DynamicBiasOptions.DISABLED:
                raise ValueError(
                    "If the `bias_option` is set to `DISABLED`, this class should not be initialized"
                )


class DynamicMemoryFactory(Module):
    def __init__(
        self,
        cfg: "AdaptiveParameterAugmentationConfig",
        overrides: "AdaptiveParameterAugmentationConfig | None" = None,
    ):
        super().__init__()
        self.cfg: "AdaptiveParameterAugmentationConfig" = self._overwrite_config(
            cfg, overrides
        )
        self.memory_option = self.cfg.memory_option

    def build(self) -> MemoryHandlerAbstract:
        match self.memory_option:
            case LinearMemoryOptions.FUSION:
                return MemoryFusionHandler(self.cfg)
            case LinearMemoryOptions.WEIGHTED:
                return WeightedMemoryHandler(self.cfg)
            case LinearMemoryOptions.DISABLED:
                raise ValueError(
                    "If the `memory_option` is set to `DISABLED`, this class should not be initialized"
                )


class MaskHandlerFactory(Module):
    def __init__(
        self,
        cfg: "AdaptiveParameterAugmentationConfig",
        overrides: "AdaptiveParameterAugmentationConfig | None" = None,
    ):
        super().__init__()
        self.cfg: "AdaptiveParameterAugmentationConfig" = self._overwrite_config(
            cfg, overrides
        )
        self.row_mask_option = self.cfg.row_mask_option

    def build(self) -> MaskHandlerAbstract:
        match self.row_mask_option:
            case RowMaskOptions.GLOBAL_SCORE:
                return RowMaskHandler(self.cfg)
            case RowMaskOptions.PER_ROW_SCORE:
                return PerRowMaskHandler(self.cfg)
            case RowMaskOptions.TOP_SLICE:
                return TopSliceMaskHandler(self.cfg)
            case RowMaskOptions.DISABLED:
                raise ValueError(
                    "If the `row_mask_option` is set to `DISABLED`, this class should not be initialized"
                )

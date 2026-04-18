from emperor.base.utils import Module
from emperor.augmentations.adaptive_parameters.options import (
    DynamicDiagonalOptions,
    LinearMemoryOptions,
    RowMaskOptions,
)
from emperor.augmentations.adaptive_parameters.core.handlers.mask import (
    DiagonalMaskHandler,
    MaskHandlerAbstract,
    PerRowMaskHandler,
    RowMaskHandler,
    TopSliceMaskHandler,
)
from emperor.augmentations.adaptive_parameters.core.handlers.diagonal import (
    AntiDiagonalHandler,
    DiagonalAndAntiDiagonalHandler,
    DiagonalHandler,
    DiagonalHandlerAbstract,
)
from emperor.augmentations.adaptive_parameters.core.handlers.memory import (
    MemoryFusionHandler,
    MemoryHandlerAbstract,
    WeightedMemoryHandler,
)
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from emperor.augmentations.adaptive_parameters.config import AdaptiveParameterAugmentationConfig


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
        self.cfg: "AdaptiveParameterAugmentationConfig" = self._override_config(
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


class DynamicMemoryFactory(Module):
    def __init__(
        self,
        cfg: "AdaptiveParameterAugmentationConfig",
        overrides: "AdaptiveParameterAugmentationConfig | None" = None,
    ):
        super().__init__()
        self.cfg: "AdaptiveParameterAugmentationConfig" = self._override_config(
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
        self.cfg: "AdaptiveParameterAugmentationConfig" = self._override_config(
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
            case RowMaskOptions.DIAGONAL:
                return DiagonalMaskHandler(self.cfg)
            case RowMaskOptions.DISABLED:
                raise ValueError(
                    "If the `row_mask_option` is set to `DISABLED`, this class should not be initialized"
                )

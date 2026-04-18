from emperor.base.utils import Module
from emperor.augmentations.adaptive_parameters.options import (
    RowMaskOptions,
)
from emperor.augmentations.adaptive_parameters.core.handlers.mask import (
    DiagonalMaskHandler,
    MaskHandlerAbstract,
    PerRowMaskHandler,
    RowMaskHandler,
    TopSliceMaskHandler,
)
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from emperor.augmentations.adaptive_parameters.config import AdaptiveParameterAugmentationConfig


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

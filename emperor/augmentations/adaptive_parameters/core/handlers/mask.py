import torch

from typing import cast
from torch import Tensor
from dataclasses import dataclass, field
from emperor.base.utils import Module, ConfigBase
from emperor.base.layer import LayerStackConfig
from emperor.augmentations.adaptive_parameters.options import (
    MaskDimensionOptions,
    RowMaskOptions,
)


@dataclass
class MaskHandlerConfig(ConfigBase):
    input_dim: int | None = field(
        default=None,
        metadata={"help": "Input dimension of the mask transformation."},
    )
    output_dim: int | None = field(
        default=None,
        metadata={"help": "Output dimension of the mask transformation."},
    )
    row_mask_option: RowMaskOptions | None = field(
        default=None,
        metadata={
            "help": "Input-dependent row masking of the weight matrix after weight updates."
        },
    )
    mask_dimension_option: MaskDimensionOptions | None = field(
        default=None,
        metadata={"help": "Whether to mask rows or columns of the weight matrix."},
    )
    model_config: LayerStackConfig | None = field(
        default=None,
        metadata={"help": "Layer stack configuration for the internal generator network."},
    )

    def build(
        self, overrides: "ConfigBase | None" = None
    ) -> "MaskHandlerAbstract":
        if self.row_mask_option is None:
            raise ValueError("`row_mask_option` must be set before building the handler")
        handler_cls = MaskHandlerAbstract.resolve(self.row_mask_option)
        return handler_cls(self, cast("MaskHandlerConfig | None", overrides))


class MaskHandlerAbstract(Module):
    _registry: dict[RowMaskOptions, type["MaskHandlerAbstract"]] = {}

    @classmethod
    def register(cls, option: RowMaskOptions):
        def decorator(handler_cls: type["MaskHandlerAbstract"]):
            cls._registry[option] = handler_cls
            return handler_cls

        return decorator

    @classmethod
    def resolve(cls, option: RowMaskOptions) -> type["MaskHandlerAbstract"]:
        if option not in cls._registry:
            raise ValueError(f"No handler registered for mask option: {option}")
        return cls._registry[option]

    def __init__(
        self,
        cfg: MaskHandlerConfig,
        overrides: MaskHandlerConfig | None = None,
    ):
        super().__init__()
        self.cfg: MaskHandlerConfig = self._override_config(cfg, overrides)
        self.input_dim = self.cfg.input_dim
        self.mask_dimension_option = self.cfg.mask_dimension_option

    @property
    def _count_dim(self) -> int:
        return -1 if self.mask_dimension_option == MaskDimensionOptions.COLUMN else -2

    @property
    def _magnitude_dim(self) -> int:
        return -2 if self.mask_dimension_option == MaskDimensionOptions.COLUMN else -1

    @property
    def _broadcast_dim(self) -> int:
        return -2 if self.mask_dimension_option == MaskDimensionOptions.COLUMN else -1


@MaskHandlerAbstract.register(RowMaskOptions.GLOBAL_SCORE)
class RowMaskHandler(MaskHandlerAbstract):
    def __init__(
        self,
        cfg: MaskHandlerConfig,
        overrides: MaskHandlerConfig | None = None,
    ):
        super().__init__(cfg, overrides)
        self.score_generator = self.__init_score_generator()

    def __init_score_generator(self):
        from emperor.linears.core.stack import LinearLayerStack

        layer_overrides = LayerStackConfig(
            input_dim=self.input_dim,
            output_dim=1,
        )
        return LinearLayerStack(self.cfg.model_config, layer_overrides).build_model()

    def forward(
        self,
        weight_params: Tensor,
        logits: Tensor,
    ) -> Tensor:
        keep_fraction_logit = self.score_generator(logits)
        keep_fraction = torch.sigmoid(keep_fraction_logit)
        total_count = weight_params.shape[self._count_dim]
        items_to_keep = torch.clamp(
            (keep_fraction * total_count).long(), min=1
        ).squeeze(-1)
        magnitudes = weight_params.norm(dim=self._magnitude_dim)
        top_k_mask = self.__build_mask(magnitudes, items_to_keep)
        sparsified_weights = weight_params * top_k_mask.unsqueeze(self._broadcast_dim)
        if self.training:
            return sparsified_weights + (weight_params - weight_params.detach())
        return sparsified_weights

    def __build_mask(
        self,
        row_magnitudes: Tensor,
        rows_to_keep: Tensor,
    ) -> Tensor:
        sorted_magnitudes, _ = row_magnitudes.sort(dim=-1, descending=True)
        thresholds = sorted_magnitudes.gather(-1, (rows_to_keep - 1).unsqueeze(-1))
        return (row_magnitudes >= thresholds).float()


@MaskHandlerAbstract.register(RowMaskOptions.PER_ROW_SCORE)
class PerRowMaskHandler(MaskHandlerAbstract):
    def __init__(
        self,
        cfg: MaskHandlerConfig,
        overrides: MaskHandlerConfig | None = None,
    ):
        super().__init__(cfg, overrides)
        self.score_generator = self.__init_score_generator()

    def __init_score_generator(self):
        from emperor.linears.core.stack import LinearLayerStack

        layer_overrides = LayerStackConfig(
            input_dim=self.input_dim,
            output_dim=self.input_dim,
        )
        return LinearLayerStack(self.cfg.model_config, layer_overrides).build_model()

    def forward(
        self,
        weight_params: Tensor,
        logits: Tensor,
    ) -> Tensor:
        keep_fraction_logits = self.score_generator(logits)
        keep_fractions = torch.sigmoid(keep_fraction_logits)
        per_mask = (keep_fractions >= 0.5).float()
        sparsified_weights = weight_params * per_mask.unsqueeze(self._broadcast_dim)
        if self.training:
            return sparsified_weights + (weight_params - weight_params.detach())
        return sparsified_weights


@MaskHandlerAbstract.register(RowMaskOptions.TOP_SLICE)
class TopSliceMaskHandler(MaskHandlerAbstract):
    def __init__(
        self,
        cfg: MaskHandlerConfig,
        overrides: MaskHandlerConfig | None = None,
    ):
        super().__init__(cfg, overrides)
        self.score_generator = self.__init_score_generator()

    def __init_score_generator(self):
        from emperor.linears.core.stack import LinearLayerStack

        layer_overrides = LayerStackConfig(
            input_dim=self.input_dim,
            output_dim=1,
        )
        return LinearLayerStack(self.cfg.model_config, layer_overrides).build_model()

    def forward(
        self,
        weight_params: Tensor,
        logits: Tensor,
    ) -> Tensor:
        keep_fraction_logit = self.score_generator(logits)
        keep_fraction = torch.sigmoid(keep_fraction_logit)
        total_count = weight_params.shape[self._count_dim]
        items_to_keep = torch.clamp((keep_fraction * total_count).long(), min=1)
        indices = torch.arange(total_count, device=weight_params.device)
        top_k_mask = (indices.unsqueeze(0) < items_to_keep).float()
        sparsified_weights = weight_params * top_k_mask.unsqueeze(self._broadcast_dim)
        if self.training:
            return sparsified_weights + (weight_params - weight_params.detach())
        return sparsified_weights


@MaskHandlerAbstract.register(RowMaskOptions.DIAGONAL)
class DiagonalMaskHandler(MaskHandlerAbstract):
    def __init__(
        self,
        cfg: MaskHandlerConfig,
        overrides: MaskHandlerConfig | None = None,
    ):
        super().__init__(cfg, overrides)
        self.score_generator = self.__init_score_generator()

    def __init_score_generator(self):
        from emperor.linears.core.stack import LinearLayerStack

        layer_overrides = LayerStackConfig(
            input_dim=self.input_dim,
            output_dim=1,
        )
        return LinearLayerStack(self.cfg.model_config, layer_overrides).build_model()

    def forward(
        self,
        weight_params: Tensor,
        logits: Tensor,
    ) -> Tensor:
        keep_fraction_logit = self.score_generator(logits)
        keep_fraction = torch.sigmoid(keep_fraction_logit)
        row_count = weight_params.shape[-2]
        col_count = weight_params.shape[-1]
        min_diagonal_shift = 1 - row_count
        raw_diagonal_shift = (keep_fraction * (row_count + col_count)).long().squeeze(
            -1
        ) - row_count
        diagonal_shift = raw_diagonal_shift.clamp(min=min_diagonal_shift)
        row_indices = torch.arange(row_count, device=weight_params.device)
        col_indices = torch.arange(col_count, device=weight_params.device)
        diagonal_mask = (
            col_indices.unsqueeze(0)
            <= (row_count - 1 - row_indices).unsqueeze(1)
            + diagonal_shift[..., None, None]
        ).float()
        sparsified_weights = weight_params * diagonal_mask
        if self.training:
            return sparsified_weights + (weight_params - weight_params.detach())
        return sparsified_weights

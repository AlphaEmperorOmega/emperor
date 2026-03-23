import torch

from torch import Tensor
from emperor.base.utils import Module
from emperor.base.layer import LayerStackConfig

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.augmentations.adaptive_parameters.config import (
        AdaptiveParameterAugmentationConfig,
    )


class RowMaskHandler(Module):
    def __init__(
        self,
        cfg: "AdaptiveParameterAugmentationConfig",
        overrides: "AdaptiveParameterAugmentationConfig | None" = None,
    ):
        super().__init__()
        self.cfg: "AdaptiveParameterAugmentationConfig" = self._overwrite_config(
            cfg, overrides
        )
        self.input_dim = self.cfg.input_dim
        self.score_generator = self.__init_score_generator()

    def __init_score_generator(self):
        from emperor.linears.utils.stack import LinearLayerStack

        overrides = LayerStackConfig(
            input_dim=self.input_dim,
            output_dim=1,
        )
        return LinearLayerStack(self.cfg, overrides).build_model()

    def forward(
        self,
        weight_params: Tensor,
        logits: Tensor,
    ) -> Tensor:
        keep_fraction_logit = self.score_generator(logits)
        keep_fraction = torch.sigmoid(keep_fraction_logit)
        total_row_count = weight_params.shape[-2]
        rows_to_keep = torch.clamp(
            (keep_fraction * total_row_count).long(), min=1
        ).squeeze(-1)
        row_magnitudes = weight_params.norm(dim=-1)
        top_k_row_mask = self.__build_mask(row_magnitudes, rows_to_keep)
        sparsified_weights = weight_params * top_k_row_mask.unsqueeze(-1)
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


class PerRowMaskHandler(Module):
    def __init__(
        self,
        cfg: "AdaptiveParameterAugmentationConfig",
        overrides: "AdaptiveParameterAugmentationConfig | None" = None,
    ):
        super().__init__()
        self.cfg: "AdaptiveParameterAugmentationConfig" = self._overwrite_config(
            cfg, overrides
        )
        self.input_dim = self.cfg.input_dim
        self.score_generator = self.__init_score_generator()

    def __init_score_generator(self):
        from emperor.linears.utils.stack import LinearLayerStack

        overrides = LayerStackConfig(
            input_dim=self.input_dim,
            output_dim=self.input_dim,
        )
        main_cfg = self._resolve_main_config(self.cfg, self.cfg)
        return LinearLayerStack(main_cfg, overrides).build_model()

    def forward(
        self,
        weight_params: Tensor,
        logits: Tensor,
    ) -> Tensor:
        keep_fraction_logits = self.score_generator(logits)
        keep_fractions = torch.sigmoid(keep_fraction_logits)
        per_row_mask = (keep_fractions >= 0.5).float()
        sparsified_weights = weight_params * per_row_mask.unsqueeze(-1)
        if self.training:
            return sparsified_weights + (weight_params - weight_params.detach())
        return sparsified_weights


class TopSliceMaskHandler(Module):
    def __init__(
        self,
        cfg: "AdaptiveParameterAugmentationConfig",
        overrides: "AdaptiveParameterAugmentationConfig | None" = None,
    ):
        super().__init__()
        self.cfg: "AdaptiveParameterAugmentationConfig" = self._overwrite_config(
            cfg, overrides
        )
        self.input_dim = self.cfg.input_dim
        self.score_generator = self.__init_score_generator()

    def __init_score_generator(self):
        from emperor.linears.utils.stack import LinearLayerStack

        overrides = LayerStackConfig(
            input_dim=self.input_dim,
            output_dim=1,
        )
        main_cfg = self._resolve_main_config(self.cfg, self.cfg)
        return LinearLayerStack(main_cfg, overrides).build_model()

    def forward(
        self,
        weight_params: Tensor,
        logits: Tensor,
    ) -> Tensor:
        keep_fraction_logit = self.score_generator(logits)
        keep_fraction = torch.sigmoid(keep_fraction_logit)
        total_row_count = weight_params.shape[-2]
        rows_to_keep = torch.clamp((keep_fraction * total_row_count).long(), min=1)
        row_indices = torch.arange(total_row_count, device=weight_params.device)
        top_k_row_mask = (row_indices.unsqueeze(0) < rows_to_keep).float()
        sparsified_weights = weight_params * top_k_row_mask.unsqueeze(-1)
        if self.training:
            return sparsified_weights + (weight_params - weight_params.detach())
        return sparsified_weights

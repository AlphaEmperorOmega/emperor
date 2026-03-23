import torch

from torch import Tensor
from emperor.base.utils import Module
from emperor.base.layer import LayerStackConfig

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.augmentations.adaptive_parameters.model import AdaptiveParameterBehaviourConfig


class RowMaskHandler(Module):
    def __init__(
        self,
        cfg: "AdaptiveParameterBehaviourConfig",
        overrides: "AdaptiveParameterBehaviourConfig | None" = None,
    ):
        super().__init__()
        self.cfg: "AdaptiveParameterBehaviourConfig" = self._overwrite_config(
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
        score = torch.sigmoid(self.score_generator(logits))
        num_rows = weight_params.shape[-2]
        k_per_sample = torch.clamp((score * num_rows).long(), min=1).squeeze(-1)
        row_norms = weight_params.norm(dim=-1)
        mask = self.__build_mask(row_norms, k_per_sample)
        masked = weight_params * mask.unsqueeze(-1)
        if self.training:
            return masked + (weight_params - weight_params.detach())
        return masked

    def __build_mask(
        self,
        row_norms: Tensor,
        k_per_sample: Tensor,
    ) -> Tensor:
        sorted_norms, _ = row_norms.sort(dim=-1, descending=True)
        thresholds = sorted_norms.gather(
            -1, (k_per_sample - 1).unsqueeze(-1)
        )
        return (row_norms >= thresholds).float()


class PerRowMaskHandler(Module):
    def __init__(
        self,
        cfg: "AdaptiveParameterBehaviourConfig",
        overrides: "AdaptiveParameterBehaviourConfig | None" = None,
    ):
        super().__init__()
        self.cfg: "AdaptiveParameterBehaviourConfig" = self._overwrite_config(
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
        scores = torch.sigmoid(self.score_generator(logits))
        hard_mask = (scores >= 0.5).float()
        masked = weight_params * hard_mask.unsqueeze(-1)
        if self.training:
            return masked + (weight_params - weight_params.detach())
        return masked


class TopSliceMaskHandler(Module):
    def __init__(
        self,
        cfg: "AdaptiveParameterBehaviourConfig",
        overrides: "AdaptiveParameterBehaviourConfig | None" = None,
    ):
        super().__init__()
        self.cfg: "AdaptiveParameterBehaviourConfig" = self._overwrite_config(
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
        score = torch.sigmoid(self.score_generator(logits))
        num_rows = weight_params.shape[-2]
        k = torch.clamp((score * num_rows).long(), min=1)
        row_indices = torch.arange(num_rows, device=weight_params.device)
        mask = (row_indices.unsqueeze(0) < k).float()
        masked = weight_params * mask.unsqueeze(-1)
        if self.training:
            return masked + (weight_params - weight_params.detach())
        return masked

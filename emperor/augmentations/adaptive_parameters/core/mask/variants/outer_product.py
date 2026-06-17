import torch

from torch import Tensor
from emperor.base.layer import Layer
from emperor.augmentations.adaptive_parameters.core.mask.base import AxisMaskAbstract
from emperor.augmentations.adaptive_parameters.core.mask.config import (
    OuterProductMaskConfig,
)


class OuterProductMask(AxisMaskAbstract):
    def __init__(
        self,
        cfg: OuterProductMaskConfig,
        overrides: OuterProductMaskConfig | None = None,
    ):
        super().__init__(cfg, overrides)
        self.input_model = self._init_model(self.input_dim)
        self.output_model = self._init_model(self.output_dim)

    def forward(
        self,
        weight_params: Tensor,
        logits: Tensor,
    ) -> Tensor:
        input_vectors = Layer.run_model_returning_hidden(self.input_model, logits)
        output_vectors = Layer.run_model_returning_hidden(self.output_model, logits)
        outer_product = torch.einsum("bi,bj->bij", input_vectors, output_vectors)
        scores = torch.sigmoid(outer_product)
        hard_mask = self._compute_hard_mask(scores)
        soft_mask = self._compute_soft_mask(scores)
        return self._apply_hybrid_mask(weight_params, hard_mask, soft_mask)

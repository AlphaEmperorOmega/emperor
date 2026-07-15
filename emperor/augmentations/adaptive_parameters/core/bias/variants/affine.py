from torch import Tensor

from emperor.augmentations.adaptive_parameters.core._validator import (
    DynamicBiasValidator,
)
from emperor.augmentations.adaptive_parameters.core.bias.base import DynamicBiasAbstract
from emperor.augmentations.adaptive_parameters.core.bias.config import (
    AffineTransformDynamicBiasConfig,
)
from emperor.base.layer import Layer


class AffineTransformDynamicBias(DynamicBiasAbstract):
    def __init__(
        self,
        cfg: AffineTransformDynamicBiasConfig,
        overrides: AffineTransformDynamicBiasConfig | None = None,
    ):
        super().__init__(cfg, overrides)
        affine_parameter_dim = 2
        self.model = self._init_model(affine_parameter_dim)

    def forward(self, bias_params: Tensor, logits: Tensor) -> Tensor:
        DynamicBiasValidator.ensure_parameters_exist(bias_params)
        affine_parameters = Layer.run_model_returning_hidden(self.model, logits)
        bias_scale, bias_offset = affine_parameters.chunk(2, dim=-1)
        return bias_scale * bias_params + bias_offset

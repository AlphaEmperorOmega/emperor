from torch import Tensor

from emperor.augmentations.adaptive_parameters.core._validator import (
    DynamicBiasValidator,
)
from emperor.augmentations.adaptive_parameters.core.bias.base import DynamicBiasAbstract
from emperor.augmentations.adaptive_parameters.core.bias.config import (
    MultiplicativeDynamicBiasConfig,
)
from emperor.base.layer import Layer


class MultiplicativeDynamicBias(DynamicBiasAbstract):
    def __init__(
        self,
        cfg: MultiplicativeDynamicBiasConfig,
        overrides: MultiplicativeDynamicBiasConfig | None = None,
    ):
        super().__init__(cfg, overrides)
        self.model = self._init_model(self.output_dim)

    def forward(self, bias_params: Tensor, logits: Tensor) -> Tensor:
        DynamicBiasValidator.ensure_parameters_exist(bias_params)
        bias_scale = Layer.run_model_returning_hidden(self.model, logits)
        return bias_params * bias_scale

from torch import Tensor

from emperor.augmentations.adaptive_parameters.core.bias.base import DynamicBiasAbstract
from emperor.augmentations.adaptive_parameters.core.bias.config import (
    AdditiveDynamicBiasConfig,
)
from emperor.base.layer import Layer


class AdditiveDynamicBias(DynamicBiasAbstract):
    def __init__(
        self,
        cfg: AdditiveDynamicBiasConfig,
        overrides: AdditiveDynamicBiasConfig | None = None,
    ):
        super().__init__(cfg, overrides)
        self.model = self._init_model(self.output_dim)

    def forward(self, bias_params: Tensor, logits: Tensor) -> Tensor:
        self.VALIDATOR.ensure_parameters_exist(bias_params)
        bias_params = self._maybe_apply_bias_decay(bias_params)
        generated_bias_offset = Layer.run_model_returning_hidden(self.model, logits)
        return bias_params + generated_bias_offset

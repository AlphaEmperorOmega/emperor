from torch import Tensor

from emperor.augmentations.adaptive_parameters._biases.base import DynamicBiasAbstract
from emperor.augmentations.adaptive_parameters._biases.config import (
    MultiplicativeDynamicBiasConfig,
)
from emperor.layers import Layer


class MultiplicativeDynamicBias(DynamicBiasAbstract):
    def __init__(
        self,
        cfg: MultiplicativeDynamicBiasConfig,
        overrides: MultiplicativeDynamicBiasConfig | None = None,
    ):
        super().__init__(cfg, overrides)
        self.model = self._init_model(self.output_dim)

    def forward(self, bias_params: Tensor, logits: Tensor) -> Tensor:
        self.VALIDATOR.ensure_parameters_exist(bias_params)
        bias_scale_state = Layer.run_model_from_hidden(self.model, logits)
        bias_scale = bias_scale_state.hidden
        return bias_params * bias_scale

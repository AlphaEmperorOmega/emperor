from torch import Tensor
from emperor.base.layer import Layer
from emperor.augmentations.adaptive_parameters.core.bias.base import DynamicBiasAbstract
from emperor.augmentations.adaptive_parameters.core.bias.config import (
    GeneratorDynamicBiasConfig,
)


class GeneratorDynamicBias(DynamicBiasAbstract):
    def __init__(
        self,
        cfg: GeneratorDynamicBiasConfig,
        overrides: GeneratorDynamicBiasConfig | None = None,
    ):
        super().__init__(cfg, overrides)
        self.model = self._init_model(self.output_dim)

    def forward(self, _bias_params: Tensor, logits: Tensor) -> Tensor:
        return Layer.run_model_returning_hidden(self.model, logits)

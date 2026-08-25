from torch import Tensor

from emperor.augmentations.adaptive_parameters._biases.base import DynamicBiasAbstract
from emperor.augmentations.adaptive_parameters._biases.config import (
    GeneratorDynamicBiasConfig,
)
from emperor.layers import Layer


class GeneratorDynamicBias(DynamicBiasAbstract):
    def __init__(
        self,
        cfg: GeneratorDynamicBiasConfig,
        overrides: GeneratorDynamicBiasConfig | None = None,
    ):
        super().__init__(cfg, overrides)
        self.model = self._init_model(self.output_dim)

    def forward(self, _bias_params: Tensor, logits: Tensor) -> Tensor:
        generated_bias_state = Layer.run_model_from_hidden(self.model, logits)
        return generated_bias_state.hidden

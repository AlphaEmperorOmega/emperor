from torch.types import Tensor
from Emperor.base.utils import Module

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch.nn import Sequential
    from Emperor.base.layer import Layer
    from Emperor.base.layer import LayerStackConfig
    from Emperor.config import ModelConfig


class LinearLayerStack(Module):
    def __init__(
        self,
        cfg: "LayerStackConfig | ModelConfig",
        overrides: "LayerStackConfig | None" = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.overrides = overrides
        self.identifier = "layer_stack_config"
        overrides = self.__override_config(overrides)
        self.model = self.__create_multi_layer_model(cfg, overrides)

    def __create_multi_layer_model(self) -> "Layer | Sequential":
        from Emperor.base.layer import LayerStack

        return LayerStack(self.cfg, self.overrides).build_model()

    def __override_config(
        self, overrides: "LayerStackConfig | None"
    ) -> LayerStackConfig:
        from Emperor.linears.options import LinearLayerOptions

        if overrides is None:
            return LayerStackConfig(model_type=LinearLayerOptions.BASE)
        overrides.model_type = LinearLayerOptions.BASE
        return overrides

    def forward(self, input_batch: Tensor) -> Tensor:
        return self.model(input_batch)

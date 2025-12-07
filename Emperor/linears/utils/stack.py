from Emperor.base.layer import LayerStack, LayerStackConfig

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.config import ModelConfig


class LinearLayerStack(LayerStack):
    def __init__(
        self,
        cfg: "LayerStackConfig | ModelConfig",
        overrides: "LayerStackConfig | None" = None,
    ):
        super().__init__(cfg, overrides)
        self.model_type = self.__get_model_type()

    def __get_model_type(self):
        from Emperor.linears.options import LinearLayerOptions

        return LinearLayerOptions.BASE


class AdaptiveLinearLayerStack(LayerStack):
    def __init__(
        self,
        cfg: "LayerStackConfig | ModelConfig",
        overrides: "LayerStackConfig | None" = None,
    ):
        super().__init__(cfg, overrides)
        self.model_type = self.__get_model_type()

    def __get_model_type(self):
        from Emperor.linears.options import LinearLayerOptions

        return LinearLayerOptions.ADAPTIVE

from emperor.base.layer import LayerStack, LayerStackConfig

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.config import ModelConfig
    from emperor.linears.options import LinearLayerOptions


class LinearLayerStack(LayerStack):
    def __init__(
        self,
        cfg: "LayerStackConfig | ModelConfig",
        overrides: "LayerStackConfig | None" = None,
    ):
        overrides = self.__get_model_type(overrides)
        super().__init__(cfg, overrides)

    def __get_model_type(self, overrides: "LayerStackConfig") -> "LayerStackConfig":
        from emperor.linears.options import LinearLayerOptions

        return super()._override_model_type(overrides, LinearLayerOptions.BASE)


class AdaptiveLinearLayerStack(LayerStack):
    def __init__(
        self,
        cfg: "LayerStackConfig | ModelConfig",
        overrides: "LayerStackConfig | None" = None,
    ):
        overrides = self.__get_model_type(overrides)
        super().__init__(cfg, overrides)

    def __get_model_type(self, overrides: "LayerStackConfig") -> "LinearLayerOptions":
        from emperor.linears.options import LinearLayerOptions

        return super()._override_model_type(overrides, LinearLayerOptions.ADAPTIVE)

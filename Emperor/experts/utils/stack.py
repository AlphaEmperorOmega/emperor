from Emperor.base.layer import LayerStack, LayerStackConfig

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.config import ModelConfig


class MixtureOfExpertsStack(LayerStack):
    def __init__(
        self,
        cfg: "LayerStackConfig | ModelConfig",
        overrides: "LayerStackConfig | None" = None,
    ):
        overrides = self.__get_model_type(overrides)
        super().__init__(cfg, overrides)

    def __get_model_type(self, overrides: "LayerStackConfig") -> "LayerStackConfig":
        from Emperor.experts.options import MixtureOfExpertsOptions

        return super()._override_model_type(overrides, MixtureOfExpertsOptions.BASE)

from torch import Tensor
from Emperor.base.utils import Module
from Emperor.adaptive.utils.mixtures.base import AdaptiveMixtureConfig
from Emperor.adaptive.utils.mixtures.types.vector import VectorWeightsMixture
from Emperor.adaptive.utils.mixtures.types.matrix import (
    MatrixBiasMixture,
    MatrixWeightsMixture,
)
from Emperor.adaptive.utils.mixtures.types.generator import (
    GeneratorBiasMixture,
    GeneratorWeightsMixture,
)
from Emperor.adaptive.utils.mixtures.options import (
    AdaptiveBiasOptions,
    AdaptiveWeightOptions,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.adaptive.utils.layers import AdaptiveParameterLayerConfig
    from Emperor.adaptive.utils.mixtures.base import AdaptiveMixtureBase


class AdaptiveWeightSelector(Module):
    def __init__(
        self,
        cfg: "AdaptiveParameterLayerConfig",
        overrides: "AdaptiveParameterLayerConfig | None" = None,
    ):
        super().__init__()
        config = getattr(cfg, "linear_layer_config", cfg)
        self.cfg: "AdaptiveMixtureConfig" = self._overwrite_config(config, overrides)
        self.weight_parameter_option = cfg.weight_parameter_option

    def build_model(self) -> "AdaptiveMixtureBase":
        match self.weight_parameter_option:
            case AdaptiveWeightOptions.VECTOR:
                return VectorWeightsMixture(self.cfg)
            case AdaptiveWeightOptions.MATRIX:
                return MatrixWeightsMixture(self.cfg)
            case AdaptiveWeightOptions.GENERATOR:
                return GeneratorWeightsMixture(self.cfg)
            case _:
                raise ValueError(
                    f"Invalid weight parameter option provided: {self.weight_parameter_option}. Expected one of `AdaptiveWeightOptions`."
                )


class AdaptiveBiasSelector(Module):
    def __init__(
        self,
        cfg: "AdaptiveParameterLayerConfig",
        overrides: "AdaptiveParameterLayerConfig | None" = None,
    ):
        super().__init__()
        config = getattr(cfg, "linear_layer_config", cfg)
        self.cfg: "AdaptiveMixtureConfig" = self._overwrite_config(config, overrides)
        self.bias_parameter_option = cfg.bias_parameter_option
        self.model = self.__init_model()

    def build_model(self) -> "AdaptiveMixtureBase":
        match self.bias_parameter_option:
            case AdaptiveBiasOptions.MATRIX:
                return MatrixBiasMixture(self.cfg)
            case AdaptiveBiasOptions.GENERATOR:
                return GeneratorBiasMixture(self.cfg)
            case AdaptiveBiasOptions.DISABLED:
                raise ValueError(
                    "If the `bias_parameter_option` is set to `DISABLED`, this class should not be initialized"
                )

    def forward(
        self,
        bias_params: Tensor,
        logits: Tensor,
    ) -> Tensor | None:
        return self.model(bias_params, logits)

from Emperor.base.utils import Module
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


# TODO: In the future move those selectors in the
# `behaviours` modules
class AdaptiveWeightSelector(Module):
    def __init__(
        self,
        cfg: "AdaptiveParameterLayerConfig",
        overrides: "AdaptiveParameterLayerConfig | None" = None,
    ):
        super().__init__()
        self.cfg = self._overwrite_config(cfg, overrides)
        self.main_cfg = self._resolve_main_config(self.cfg, cfg)
        self.adaptive_weight_option = self.cfg.adaptive_weight_option

    def build_model(self) -> "AdaptiveMixtureBase":
        match self.adaptive_weight_option:
            case AdaptiveWeightOptions.VECTOR:
                return VectorWeightsMixture(self.main_cfg)
            case AdaptiveWeightOptions.MATRIX:
                return MatrixWeightsMixture(self.main_cfg)
            case AdaptiveWeightOptions.GENERATOR:
                return GeneratorWeightsMixture(self.main_cfg)
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
        self.cfg: "AdaptiveParameterLayerConfig" = self._overwrite_config(
            cfg, overrides
        )
        self.main_cfg = self._resolve_main_config(self.cfg, cfg)
        self.adaptive_bias_option = self.cfg.adaptive_bias_option

    def build_model(self) -> "AdaptiveMixtureBase":
        match self.adaptive_bias_option:
            case AdaptiveBiasOptions.MATRIX:
                return MatrixBiasMixture(self.main_cfg)
            case AdaptiveBiasOptions.GENERATOR:
                return GeneratorBiasMixture(self.main_cfg)
            case AdaptiveBiasOptions.DISABLED:
                raise ValueError(
                    "If the `bias_parameter_option` is set to `DISABLED`, this class should not be initialized"
                )
            case _:
                raise ValueError(
                    f"Invalid weight parameter option provided: {self.weight_parameter_option}. Expected one of `AdaptiveWeightOptions`."
                )

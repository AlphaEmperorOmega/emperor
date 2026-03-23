from emperor.parametric.utils.mixtures.base import AdaptiveMixtureConfig
from emperor.base.utils import Module
from emperor.parametric.utils.mixtures.types.vector import VectorWeightsMixture
from emperor.parametric.utils.mixtures.types.matrix import (
    MatrixBiasMixture,
    MatrixWeightsMixture,
)
from emperor.parametric.utils.mixtures.types.generator import (
    GeneratorBiasMixture,
    GeneratorWeightsMixture,
)
from emperor.parametric.utils.mixtures.options import (
    AdaptiveBiasOptions,
    AdaptiveWeightOptions,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.parametric.utils.layers import AdaptiveParameterLayerConfig
    from emperor.parametric.utils.mixtures.base import AdaptiveMixtureBase


# TODO: In the future move those selectors in the
# `augmentations` modules
class AdaptiveWeightSelector(Module):
    def __init__(
        self,
        cfg: "AdaptiveParameterLayerConfig",
        overrides: "AdaptiveParameterLayerConfig | None" = None,
    ):
        super().__init__()
        self.cfg = self._overwrite_config(cfg, overrides)
        self.main_cfg = self._resolve_main_config(self.cfg, cfg)
        self.input_dim = self.cfg.input_dim
        self.output_dim = self.cfg.output_dim
        self.adaptive_weight_option = self.cfg.adaptive_weight_option
        self.overrides = AdaptiveMixtureConfig(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
        )

    def build_model(self) -> "AdaptiveMixtureBase":
        match self.adaptive_weight_option:
            case AdaptiveWeightOptions.VECTOR:
                return VectorWeightsMixture(self.main_cfg, self.overrides)
            case AdaptiveWeightOptions.MATRIX:
                return MatrixWeightsMixture(self.main_cfg, self.overrides)
            case AdaptiveWeightOptions.GENERATOR:
                return GeneratorWeightsMixture(self.main_cfg, self.overrides)
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
        self.input_dim = self.cfg.input_dim
        self.output_dim = self.cfg.output_dim
        self.adaptive_bias_option = self.cfg.adaptive_bias_option
        self.overrides = AdaptiveMixtureConfig(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
        )

    def build_model(self) -> "AdaptiveMixtureBase":
        match self.adaptive_bias_option:
            case AdaptiveBiasOptions.MATRIX:
                return MatrixBiasMixture(self.main_cfg, self.overrides)
            case AdaptiveBiasOptions.GENERATOR:
                return GeneratorBiasMixture(self.main_cfg, self.overrides)
            case AdaptiveBiasOptions.DISABLED:
                raise ValueError(
                    "If the `bias_parameter_option` is set to `DISABLED`, this class should not be initialized"
                )
            case _:
                raise ValueError(
                    f"Invalid weight parameter option provided: {self.weight_parameter_option}. Expected one of `AdaptiveWeightOptions`."
                )

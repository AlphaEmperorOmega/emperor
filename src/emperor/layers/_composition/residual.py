from typing import TYPE_CHECKING, ClassVar

from torch import Tensor

from emperor.layers._composition.pairwise_residual import (
    AdditiveResidual,
    PairwiseResidual,
    WeightedBlendResidual,
    WeightedResidual,
)
from emperor.layers._config import ResidualConfig
from emperor.layers._options import ResidualConnectionOptions
from emperor.layers._validation import ResidualConnectionValidator
from emperor.nn import Module

if TYPE_CHECKING:
    from emperor.linears import LinearLayerConfig


class ResidualConnection(Module):
    VALIDATOR = ResidualConnectionValidator
    PAIRWISE_RESIDUAL_TYPES: ClassVar[
        dict[ResidualConnectionOptions, type[PairwiseResidual]]
    ] = {
        ResidualConnectionOptions.RESIDUAL: AdditiveResidual,
        ResidualConnectionOptions.WEIGHTED_RESIDUAL: WeightedResidual,
        ResidualConnectionOptions.WEIGHTED_BLEND: WeightedBlendResidual,
    }
    RESIDUAL_OPTION_TYPES = PAIRWISE_RESIDUAL_TYPES
    WEIGHTED_BLEND_INITIAL_ALPHA = WeightedBlendResidual.DEFAULT_INITIAL_ALPHA

    def __init__(
        self,
        cfg: ResidualConfig,
        overrides: ResidualConfig | None = None,
    ):
        super().__init__()
        self.cfg: ResidualConfig = self._override_config(cfg, overrides)
        self.VALIDATOR.validate(self)
        self.option: ResidualConnectionOptions = self.cfg.option
        self.residual_dim: int | None = self.cfg.residual_dim
        self.model_config: LinearLayerConfig | None = self.cfg.model_config
        residual_type = self.__residual_type_for_construction()
        parameters = residual_type.build_parameters(
            model_config=self.model_config,
            residual_dim=self.residual_dim,
            blend_initial_alpha=self.WEIGHTED_BLEND_INITIAL_ALPHA,
            build_model=self._build_from_config,
        )
        self.raw_weight = parameters.raw_weight
        self.model = parameters.model

    def __residual_type_for_construction(self) -> type[PairwiseResidual]:
        try:
            residual_type = self.PAIRWISE_RESIDUAL_TYPES.get(self.option)
        except TypeError:
            residual_type = None
        if residual_type is None:
            raise ValueError(
                f"Residual option does not use mixing coefficients: {self.option}."
            )
        return residual_type

    def __residual_option_type(self) -> type[PairwiseResidual]:
        try:
            residual_option_type = self.RESIDUAL_OPTION_TYPES.get(self.option)
        except TypeError:
            residual_option_type = None
        if residual_option_type is None:
            raise ValueError(
                "Unsupported residual connection option "
                f"{self.option} for ResidualConnection."
            )
        return residual_option_type

    def forward(self, current: Tensor, previous: Tensor) -> Tensor:
        return self.__residual_option_type().forward(
            self,
            current,
            previous,
        )

from typing import TYPE_CHECKING, ClassVar, cast

from torch import Tensor

from emperor.layers._composition.attention_residual import AttentionResidualOption
from emperor.layers._composition.pairwise_residual import (
    AdditiveResidual,
    PairwiseResidual,
    WeightedBlendResidual,
    WeightedResidual,
    _PairwiseResidualParameters,
)
from emperor.layers._config import ResidualConfig
from emperor.layers._options import ResidualConnectionOptions
from emperor.layers._validation import ResidualConnectionValidator
from emperor.nn import Module

if TYPE_CHECKING:
    from emperor.layers._composition.attention_residual import (
        AttentionResidual,
        AttentionResidualState,
    )
    from emperor.layers._config import AttentionResidualConfig
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
    RESIDUAL_OPTION_TYPES: ClassVar[
        dict[
            ResidualConnectionOptions,
            type[PairwiseResidual] | type[AttentionResidualOption],
        ]
    ] = {
        **PAIRWISE_RESIDUAL_TYPES,
        ResidualConnectionOptions.ATTENTION_RESIDUAL: AttentionResidualOption,
    }
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
        self.attention_config: AttentionResidualConfig | None = (
            self.cfg.attention_config
        )
        pairwise_parameters = self.__build_pairwise_parameters()
        self.raw_weight = pairwise_parameters.raw_weight
        self.model = pairwise_parameters.model
        self.attention_residual = self.__initialize_attention_residual()

    def __build_pairwise_parameters(
        self,
    ) -> _PairwiseResidualParameters:
        pairwise_residual_type = self.__pairwise_residual_type()
        if pairwise_residual_type is None:
            if self.option != ResidualConnectionOptions.ATTENTION_RESIDUAL:
                return self.VALIDATOR.reject_unsupported_mixing_coefficient_option(
                    self.option,
                )
            return _PairwiseResidualParameters()
        return pairwise_residual_type.build_parameters(
            model_config=self.model_config,
            residual_dim=self.residual_dim,
            blend_initial_alpha=self.WEIGHTED_BLEND_INITIAL_ALPHA,
            build_model=self._build_from_config,
        )

    def __pairwise_residual_type(self) -> "type[PairwiseResidual] | None":
        try:
            return self.PAIRWISE_RESIDUAL_TYPES.get(self.option)
        except TypeError:
            return None

    def __residual_option_type(
        self,
    ) -> "type[PairwiseResidual] | type[AttentionResidualOption]":
        try:
            residual_option_type = self.RESIDUAL_OPTION_TYPES.get(self.option)
        except TypeError:
            residual_option_type = None
        if residual_option_type is None:
            return self.VALIDATOR.reject_unsupported_runtime_option(self.option)
        return residual_option_type

    def __initialize_attention_residual(self) -> "AttentionResidual | None":
        if self.option != ResidualConnectionOptions.ATTENTION_RESIDUAL:
            return None
        from emperor.layers._config import AttentionResidualConfig

        attention_config = self.attention_config or AttentionResidualConfig()
        return self._build_from_config(
            attention_config,
            residual_dim=self.residual_dim,
        )

    def new_state(self, initial_source: Tensor) -> "AttentionResidualState":
        attention_residual = self.attention_residual
        self.VALIDATOR.validate_attention_residual_available(
            attention_residual,
        )
        return cast("AttentionResidual", attention_residual).new_state(initial_source)

    def forward(
        self,
        current: Tensor,
        previous: Tensor,
        *,
        residual_state: "AttentionResidualState | None" = None,
    ) -> Tensor:
        return self.__residual_option_type().forward(
            self,
            current,
            previous,
            residual_state=residual_state,
        )

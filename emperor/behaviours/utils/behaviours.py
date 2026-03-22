from emperor.base.utils import Module
from emperor.behaviours.utils.enums import (
    DynamicBiasOptions,
    DynamicDiagonalOptions,
    DynamicWeightOptions,
    LinearMemoryOptions,
)
from emperor.behaviours.utils.handlers.bias import (
    AffineBiasTransformHandler,
    BiasGeneratorHandler,
    BiasHandlerAbstract,
    ElementwiseBiasHandler,
)
from emperor.behaviours.utils.handlers.diagonal import (
    AntiDiagonalHandler,
    DiagonalAndAntiDiagonalHandler,
    DiagonalHandler,
    DiagonalHandlerAbstract,
)
from emperor.behaviours.utils.handlers.memory import (
    MemoryFusionHandler,
    MemoryHandlerAbstract,
    WeightedMemoryHandler,
)
from emperor.behaviours.utils.handlers.weight import (
    DualModelWeightHandler,
    SingleModelWeightHandler,
    WeightHandlerAbstract,
)

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from emperor.behaviours.model import AdaptiveParameterBehaviourConfig


class DynamicWeightFactory(Module):
    def __init__(
        self,
        cfg: "AdaptiveParameterBehaviourConfig",
        overrides: "AdaptiveParameterBehaviourConfig | None" = None,
    ):
        super().__init__()
        self.cfg: "AdaptiveParameterBehaviourConfig" = self._overwrite_config(
            cfg, overrides
        )
        self.weight_option = self.cfg.weight_option

    def build(self) -> WeightHandlerAbstract:
        match self.weight_option:
            case DynamicWeightOptions.DUAL_MODEL:
                return DualModelWeightHandler(self.cfg)
            case DynamicWeightOptions.SINGLE_MODEL:
                return SingleModelWeightHandler(self.cfg)
            case DynamicWeightOptions.DISABLED:
                raise ValueError(
                    "If the `weight_option` is set to `DISABLED`, this class should not be initialized"
                )


# TODO: Add option for a kernel to take the context
# of every token into account when computing the dynamic parameters
class DynamicDiagonalFactory(Module):
    def __init__(
        self,
        cfg: "AdaptiveParameterBehaviourConfig",
        overrides: "AdaptiveParameterBehaviourConfig | None" = None,
    ):
        super().__init__()
        config = getattr(cfg, "linear_layer_config", cfg)
        self.cfg: "AdaptiveParameterBehaviourConfig" = self._overwrite_config(
            config, overrides
        )
        self.diagonal_option = self.cfg.diagonal_option

    def build(self) -> DiagonalHandlerAbstract:
        match self.diagonal_option:
            case DynamicDiagonalOptions.DIAGONAL:
                return DiagonalHandler(self.cfg)
            case DynamicDiagonalOptions.ANTI_DIAGONAL:
                return AntiDiagonalHandler(self.cfg)
            case DynamicDiagonalOptions.DIAGONAL_AND_ANTI_DIAGONAL:
                return DiagonalAndAntiDiagonalHandler(self.cfg)
            case DynamicDiagonalOptions.DISABLED:
                raise ValueError(
                    "If the `diagonal_option` is set to `DISABLED`, this class should not be initialized"
                )


class DynamicBiasFactory(Module):
    def __init__(
        self,
        cfg: "AdaptiveParameterBehaviourConfig",
        overrides: "AdaptiveParameterBehaviourConfig | None" = None,
    ):
        super().__init__()
        self.cfg: "AdaptiveParameterBehaviourConfig" = self._overwrite_config(
            cfg, overrides
        )
        self.bias_option = self.cfg.bias_option

    def build(self) -> BiasHandlerAbstract:
        match self.bias_option:
            case DynamicBiasOptions.SCALE_AND_OFFSET:
                return AffineBiasTransformHandler(self.cfg)
            case DynamicBiasOptions.ELEMENT_WISE_OFFSET:
                return ElementwiseBiasHandler(self.cfg)
            case DynamicBiasOptions.DYNAMIC_PARAMETERS:
                return BiasGeneratorHandler(self.cfg)
            case DynamicBiasOptions.DISABLED:
                raise ValueError(
                    "If the `bias_option` is set to `DISABLED`, this class should not be initialized"
                )


class DynamicMemoryFactory(Module):
    def __init__(
        self,
        cfg: "AdaptiveParameterBehaviourConfig",
        overrides: "AdaptiveParameterBehaviourConfig | None" = None,
    ):
        super().__init__()
        self.cfg: "AdaptiveParameterBehaviourConfig" = self._overwrite_config(
            cfg, overrides
        )
        self.memory_option = self.cfg.memory_option

    def build(self) -> MemoryHandlerAbstract:
        match self.memory_option:
            case LinearMemoryOptions.FUSION:
                return MemoryFusionHandler(self.cfg)
            case LinearMemoryOptions.WEIGHTED:
                return WeightedMemoryHandler(self.cfg)
            case LinearMemoryOptions.DISABLED:
                raise ValueError(
                    "If the `memory_option` is set to `DISABLED`, this class should not be initialized"
                )

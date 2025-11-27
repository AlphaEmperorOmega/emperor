import torch

from torch import Tensor
from Emperor.base.utils import Module
from Emperor.linears.utils.handlers.parameter import DepthMappingLayerStack
from Emperor.linears.utils.enums import (
    DynamicBiasOptions,
    DynamicDiagonalOptions,
    LinearMemoryOptions,
)
from Emperor.linears.utils.handlers.memory import (
    MemoryFusionHandler,
    MemoryHandlerAbstract,
    WeightedMemoryHandler,
)
from Emperor.linears.utils.handlers.bias import (
    BiasGeneratorHandler,
    BiasHandlerAbstract,
    AffineBiasTransformHandler,
    ElementwiseBiasHandler,
)
from Emperor.linears.utils.handlers.diagonal import (
    AntiDiagonalHandler,
    DiagonalAndAntiDiagonalHandler,
    DiagonalHandler,
    DiagonalHandlerAbstract,
)
from Emperor.base.layer import (
    LayerStackConfig,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.linears.utils.layers import DynamicLinearLayerConfig


class DynamicParametersBehaviour(Module):
    def __init__(
        self,
        cfg: "DynamicLinearLayerConfig",
        overrides: "DynamicLinearLayerConfig | None" = None,
    ):
        super().__init__()
        config = getattr(cfg, "linear_layer_config", cfg)
        self.cfg: "DynamicLinearLayerConfig" = self._overwrite_config(config, overrides)
        self.input_dim = self.cfg.input_dim
        self.output_dim = self.cfg.output_dim
        self.input_model = self.__init_input_model()
        self.output_model = self.__init_output_model()

    def __init_input_model(self) -> DepthMappingLayerStack:
        overrides = LayerStackConfig(
            input_dim=self.input_dim,
            output_dim=self.input_dim,
        )
        return self.__init_generator_model(overrides)

    def __init_output_model(self) -> DepthMappingLayerStack:
        overrides = LayerStackConfig(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
        )
        return self.__init_generator_model(overrides)

    def __init_generator_model(
        self, overrides: "LayerStackConfig"
    ) -> DepthMappingLayerStack:
        return DepthMappingLayerStack(self.cfg, overrides)

    def forward(
        self,
        weight_params: Tensor,
        logits: Tensor,
    ) -> Tensor:
        input_vectors = self.input_model(logits)
        output_vectors = self.output_model(logits)
        outer_product = self.__compute_outer_product(input_vectors, output_vectors)
        dynamic_params = self.__compute_dynamic_weights(outer_product)
        return weight_params + dynamic_params

    def __compute_dynamic_weights(self, outer_product: Tensor) -> Tensor:
        return outer_product.sum(dim=1)

    def __compute_outer_product(
        self,
        input_vectors: Tensor,
        output_vectors: Tensor,
    ) -> Tensor:
        input_vectors = self.__normalize_vectors(input_vectors)
        output_vectors = self.__normalize_vectors(output_vectors)
        return torch.einsum("bij,bik->bijk", input_vectors, output_vectors)

    def __normalize_vectors(
        self,
        outer_product: Tensor,
    ) -> Tensor:
        # TODO: Add flag to normalize the the input before or after the outer product
        # TODO: Temporary nomralization just to check what's happening
        # match norm_option:
        #     case OuterProductNormOptions.RELU:
        #         return F.relu(outer_product)
        #     case OuterProductNormOptions.TANH:
        #         return F.tanh(outer_product)
        #     case OuterProductNormOptions.SIGMOID:
        #         return F.sigmoid(outer_product)
        #     case _:
        #         return outer_product
        return torch.clamp(outer_product, -5.0, 5.0)


# TODO: Add option for a kernel to take the context
# of every token into account when computing the dynamic parameters
class DynamicDiagonalSelector(Module):
    def __init__(
        self,
        cfg: "DynamicLinearLayerConfig",
        overrides: "DynamicLinearLayerConfig | None" = None,
    ):
        super().__init__()
        config = getattr(cfg, "linear_layer_config", cfg)
        self.cfg: "DynamicLinearLayerConfig" = self._overwrite_config(config, overrides)
        self.diagonal_option = self.cfg.diagonal_option
        self.model = self.__init_bias_model()

    def __init_bias_model(self) -> DiagonalHandlerAbstract:
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

    def forward(
        self,
        weight_params: Tensor,
        logits: Tensor,
    ) -> Tensor | None:
        return self.model(weight_params, logits)


class DynamicBiasSelector(Module):
    def __init__(
        self,
        cfg: "DynamicLinearLayerConfig",
        overrides: "DynamicLinearLayerConfig | None" = None,
    ):
        super().__init__()
        config = getattr(cfg, "linear_layer_config", cfg)
        self.cfg: "DynamicLinearLayerConfig" = self._overwrite_config(config, overrides)
        self.bias_option = self.cfg.bias_option
        self.model = self.__init_bias_model()

    def __init_bias_model(self) -> BiasHandlerAbstract:
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

    def forward(
        self,
        bias_params: Tensor,
        logits: Tensor,
    ) -> Tensor | None:
        return self.model(bias_params, logits)


class DynamicMemorySelector(Module):
    def __init__(
        self,
        cfg: "DynamicLinearLayerConfig",
        overrides: "DynamicLinearLayerConfig | None" = None,
    ):
        super().__init__()
        config = getattr(cfg, "linear_layer_config", cfg)
        self.cfg: "DynamicLinearLayerConfig" = self._overwrite_config(config, overrides)
        self.memory_option = self.cfg.memory_option
        self.model = self.__init_memory_model()

    def __init_memory_model(self) -> MemoryHandlerAbstract:
        match self.memory_option:
            case LinearMemoryOptions.FUSION:
                return MemoryFusionHandler(self.cfg)
            case LinearMemoryOptions.WEIGHTED:
                return WeightedMemoryHandler(self.cfg)
            case LinearMemoryOptions.DISABLED:
                raise ValueError(
                    "If the `memory_option` is set to `DISABLED`, this class should not be initialized"
                )

    def forward(
        self,
        logits: Tensor,
    ) -> Tensor | None:
        return self.model(logits)

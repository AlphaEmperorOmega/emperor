import torch
from torch import Tensor
from torch.nn import Sequential
from emperor.base.utils import Module
from emperor.base.layer import Layer, LayerStackConfig
from emperor.behaviours.options import (
    LinearMemoryPositionOptions,
    LinearMemorySizeOptions,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.behaviours.model import AdaptiveParameterBehaviourConfig


class MemoryHandlerAbstract(Module):
    def __init__(
        self,
        cfg: "AdaptiveParameterBehaviourConfig",
    ):
        super().__init__()
        self.cfg = cfg
        self.cfg_main = self._resolve_main_config(self.cfg, cfg)
        self.input_dim = self.cfg.input_dim
        self.output_dim = self.cfg.output_dim
        self.memory_size_option = self.cfg.memory_size_option
        self.memory_position_option = self.cfg.memory_position_option

    def _validate_memory_option(self):
        if self.memory_size_option == LinearMemorySizeOptions.DISABLED:
            raise ValueError(
                "The `memory_size_option` cannot be set to LinearMemorySizeOptions.DISABLED if this handler is used."
            )

    def _get_memory_dim(self) -> int:
        if self.memory_position_option == LinearMemoryPositionOptions.BEFORE_AFFINE:
            return self.input_dim
        return self.output_dim

    def _init_model(self, config, overrides) -> "Layer | Sequential":
        from emperor.linears.utils.stack import LinearLayerStack

        return LinearLayerStack(config, overrides).build_model()


class MemoryFusionHandler(MemoryHandlerAbstract):
    def __init__(
        self,
        cfg: "AdaptiveParameterBehaviourConfig",
    ):
        super().__init__(cfg)
        self.memory_model = self.__init_memory_model()
        self.compression_model = self.__init_compression_model()
        self._validate_memory_option()

    def __init_memory_model(self) -> "Layer | Sequential":
        input_dim = self._get_memory_dim()
        output_dim = self.memory_size_option.value
        overrides = LayerStackConfig(input_dim=input_dim, output_dim=output_dim)
        return self._init_model(self.cfg_main, overrides)

    def __init_compression_model(self) -> "Layer | Sequential":
        dim = self._get_memory_dim()
        compression_input_dim = dim + self.memory_size_option.value
        overrides = LayerStackConfig(
            input_dim=compression_input_dim,
            hidden_dim=dim,
            output_dim=dim,
        )
        return self._init_model(self.cfg_main, overrides)

    def forward(self, logits: Tensor) -> Tensor:
        memory = self.memory_model(logits)
        combined = torch.cat([logits, memory], dim=-1)
        return self.compression_model(combined)


class WeightedMemoryHandler(MemoryHandlerAbstract):
    def __init__(
        self,
        cfg: "AdaptiveParameterBehaviourConfig",
    ):
        super().__init__(cfg)

        self.memory_model = self.__init_memory_model()
        self.memory_weight_model = self.__init_weight_model()
        self._validate_memory_option()

    def __init_memory_model(self) -> "Layer | Sequential":
        dim = self._get_memory_dim()
        overrides = LayerStackConfig(input_dim=dim, output_dim=dim)
        return self._init_model(self.cfg_main, overrides)

    def __init_weight_model(self) -> "Layer | Sequential":
        dim = self._get_memory_dim() * 2
        overrides = LayerStackConfig(input_dim=dim, hidden_dim=dim, output_dim=2)
        return self._init_model(self.cfg_main, overrides)

    def forward(self, logits: Tensor) -> Tensor:
        memory = self.memory_model(logits)
        weights = self.__compute_weights(logits, memory)
        merged_logits_memory = self.__reshape_and_concat(logits, memory)
        return self.__blend_inputs_and_memory(merged_logits_memory, weights)

    def __compute_weights(self, logits: Tensor, memory: Tensor) -> Tensor:
        combined = torch.cat([logits, memory], dim=-1)
        weight_logits = self.memory_weight_model(combined)
        weights = torch.softmax(weight_logits, dim=-1)
        return weights.unsqueeze(-1)

    def __reshape_and_concat(self, logits: Tensor, memory: Tensor) -> Tensor:
        logits = logits.unsqueeze(-2)
        memory = memory.unsqueeze(-2)
        return torch.cat((logits, memory), dim=-2)

    def __blend_inputs_and_memory(
        self, input_and_memory: Tensor, weights: Tensor
    ) -> Tensor:
        weighted_logits = input_and_memory * weights
        return torch.sum(weighted_logits, dim=-2)

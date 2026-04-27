import torch

from torch import Tensor
from torch.nn import Sequential

from emperor.base.layer import Layer, LayerStackConfig
from emperor.memory.config import DynamicMemoryConfig
from emperor.memory.core.base import DynamicMemoryAbstract
from emperor.memory.options import DynamicMemoryOptions


@DynamicMemoryAbstract.register(DynamicMemoryOptions.WEIGHTED)
class WeightedDynamicMemory(DynamicMemoryAbstract):
    def __init__(
        self,
        cfg: DynamicMemoryConfig,
        overrides: DynamicMemoryConfig | None = None,
    ):
        super().__init__(cfg, overrides)
        self.memory_model = self.__init_memory_model()
        self.memory_decoder = (
            self.__init_memory_decoder() if self.test_time_training_flag else None
        )
        self.memory_weight_model = self.__init_weight_model()

    def __init_memory_model(self) -> "Layer | Sequential":
        layer_overrides = LayerStackConfig(
            input_dim=self.memory_dim,
            output_dim=self.memory_dim,
        )
        return self._init_model(layer_overrides)

    def __init_memory_decoder(self) -> "Layer | Sequential":
        layer_overrides = LayerStackConfig(
            input_dim=self.memory_dim,
            output_dim=self.memory_dim,
        )
        return self._init_model(layer_overrides)

    def __init_weight_model(self) -> "Layer | Sequential":
        layer_overrides = LayerStackConfig(
            input_dim=self.memory_dim * 2,
            output_dim=2,
        )
        return self._init_model(layer_overrides)

    def forward(self, logits: Tensor) -> Tensor:
        if self.test_time_training_flag:
            memory = self._adapt_and_retrieve(
                logits, self.memory_model, self.memory_decoder
            )
        else:
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

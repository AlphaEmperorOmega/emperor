import torch
from torch import Tensor

from emperor.memory.config import WeightedDynamicMemoryConfig
from emperor.memory.core.base import DynamicMemoryAbstract


class WeightedDynamicMemory(DynamicMemoryAbstract):
    def __init__(
        self,
        cfg: WeightedDynamicMemoryConfig,
        overrides: WeightedDynamicMemoryConfig | None = None,
    ):
        super().__init__(cfg, overrides)
        self.memory_model = self.__build_memory_model()
        self.memory_decoder = self.__build_memory_decoder()
        self.memory_weight_model = self.__build_memory_weight_model()

    def __build_memory_model(self):
        return self._build_generator_with_dims(
            input_dim=self.memory_dim,
            output_dim=self.memory_dim,
            validate_test_time_training_target=self.test_time_training_flag,
        )

    def __build_memory_decoder(self):
        if not self.test_time_training_flag:
            return None
        return self._build_generator_with_dims(
            input_dim=self.memory_dim,
            output_dim=self.memory_dim,
        )

    def __build_memory_weight_model(self):
        return self._build_generator_with_dims(
            input_dim=self.memory_dim * 2,
            output_dim=2,
        )

    def forward(self, logits: Tensor) -> Tensor:
        self.VALIDATOR.validate_forward_inputs(logits, self.memory_dim)
        if self.test_time_training_flag:
            memory = self._adapt_and_retrieve(
                logits, self.memory_model, self.memory_decoder
            )
        else:
            memory = self._run_model(self.memory_model, logits)
        weights = self.__compute_weights(logits, memory)
        merged_logits_memory = self.__reshape_and_concat(logits, memory)
        return self.__blend_inputs_and_memory(merged_logits_memory, weights)

    def __compute_weights(self, logits: Tensor, memory: Tensor) -> Tensor:
        combined = torch.cat([logits, memory], dim=-1)
        weight_logits = self._run_model(self.memory_weight_model, combined)
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

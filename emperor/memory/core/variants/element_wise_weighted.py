import torch
from torch import Tensor

from emperor.memory.config import (
    ElementWiseWeightedDynamicMemoryConfig,
)
from emperor.memory.core._validator import DynamicMemoryValidator
from emperor.memory.core.base import DynamicMemoryAbstract


class ElementWiseWeightedDynamicMemory(DynamicMemoryAbstract):
    def __init__(
        self,
        cfg: ElementWiseWeightedDynamicMemoryConfig,
        overrides: ElementWiseWeightedDynamicMemoryConfig | None = None,
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
            output_dim=self.memory_dim,
        )

    def forward(self, logits: Tensor) -> Tensor:
        DynamicMemoryValidator.validate_forward_inputs(logits, self.memory_dim)
        if self.test_time_training_flag:
            memory = self._adapt_and_retrieve(
                logits, self.memory_model, self.memory_decoder
            )
        else:
            memory = self._run_model(self.memory_model, logits)
        feature_weights = self.__compute_feature_weights(logits, memory)
        return self.__blend_inputs_and_memory(logits, memory, feature_weights)

    def __compute_feature_weights(self, logits: Tensor, memory: Tensor) -> Tensor:
        combined = torch.cat([logits, memory], dim=-1)
        weight_logits = self._run_model(self.memory_weight_model, combined)
        return torch.sigmoid(weight_logits)

    def __blend_inputs_and_memory(
        self,
        logits: Tensor,
        memory: Tensor,
        feature_weights: Tensor,
    ) -> Tensor:
        return (1 - feature_weights) * logits + feature_weights * memory

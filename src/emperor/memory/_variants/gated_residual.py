import torch
from torch import Tensor

from emperor.memory._base import DynamicMemoryAbstract
from emperor.memory._config import GatedResidualDynamicMemoryConfig


class GatedResidualDynamicMemory(DynamicMemoryAbstract):
    def __init__(
        self,
        cfg: GatedResidualDynamicMemoryConfig,
        overrides: GatedResidualDynamicMemoryConfig | None = None,
    ):
        super().__init__(cfg, overrides)
        self.memory_model = self.__build_memory_model()
        self.memory_decoder = self.__build_memory_decoder()
        self.memory_gate_model = self.__build_memory_gate_model()

    def __build_memory_model(self):
        return self._build_memory_generator_with_dims(
            input_dim=self.memory_dim,
            output_dim=self.memory_dim,
        )

    def __build_memory_decoder(self):
        if not self.test_time_training_flag:
            return None
        return self._build_generator_with_dims(
            input_dim=self.memory_dim,
            output_dim=self.memory_dim,
        )

    def __build_memory_gate_model(self):
        return self._build_generator_with_dims(
            input_dim=self.memory_dim * 2,
            hidden_dim=self.memory_dim * 2,
            output_dim=self.memory_dim,
        )

    def forward(self, logits: Tensor) -> Tensor:
        self.VALIDATOR.validate_forward_inputs(logits, self.memory_dim)
        memory = self.__retrieve_memory(logits)
        memory_gate = self.__compute_memory_gate(logits, memory)
        return logits + memory_gate * memory

    def __retrieve_memory(self, logits: Tensor) -> Tensor:
        if self.test_time_training_flag:
            return self._adapt_and_retrieve(
                logits, self.memory_model, self.memory_decoder
            )
        return self._run_model(self.memory_model, logits)

    def __compute_memory_gate(self, logits: Tensor, memory: Tensor) -> Tensor:
        combined = torch.cat([logits, memory], dim=-1)
        gate_logits = self._run_model(self.memory_gate_model, combined)
        return torch.sigmoid(gate_logits)

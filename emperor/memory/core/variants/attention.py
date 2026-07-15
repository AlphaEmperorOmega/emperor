import torch
from torch import Tensor

from emperor.memory.config import AttentionDynamicMemoryConfig
from emperor.memory.core.base import DynamicMemoryAbstract


class AttentionDynamicMemory(DynamicMemoryAbstract):
    def __init__(
        self,
        cfg: AttentionDynamicMemoryConfig,
        overrides: AttentionDynamicMemoryConfig | None = None,
    ):
        super().__init__(cfg, overrides)
        self.VALIDATOR.validate_attention_num_memory_slots(self.cfg)
        self.num_memory_slots = self.cfg.num_memory_slots
        memory_bank_dim = self.num_memory_slots * self.memory_dim
        self.memory_model = self.__build_memory_model(memory_bank_dim)
        self.memory_decoder = self.__build_memory_decoder(memory_bank_dim)
        self.query_model = self.__build_projection_model()
        self.key_model = self.__build_projection_model()
        self.value_model = self.__build_projection_model()
        self.output_model = self.__build_projection_model()
        self.memory_gate_model = self.__build_memory_gate_model()

    def __build_memory_model(self, memory_bank_dim: int):
        return self._build_generator_with_dims(
            input_dim=self.memory_dim,
            output_dim=memory_bank_dim,
            validate_test_time_training_target=self.test_time_training_flag,
        )

    def __build_memory_decoder(self, memory_bank_dim: int):
        if not self.test_time_training_flag:
            return None
        return self._build_generator_with_dims(
            input_dim=memory_bank_dim,
            output_dim=self.memory_dim,
        )

    def __build_projection_model(self):
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
        if self.test_time_training_flag:
            memory_bank_flat = self._adapt_and_retrieve(
                logits, self.memory_model, self.memory_decoder
            )
        else:
            memory_bank_flat = self._run_model(self.memory_model, logits)
        memory_bank = self.__reshape_memory_bank(memory_bank_flat)
        attended_memory = self.__attend_to_memory_bank(logits, memory_bank)
        memory_update = self._run_model(self.output_model, attended_memory)
        memory_gate = self.__compute_memory_gate(logits, memory_update)
        return logits + memory_gate * memory_update

    def __reshape_memory_bank(self, memory_bank_flat: Tensor) -> Tensor:
        return memory_bank_flat.reshape(
            *memory_bank_flat.shape[:-1], self.num_memory_slots, self.memory_dim
        )

    def __attend_to_memory_bank(self, logits: Tensor, memory_bank: Tensor) -> Tensor:
        query = self._run_model(self.query_model, logits).unsqueeze(-2)
        keys = self._run_model(self.key_model, memory_bank)
        values = self._run_model(self.value_model, memory_bank)
        attention_scores = torch.matmul(query, keys.transpose(-2, -1))
        attention_scores = attention_scores / (self.memory_dim**0.5)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        return torch.matmul(attention_weights, values).squeeze(-2)

    def __compute_memory_gate(
        self,
        logits: Tensor,
        memory_update: Tensor,
    ) -> Tensor:
        combined = torch.cat([logits, memory_update], dim=-1)
        gate_logits = self._run_model(self.memory_gate_model, combined)
        return torch.sigmoid(gate_logits)

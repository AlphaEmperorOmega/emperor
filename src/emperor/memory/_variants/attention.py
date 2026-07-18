import torch
from torch import Tensor

from emperor.memory._base import DynamicMemoryAbstract
from emperor.memory._config import AttentionDynamicMemoryConfig


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
        return self._build_memory_generator_with_dims(
            input_dim=self.memory_dim,
            output_dim=memory_bank_dim,
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
        memory_bank_flat = self.__retrieve_memory_bank(logits)
        memory_bank = self.__reshape_memory_bank(memory_bank_flat)
        attended_memory = self.__attend_to_memory_bank(logits, memory_bank)
        memory_update = self._run_model(self.output_model, attended_memory)
        memory_gate = self.__compute_memory_gate(logits, memory_update)
        gated_memory_update = memory_gate * memory_update
        updated_logits = logits + gated_memory_update
        return updated_logits

    def __retrieve_memory_bank(self, logits: Tensor) -> Tensor:
        if self.test_time_training_flag:
            return self._adapt_and_retrieve(
                logits, self.memory_model, self.memory_decoder
            )
        return self._run_model(self.memory_model, logits)

    def __reshape_memory_bank(self, memory_bank_flat: Tensor) -> Tensor:
        logit_positions_shape = memory_bank_flat.shape[:-1]
        memory_bank = memory_bank_flat.reshape(
            *logit_positions_shape, self.num_memory_slots, self.memory_dim
        )
        return memory_bank

    def __attend_to_memory_bank(self, logits: Tensor, memory_bank: Tensor) -> Tensor:
        query = self._run_model(self.query_model, logits).unsqueeze(-2)
        keys = self._run_model(self.key_model, memory_bank)
        values = self._run_model(self.value_model, memory_bank)
        attention_scores = torch.matmul(query, keys.transpose(-2, -1))
        attention_score_scale = self.memory_dim**0.5
        scaled_attention_scores = attention_scores / attention_score_scale
        attention_weights = torch.softmax(scaled_attention_scores, dim=-1)
        attended_memory = torch.matmul(attention_weights, values)
        singleton_query_dimension = -2
        return attended_memory.squeeze(singleton_query_dimension)

    def __compute_memory_gate(
        self,
        logits: Tensor,
        memory_update: Tensor,
    ) -> Tensor:
        combined = torch.cat([logits, memory_update], dim=-1)
        gate_logits = self._run_model(self.memory_gate_model, combined)
        return torch.sigmoid(gate_logits)

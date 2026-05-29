import torch

from torch import Tensor
from torch.nn import Sequential

from emperor.base.layer import Layer, LayerStackConfig
from emperor.memory.config import AttentionDynamicMemoryConfig, DynamicMemoryConfig
from emperor.memory.core.base import DynamicMemoryAbstract
from emperor.memory.core._validator import DynamicMemoryValidator


class AttentionDynamicMemory(DynamicMemoryAbstract):
    def __init__(
        self,
        cfg: AttentionDynamicMemoryConfig,
        overrides: DynamicMemoryConfig | None = None,
    ):
        super().__init__(cfg, overrides)
        DynamicMemoryValidator.validate_attention_num_memory_slots(self.cfg)
        self.num_memory_slots = self.cfg.num_memory_slots
        self.memory_model = self.__init_memory_model()
        self.memory_decoder = (
            self.__init_memory_decoder() if self.test_time_training_flag else None
        )
        self.query_model = self.__init_query_model()
        self.key_model = self.__init_key_model()
        self.value_model = self.__init_value_model()
        self.output_model = self.__init_output_model()
        self.memory_gate_model = self.__init_memory_gate_model()

    def __init_memory_model(self) -> "Layer | Sequential":
        layer_overrides = LayerStackConfig(
            input_dim=self.memory_dim,
            output_dim=self.num_memory_slots * self.memory_dim,
        )
        return self._init_model(layer_overrides)

    def __init_memory_decoder(self) -> "Layer | Sequential":
        layer_overrides = LayerStackConfig(
            input_dim=self.num_memory_slots * self.memory_dim,
            output_dim=self.memory_dim,
        )
        return self._init_model(layer_overrides)

    def __init_query_model(self) -> "Layer | Sequential":
        layer_overrides = LayerStackConfig(
            input_dim=self.memory_dim,
            output_dim=self.memory_dim,
        )
        return self._init_model(layer_overrides)

    def __init_key_model(self) -> "Layer | Sequential":
        layer_overrides = LayerStackConfig(
            input_dim=self.memory_dim,
            output_dim=self.memory_dim,
        )
        return self._init_model(layer_overrides)

    def __init_value_model(self) -> "Layer | Sequential":
        layer_overrides = LayerStackConfig(
            input_dim=self.memory_dim,
            output_dim=self.memory_dim,
        )
        return self._init_model(layer_overrides)

    def __init_output_model(self) -> "Layer | Sequential":
        layer_overrides = LayerStackConfig(
            input_dim=self.memory_dim,
            output_dim=self.memory_dim,
        )
        return self._init_model(layer_overrides)

    def __init_memory_gate_model(self) -> "Layer | Sequential":
        layer_overrides = LayerStackConfig(
            input_dim=self.memory_dim * 2,
            hidden_dim=self.memory_dim * 2,
            output_dim=self.memory_dim,
        )
        return self._init_model(layer_overrides)

    def forward(self, logits: Tensor) -> Tensor:
        DynamicMemoryValidator.validate_forward_inputs(logits, self.memory_dim)
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

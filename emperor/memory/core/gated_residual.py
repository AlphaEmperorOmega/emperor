import torch

from torch import Tensor
from torch.nn import Sequential

from emperor.base.layer import Layer, LayerStackConfig
from emperor.memory.config import DynamicMemoryConfig, GatedResidualDynamicMemoryConfig
from emperor.memory.core.base import DynamicMemoryAbstract
from emperor.memory.core._validator import DynamicMemoryValidator


class GatedResidualDynamicMemory(DynamicMemoryAbstract):
    def __init__(
        self,
        cfg: GatedResidualDynamicMemoryConfig,
        overrides: DynamicMemoryConfig | None = None,
    ):
        super().__init__(cfg, overrides)
        self.memory_model = self.__init_memory_model()
        self.memory_decoder = (
            self.__init_memory_decoder() if self.test_time_training_flag else None
        )
        self.memory_gate_model = self.__init_memory_gate_model()

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

    def __init_memory_gate_model(self) -> "Layer | Sequential":
        memory_gate_model_input_dim = self.memory_dim * 2
        layer_overrides = LayerStackConfig(
            input_dim=memory_gate_model_input_dim,
            hidden_dim=memory_gate_model_input_dim,
            output_dim=self.memory_dim,
        )
        return self._init_model(layer_overrides)

    def forward(self, logits: Tensor) -> Tensor:
        DynamicMemoryValidator.validate_forward_inputs(logits, self.memory_dim)
        if self.test_time_training_flag:
            memory = self._adapt_and_retrieve(
                logits, self.memory_model, self.memory_decoder
            )
        else:
            memory = self._run_model(self.memory_model, logits)
        memory_gate = self.__compute_memory_gate(logits, memory)
        return logits + memory_gate * memory

    def __compute_memory_gate(self, logits: Tensor, memory: Tensor) -> Tensor:
        combined = torch.cat([logits, memory], dim=-1)
        gate_logits = self._run_model(self.memory_gate_model, combined)
        return torch.sigmoid(gate_logits)

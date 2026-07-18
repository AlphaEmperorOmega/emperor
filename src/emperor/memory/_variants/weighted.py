import torch
from torch import Tensor

from emperor.memory._base import DynamicMemoryAbstract
from emperor.memory._config import WeightedDynamicMemoryConfig


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

    def __build_memory_weight_model(self):
        return self._build_generator_with_dims(
            input_dim=self.memory_dim * 2,
            output_dim=2,
        )

    def forward(self, logits: Tensor) -> Tensor:
        self.VALIDATOR.validate_forward_inputs(logits, self.memory_dim)
        memory = self.__retrieve_memory(logits)
        broadcastable_blend_weights = self.__compute_weights(logits, memory)
        blend_sources = self.__reshape_and_concat(logits, memory)
        blended_logits = self.__blend_inputs_and_memory(
            blend_sources, broadcastable_blend_weights
        )
        return blended_logits

    def __retrieve_memory(self, logits: Tensor) -> Tensor:
        if self.test_time_training_flag:
            return self._adapt_and_retrieve(
                logits, self.memory_model, self.memory_decoder
            )
        return self._run_model(self.memory_model, logits)

    def __compute_weights(self, logits: Tensor, memory: Tensor) -> Tensor:
        weighting_context = torch.cat([logits, memory], dim=-1)
        blend_weight_logits = self._run_model(
            self.memory_weight_model, weighting_context
        )
        blend_weights = torch.softmax(blend_weight_logits, dim=-1)
        broadcastable_blend_weights = blend_weights.unsqueeze(-1)
        return broadcastable_blend_weights

    def __reshape_and_concat(self, logits: Tensor, memory: Tensor) -> Tensor:
        blend_source_dimension = -2
        logits_source = logits.unsqueeze(blend_source_dimension)
        memory_source = memory.unsqueeze(blend_source_dimension)
        blend_sources = torch.cat(
            (logits_source, memory_source), dim=blend_source_dimension
        )
        return blend_sources

    def __blend_inputs_and_memory(
        self,
        blend_sources: Tensor,
        broadcastable_blend_weights: Tensor,
    ) -> Tensor:
        weighted_blend_sources = blend_sources * broadcastable_blend_weights
        blend_source_dimension = -2
        blended_logits = torch.sum(weighted_blend_sources, dim=blend_source_dimension)
        return blended_logits

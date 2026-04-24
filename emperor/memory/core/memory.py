import torch

from torch import Tensor
from torch.nn import Sequential
from torch.nn import functional as F

from emperor.base.layer import Layer, LayerStackConfig
from emperor.base.registry import subclass_registry
from emperor.base.utils import Module
from emperor.memory.config import DynamicMemoryConfig
from emperor.memory.core._validator import DynamicMemoryValidator
from emperor.memory.options import DynamicMemoryOptions, MemoryPositionOptions


@subclass_registry
class DynamicMemoryAbstract(Module):
    def __init__(
        self,
        cfg: DynamicMemoryConfig,
        overrides: DynamicMemoryConfig | None = None,
    ):
        super().__init__()
        self.cfg: DynamicMemoryConfig = self._override_config(cfg, overrides)
        DynamicMemoryValidator.validate(self)
        self.input_dim = self.cfg.input_dim
        self.output_dim = self.cfg.output_dim
        self.num_memory_slots = self.cfg.num_memory_slots
        self.memory_position_option = self.cfg.memory_position_option
        self.memory_dim = self.__get_memory_dim()
        self.test_time_training_learning_rate = (
            self.cfg.test_time_training_learning_rate
        )
        self.test_time_training_num_inner_steps = (
            self.cfg.test_time_training_num_inner_steps
        )
        self.model_config = self.cfg.model_config
        self.test_time_training_flag = (
            self.test_time_training_learning_rate is not None
            and self.test_time_training_num_inner_steps is not None
        )

    def _init_model(self, overrides: LayerStackConfig) -> "Layer | Sequential":
        generator_model = self.model_config.build(overrides)
        DynamicMemoryValidator.validate_generator_model(generator_model)
        return generator_model

    def __get_memory_dim(self) -> int:
        if self.memory_position_option == MemoryPositionOptions.BEFORE_AFFINE:
            return self.input_dim
        return self.output_dim

    def _adapt_and_retrieve(
        self,
        logits: Tensor,
        memory_model: "Layer | Sequential",
        decoder: "Layer | Sequential",
    ) -> Tensor:
        from torch.func import functional_call

        params = {k: v.clone() for k, v in memory_model.named_parameters()}

        for _ in range(self.test_time_training_num_inner_steps):
            memory = functional_call(memory_model, params, (logits,))
            reconstruction = decoder(memory)
            loss = F.mse_loss(reconstruction, logits.detach())
            grads = torch.autograd.grad(
                loss,
                list(params.values()),
                create_graph=self.training,
            )
            params = {
                k: p - self.test_time_training_learning_rate * g
                for (k, p), g in zip(params.items(), grads)
            }

        return functional_call(memory_model, params, (logits,))


@DynamicMemoryAbstract.register(DynamicMemoryOptions.FUSION)
class GatedResidualDynamicMemory(DynamicMemoryAbstract):
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
        if self.test_time_training_flag:
            memory = self._adapt_and_retrieve(
                logits, self.memory_model, self.memory_decoder
            )
        else:
            memory = self.memory_model(logits)
        memory_gate = self.__compute_memory_gate(logits, memory)
        return logits + memory_gate * memory

    def __compute_memory_gate(self, logits: Tensor, memory: Tensor) -> Tensor:
        combined = torch.cat([logits, memory], dim=-1)
        gate_logits = self.memory_gate_model(combined)
        return torch.sigmoid(gate_logits)


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


@DynamicMemoryAbstract.register(DynamicMemoryOptions.ELEMENT_WISE_WEIGHTED)
class ElementWiseWeightedDynamicMemory(DynamicMemoryAbstract):
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
            output_dim=self.memory_dim,
        )
        return self._init_model(layer_overrides)

    def forward(self, logits: Tensor) -> Tensor:
        if self.test_time_training_flag:
            memory = self._adapt_and_retrieve(
                logits, self.memory_model, self.memory_decoder
            )
        else:
            memory = self.memory_model(logits)
        feature_weights = self.__compute_feature_weights(logits, memory)
        return self.__blend_inputs_and_memory(logits, memory, feature_weights)

    def __compute_feature_weights(self, logits: Tensor, memory: Tensor) -> Tensor:
        combined = torch.cat([logits, memory], dim=-1)
        weight_logits = self.memory_weight_model(combined)
        return torch.sigmoid(weight_logits)

    def __blend_inputs_and_memory(
        self,
        logits: Tensor,
        memory: Tensor,
        feature_weights: Tensor,
    ) -> Tensor:
        return (1 - feature_weights) * logits + feature_weights * memory


@DynamicMemoryAbstract.register(DynamicMemoryOptions.ATTENTION)
class AttentionDynamicMemory(DynamicMemoryAbstract):
    def __init__(
        self,
        cfg: DynamicMemoryConfig,
        overrides: DynamicMemoryConfig | None = None,
    ):
        super().__init__(cfg, overrides)
        self.__validate_num_memory_slots()
        self.memory_model = self.__init_memory_model()
        self.memory_decoder = (
            self.__init_memory_decoder() if self.test_time_training_flag else None
        )
        self.query_model = self.__init_query_model()
        self.key_model = self.__init_key_model()
        self.value_model = self.__init_value_model()
        self.output_model = self.__init_output_model()
        self.memory_gate_model = self.__init_memory_gate_model()

    def __validate_num_memory_slots(self) -> None:
        if self.num_memory_slots is None:
            raise ValueError(
                "num_memory_slots is required for AttentionDynamicMemory."
            )
        if not isinstance(self.num_memory_slots, int):
            raise TypeError(
                "num_memory_slots must be an integer for AttentionDynamicMemory."
            )
        if self.num_memory_slots <= 0:
            raise ValueError(
                "num_memory_slots must be greater than 0 for AttentionDynamicMemory."
            )

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
        if self.test_time_training_flag:
            memory_bank_flat = self._adapt_and_retrieve(
                logits, self.memory_model, self.memory_decoder
            )
        else:
            memory_bank_flat = self.memory_model(logits)
        memory_bank = self.__reshape_memory_bank(memory_bank_flat)
        attended_memory = self.__attend_to_memory_bank(logits, memory_bank)
        memory_update = self.output_model(attended_memory)
        memory_gate = self.__compute_memory_gate(logits, memory_update)
        return logits + memory_gate * memory_update

    def __reshape_memory_bank(self, memory_bank_flat: Tensor) -> Tensor:
        return memory_bank_flat.reshape(
            *memory_bank_flat.shape[:-1], self.num_memory_slots, self.memory_dim
        )

    def __attend_to_memory_bank(self, logits: Tensor, memory_bank: Tensor) -> Tensor:
        query = self.query_model(logits).unsqueeze(-2)
        keys = self.key_model(memory_bank)
        values = self.value_model(memory_bank)
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
        gate_logits = self.memory_gate_model(combined)
        return torch.sigmoid(gate_logits)

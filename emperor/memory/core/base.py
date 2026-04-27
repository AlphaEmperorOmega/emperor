import torch

from torch import Tensor
from torch.nn import Sequential
from torch.nn import functional as F

from emperor.base.layer import Layer, LayerStackConfig
from emperor.base.registry import subclass_registry
from emperor.base.utils import Module
from emperor.memory.config import DynamicMemoryConfig
from emperor.memory.core._validator import DynamicMemoryValidator
from emperor.memory.options import MemoryPositionOptions


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

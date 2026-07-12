import torch
from torch import Tensor
from torch.nn import Sequential
from torch.nn import functional as F

from emperor.base.layer import Layer, LayerStack, LayerStackConfig, LayerState
from emperor.base.module import Module
from emperor.memory.config import DynamicMemoryConfig
from emperor.memory.core._validator import DynamicMemoryValidator
from emperor.memory.options import MemoryPositionOptions


class DynamicMemoryAbstract(Module):
    def __init__(
        self,
        cfg: DynamicMemoryConfig,
        overrides: DynamicMemoryConfig | None = None,
    ):
        super().__init__()
        config = getattr(cfg, "memory_config", cfg)
        self.cfg: DynamicMemoryConfig = self._override_config(config, overrides)
        DynamicMemoryValidator.validate(self)
        self.input_dim = self.cfg.input_dim
        self.output_dim = self.cfg.output_dim
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

    def _init_model(self, overrides: LayerStackConfig) -> "Layer | LayerStack":
        return self._build_generator_model(overrides)

    def _build_generator_with_dims(
        self,
        *,
        input_dim: int,
        output_dim: int,
        hidden_dim: int | None = None,
        validate_test_time_training_target: bool = False,
    ) -> "Layer | LayerStack | Sequential":
        overrides = LayerStackConfig(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
        )
        generator_model = self._build_generator_model(overrides)
        if validate_test_time_training_target:
            DynamicMemoryValidator.validate_test_time_training_generator_model(
                generator_model
            )
        return generator_model

    def _build_generator_model(
        self,
        overrides: LayerStackConfig,
    ) -> "Layer | LayerStack | Sequential":
        generator_model = self.model_config.build(overrides)
        DynamicMemoryValidator.validate_generator_model(generator_model)
        return generator_model

    def _run_model(
        self,
        model: "Layer | LayerStack | Sequential",
        inputs: Tensor,
    ) -> Tensor:
        leading_shape = inputs.shape[:-1]
        flat_inputs = inputs.reshape(-1, inputs.shape[-1])
        if isinstance(model, (Layer, Sequential, LayerStack)):
            output = model(LayerState(hidden=flat_inputs))
            flat_output = self._extract_model_hidden(output)
            return flat_output.reshape(*leading_shape, flat_output.shape[-1])

        flat_output = model(flat_inputs)
        if not torch.is_tensor(flat_output):
            raise TypeError(
                "DynamicMemory generator models must return a Tensor or LayerState, "
                f"received {type(flat_output).__name__}."
            )
        return flat_output.reshape(*leading_shape, flat_output.shape[-1])

    def _functional_run_model(
        self,
        model: "Layer | LayerStack | Sequential",
        params: dict[str, Tensor],
        inputs: Tensor,
    ) -> Tensor:
        from torch.func import functional_call

        if not isinstance(model, (Layer, Sequential, LayerStack)):
            raise TypeError(
                "Test-time-training memory supports only Layer, Sequential, or "
                f"LayerStack generators, received {type(model).__name__}."
            )
        leading_shape = inputs.shape[:-1]
        flat_inputs = inputs.reshape(-1, inputs.shape[-1])
        output_state = functional_call(model, params, (LayerState(hidden=flat_inputs),))
        flat_output = self._extract_model_hidden(output_state)
        return flat_output.reshape(*leading_shape, flat_output.shape[-1])

    def _extract_model_hidden(self, output: object) -> Tensor:
        if torch.is_tensor(output):
            return output
        hidden = getattr(output, "hidden", None)
        if torch.is_tensor(hidden):
            return hidden
        raise TypeError(
            "DynamicMemory generator models must return a Tensor or LayerState "
            f"with Tensor hidden, received {type(output).__name__}."
        )

    def __get_memory_dim(self) -> int:
        if self.memory_position_option == MemoryPositionOptions.BEFORE_AFFINE:
            return self.input_dim
        return self.output_dim

    def _adapt_and_retrieve(
        self,
        logits: Tensor,
        memory_model: "Layer | LayerStack",
        decoder: "Layer | LayerStack",
    ) -> Tensor:
        with torch.inference_mode(False), torch.enable_grad():
            if logits.is_inference():
                logits = logits.clone()
            params = {k: v.clone() for k, v in memory_model.named_parameters()}

            for _ in range(self.test_time_training_num_inner_steps):
                memory = self._functional_run_model(memory_model, params, logits)
                reconstruction = self._run_model(decoder, memory)
                loss = F.mse_loss(reconstruction, logits.detach())
                grads = torch.autograd.grad(
                    loss,
                    list(params.values()),
                    create_graph=self.training,
                )
                params = {
                    k: p - self.test_time_training_learning_rate * g
                    for (k, p), g in zip(params.items(), grads, strict=True)
                }

            memory = self._functional_run_model(memory_model, params, logits)
            if not self.training:
                memory = memory.detach()
            return memory

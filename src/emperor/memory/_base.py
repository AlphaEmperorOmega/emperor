import torch
from torch import Tensor
from torch.nn import Sequential
from torch.nn import functional as F

from emperor.layers import Layer, LayerStack, LayerStackConfig, LayerState
from emperor.memory._config import DynamicMemoryConfig, MemoryPositionOptions
from emperor.memory._validation import DynamicMemoryValidator
from emperor.nn import Module


class DynamicMemoryAbstract(Module):
    VALIDATOR = DynamicMemoryValidator

    def __init__(
        self,
        cfg: DynamicMemoryConfig,
        overrides: DynamicMemoryConfig | None = None,
    ):
        super().__init__()
        memory_config = getattr(cfg, "memory_config", cfg)
        self.cfg: DynamicMemoryConfig = self._override_config(memory_config, overrides)
        self.VALIDATOR.validate(self)
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
        self.test_time_training_flag = self.test_time_training_learning_rate is not None

    def _init_model(self, overrides: LayerStackConfig) -> "Layer | LayerStack":
        return self._build_generator_model(overrides)

    def _build_generator_with_dims(
        self,
        *,
        input_dim: int,
        output_dim: int,
        hidden_dim: int | None = None,
    ) -> "Layer | LayerStack | Sequential":
        generator_dimension_overrides = LayerStackConfig(
            input_dim=input_dim,
            output_dim=output_dim,
        )
        if hidden_dim is not None:
            generator_dimension_overrides.hidden_dim = hidden_dim
        return self._build_generator_model(generator_dimension_overrides)

    def _build_memory_generator_with_dims(
        self,
        *,
        input_dim: int,
        output_dim: int,
    ) -> "Layer | LayerStack | Sequential":
        generator_model = self._build_generator_with_dims(
            input_dim=input_dim,
            output_dim=output_dim,
        )
        if self.test_time_training_flag:
            self.VALIDATOR.validate_test_time_training_generator_model(generator_model)
        return generator_model

    def _build_generator_model(
        self,
        overrides: LayerStackConfig,
    ) -> "Layer | LayerStack | Sequential":
        generator_model = self.model_config.build(overrides)
        self.VALIDATOR.validate_generator_model(generator_model)
        return generator_model

    def _run_model(
        self,
        model: "Layer | LayerStack | Sequential",
        inputs: Tensor,
    ) -> Tensor:
        input_leading_shape = inputs.shape[:-1]
        input_feature_count = inputs.shape[-1]
        flattened_inputs = inputs.reshape(-1, input_feature_count)
        flattened_leading_dimension = 0
        if isinstance(model, (Layer, Sequential, LayerStack)):
            generator_output = model(LayerState(hidden=flattened_inputs))
            flattened_output_values = self._extract_model_hidden(generator_output)
            restored_output_values = flattened_output_values.unflatten(
                flattened_leading_dimension,
                input_leading_shape,
            )
            return restored_output_values

        flattened_output_values = model(flattened_inputs)
        if not torch.is_tensor(flattened_output_values):
            raise TypeError(
                "DynamicMemory generator models must return a Tensor or LayerState, "
                f"received {type(flattened_output_values).__name__}."
            )
        restored_output_values = flattened_output_values.unflatten(
            flattened_leading_dimension,
            input_leading_shape,
        )
        return restored_output_values

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
        input_leading_shape = inputs.shape[:-1]
        input_feature_count = inputs.shape[-1]
        flattened_inputs = inputs.reshape(-1, input_feature_count)
        functional_input_state = LayerState(hidden=flattened_inputs)
        functional_output = functional_call(model, params, (functional_input_state,))
        flattened_output_values = self._extract_model_hidden(functional_output)
        flattened_leading_dimension = 0
        restored_output_values = flattened_output_values.unflatten(
            flattened_leading_dimension,
            input_leading_shape,
        )
        return restored_output_values

    def _extract_model_hidden(self, output: object) -> Tensor:
        if torch.is_tensor(output):
            return output
        hidden_output = getattr(output, "hidden", None)
        if torch.is_tensor(hidden_output):
            return hidden_output
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
            adaptation_logits = logits
            if logits.is_inference():
                adaptation_logits = logits.clone()
            batch_size = len(adaptation_logits)
            if batch_size == 0:
                adapted_memory_parameters = {
                    parameter_name: memory_parameter.clone()
                    for parameter_name, memory_parameter in (
                        memory_model.named_parameters()
                    )
                }
                retrieved_memory = self._functional_run_model(
                    memory_model,
                    adapted_memory_parameters,
                    adaptation_logits,
                )
            else:
                per_sample_logits = adaptation_logits.split(1)
                adapted_sample_memories = [
                    self.__adapt_single_sample(
                        sample_logits,
                        memory_model,
                        decoder,
                    )
                    for sample_logits in per_sample_logits
                ]
                retrieved_memory = torch.cat(adapted_sample_memories)
            if not self.training:
                retrieved_memory = retrieved_memory.detach()
            return retrieved_memory

    def __adapt_single_sample(
        self,
        logits: Tensor,
        memory_model: "Layer | LayerStack",
        decoder: "Layer | LayerStack",
    ) -> Tensor:
        adapted_memory_parameters = {
            parameter_name: memory_parameter.clone()
            for parameter_name, memory_parameter in memory_model.named_parameters()
        }
        for _ in range(self.test_time_training_num_inner_steps):
            sample_memory = self._functional_run_model(
                memory_model,
                adapted_memory_parameters,
                logits,
            )
            reconstructed_logits = self._run_model(decoder, sample_memory)
            reconstruction_targets = logits.detach()
            reconstruction_loss = F.mse_loss(
                reconstructed_logits,
                reconstruction_targets,
            )
            memory_parameters_for_gradient = list(adapted_memory_parameters.values())
            memory_parameter_gradients = torch.autograd.grad(
                reconstruction_loss,
                memory_parameters_for_gradient,
                create_graph=self.training,
            )
            updated_memory_parameters = {}
            for parameter_index, (parameter_name, memory_parameter) in enumerate(
                adapted_memory_parameters.items()
            ):
                scaled_parameter_gradient = (
                    self.test_time_training_learning_rate
                    * memory_parameter_gradients[parameter_index]
                )
                updated_memory_parameter = memory_parameter - scaled_parameter_gradient
                updated_memory_parameters[parameter_name] = updated_memory_parameter
            adapted_memory_parameters = updated_memory_parameters
        adapted_sample_memory = self._functional_run_model(
            memory_model,
            adapted_memory_parameters,
            logits,
        )
        return adapted_sample_memory

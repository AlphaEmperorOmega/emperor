from torch import Tensor
from Emperor.base.layer import Layer, LayerStack, LayerStackConfig

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.config import ModelConfig


class MixtureOfExpertsStack(LayerStack):
    def __init__(
        self,
        cfg: "LayerStackConfig | ModelConfig",
        overrides: "LayerStackConfig | None" = None,
    ):
        overrides = self.__get_model_type(overrides)
        super().__init__(cfg, overrides)

    def __get_model_type(self, overrides: "LayerStackConfig") -> "LayerStackConfig":
        from Emperor.experts.options import MixtureOfExpertsOptions

        model = MixtureOfExpertsOptions.BASE
        layer = MixtureOfExpertsLayer
        return super()._override_model_type(overrides, model, layer)


class MixtureOfExpertsLayer(Layer):
    def __init__(
        self,
        model: "Module",
        activation_function: "ActivationOptions | None" = None,
        layer_norm_dim: int | None = None,
        residual_connection_flag: bool = False,
        is_adaptive_computation: bool = False,
        dropout_probability: float = 0.0,
        layer_norm_position: "LayerNormPositionOptions | None" = None,
    ):
        super().__init__(
            model,
            activation_function,
            layer_norm_dim,
            residual_connection_flag,
            is_adaptive_computation,
            dropout_probability,
            layer_norm_position,
        )

        self.loss = None
        self.probabilities = None
        self.indices = None

    def _handle_model_input(self, model_inputs: tuple) -> Tensor:
        main_model_input, probabilities, indices, previous_loss = model_inputs
        self.probabilities = probabilities
        self.indices = indices
        self.loss = previous_loss
        return main_model_input

    def _handle_model_processing(self, model_input: Tensor) -> Tensor:
        model_output, total_loss = self.model(
            model_input, self.probabilities, self.indices
        )
        self.loss += total_loss
        return model_output

    def _handle_model_output(self, output: Tensor) -> tuple:
        return output, self.probabilities, self.indices, self.loss

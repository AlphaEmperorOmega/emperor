from torch import Tensor
from emperor.base.layer import Layer, LayerStack, LayerStackConfig

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.base.utils import Module
    from emperor.config import ModelConfig
    from emperor.base.enums import ActivationOptions
    from emperor.base.enums import LayerNormPositionOptions


class MixtureOfExpertsStack(LayerStack):
    def __init__(
        self,
        cfg: "LayerStackConfig | ModelConfig",
        overrides: "LayerStackConfig | None" = None,
    ):
        overrides = self.__get_model_type(overrides)
        super().__init__(cfg, overrides)

    def __get_model_type(self, overrides: "LayerStackConfig") -> "LayerStackConfig":
        from emperor.experts.options import MixtureOfExpertsOptions

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

        self.probabilities = None
        self.indices = None
        self.total_loss = None

    def __reset_properties(self) -> None:
        self.total_loss = None

    def _handle_model_input(self, model_inputs: dict | Tensor) -> Tensor:
        if isinstance(model_inputs, Tensor):
            return model_inputs
        if isinstance(model_inputs, tuple):
            model_inputs, total_loss = model_inputs
            if self.total_loss is None:
                self.total_loss = total_loss
            else:
                self.total_loss = self.total_loss + total_loss
            return model_inputs
        input_batch = model_inputs["input_batch"]
        self.probabilities = model_inputs["probabilities"]
        self.indices = model_inputs["indices"]
        self.total_loss = model_inputs["loss"]
        return input_batch

    def _handle_model_processing(self, model_input: Tensor) -> Tensor:
        model_output, total_loss = self.model(
            model_input, self.probabilities, self.indices
        )
        if self.total_loss is None:
            self.total_loss = total_loss
        else:
            self.total_loss = self.total_loss + total_loss
        return model_output

    def _handle_model_output(self, output: Tensor) -> tuple | dict:
        total_loss = self.total_loss
        self.__reset_properties()
        if self.last_layer_flag or self.probabilities is None:
            return output, total_loss
        return {
            "input_batch": output,
            "probabilities": self.probabilities,
            "indices": self.indices,
            "loss": total_loss,
        }

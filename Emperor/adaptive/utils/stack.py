from torch.types import Tensor
from Emperor.base.utils import Module
from Emperor.base.enums import ActivationOptions, LayerNormPositionOptions
from Emperor.base.layer import Layer, LayerStack, LayerStackConfig

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.linears.options import LinearLayerOptions


class AdaptiveParameterLayerStack(LayerStack):
    def __init__(
        self,
        cfg: "LayerStackConfig",
        overrides: "LayerStackConfig | None" = None,
    ):
        overrides = self.__get_model_type(overrides)
        super().__init__(cfg, overrides)

    def __get_model_type(
        self, overrides: "LayerStackConfig | None"
    ) -> "LinearLayerOptions":
        from Emperor.adaptive.options import AdaptiveLayerOptions

        model = AdaptiveLayerOptions.BASE
        layer = AdaptiveParameterLayerHandler
        return super()._override_model_type(overrides, model, layer)


class AdaptiveParameterLayerHandler(Layer):
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

        self.skip_mask = None
        self.total_loss = None

    def __reset_properties(self) -> None:
        self.total_loss = None
        self.skip_mask = None

    def _handle_model_input(self, model_inputs: dict | Tensor) -> Tensor:
        if isinstance(model_inputs, Tensor):
            return model_inputs

        if isinstance(model_inputs, tuple):
            input_batch, skip_mask = model_inputs
            self.skip_mask = skip_mask
            return input_batch

        input_batch = model_inputs["input_batch"]
        self.skip_mask = model_inputs["skip_mask"]
        self.total_loss = model_inputs["loss"]

        return input_batch

    def _handle_model_processing(self, model_input: Tensor) -> Tensor:
        model_output, skip_mask, total_loss = self.model(model_input, self.skip_mask)
        self.skip_mask = skip_mask
        if self.total_loss is None:
            self.total_loss = total_loss
        else:
            self.total_loss = self.total_loss + total_loss
        return model_output

    def _handle_model_output(self, output: Tensor) -> tuple | dict:
        total_loss = self.total_loss
        self.__reset_properties()
        if self.last_layer_flag:
            return output, self.skip_mask, total_loss
        return {
            "input_batch": output,
            "skip_mask": self.skip_mask,
            "loss": total_loss,
        }

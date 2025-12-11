import torch

from torch.types import Tensor
from Emperor.base.enums import ActivationOptions, LayerNormPositionOptions
from Emperor.base.layer import Layer
from Emperor.base.utils import Module


class ParameterGeneratorLayer(Layer):
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

        self.loss = torch.tensor(0.0)

    def _handle_model_input(
        self, main_model_input: Tensor | tuple[Tensor, Tensor]
    ) -> Tensor:
        if isinstance(main_model_input, tuple):
            main_model_input, previous_loss = main_model_input
            self.loss = previous_loss
        return main_model_input

    def _handle_model_processing(
        self,
        main_model_input: Tensor,
        additional_model_inputs: dict,
    ) -> Tensor:
        model_output = self.model(
            main_model_input,
            **additional_model_inputs,
        )
        output, skip_mask, loss = model_output
        self.loss = self.loss + loss
        return output

    def _handle_output(self, output: Tensor) -> tuple[Tensor, Tensor]:
        return output, self.loss


class SelfAttentionLayer(Layer):
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

        self.loss = torch.tensor(0.0)

    def _handle_model_input(
        self, main_model_input: Tensor | tuple[Tensor, Tensor]
    ) -> Tensor:
        if isinstance(main_model_input, tuple):
            main_model_input, previous_loss = main_model_input
            self.loss = previous_loss
        return main_model_input

    def _handle_model_processing(
        self,
        main_model_input: Tensor,
        additional_model_inputs: dict,
    ) -> Tensor:
        model_output = self.model(
            main_model_input,
            main_model_input,
            main_model_input,
            **additional_model_inputs,
        )
        attention_output, attention_weights = model_output
        return attention_output

    def _handle_output(self, output: Tensor) -> tuple[Tensor, Tensor]:
        return output, self.loss


class CrossAttentionLayer(Layer):
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

        self.loss = torch.tensor(0.0)

    def _handle_model_input(
        self, main_model_input: Tensor | tuple[Tensor, Tensor]
    ) -> Tensor:
        if isinstance(main_model_input, tuple):
            main_model_input, previous_loss = main_model_input
            self.loss = previous_loss
        return main_model_input

    def _handle_model_processing(
        self,
        main_model_input: Tensor,
        additional_model_inputs: dict,
    ) -> Tensor:
        model_output = self.model(
            main_model_input,
            **additional_model_inputs,
        )
        attention_output, attention_weights = model_output
        self.loss = self.loss
        return attention_output

    def _handle_output(self, output: Tensor) -> tuple[Tensor, Tensor]:
        return output, self.loss


class FeedForwardLayer(Layer):
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

        self.loss = torch.tensor(0.0)

    def _handle_model_input(
        self, main_model_input: Tensor | tuple[Tensor, Tensor]
    ) -> Tensor:
        if isinstance(main_model_input, tuple):
            main_model_input, previous_loss = main_model_input
            self.loss = previous_loss
        return main_model_input

    def _handle_model_processing(
        self,
        main_model_input: Tensor,
        additional_model_inputs: dict,
    ) -> Tensor:
        model_output = self.model(
            main_model_input,
            **additional_model_inputs,
        )
        output, loss = model_output
        self.loss = self.loss + loss
        return output

    def _handle_output(self, output: Tensor) -> tuple[Tensor, Tensor]:
        return output, self.loss

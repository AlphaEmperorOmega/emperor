from torch import Tensor
from Emperor.base.layer import Layer
from Emperor.base.utils import Module
from Emperor.base.enums import ActivationOptions, LayerNormPositionOptions


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

        self.k_padding_mask = None
        self.attention_mask = None
        self.total_loss = None

    def __reset_properties(self) -> None:
        self.k_padding_mask = None
        self.attention_mask = None
        self.total_loss = None

    def _handle_model_input(self, model_input: dict) -> Tensor:
        qkv_tensor = model_input.get("q")
        self.k_padding_mask = model_input.get("k_padding_mask", None)
        self.attention_mask = model_input.get("attention_mask")
        return qkv_tensor

    def _handle_model_processing(self, target_token_embeddings: Tensor) -> Tensor:
        model_output, total_loss = self.model(
            q=target_token_embeddings,
            k=target_token_embeddings,
            v=target_token_embeddings,
            k_padding_mask=self.k_padding_mask,
            attention_mask=self.attention_mask,
        )
        self.__update_loss(total_loss)
        return model_output

    def __update_loss(self, loss: Tensor) -> None:
        if self.total_loss is None:
            self.total_loss = loss
        else:
            self.total_loss = self.total_loss + loss

    def _handle_model_output(self, output: Tensor) -> tuple[Tensor, Tensor]:
        total_loss = self.total_loss
        self.__reset_properties()
        return output, total_loss


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
            activation_function=activation_function,
            layer_norm_dim=layer_norm_dim,
            residual_connection_flag=residual_connection_flag,
            is_adaptive_computation=is_adaptive_computation,
            dropout_probability=dropout_probability,
            layer_norm_position=layer_norm_position,
        )

        self.k_encoder_tensor = None
        self.v_encoder_tensor = None
        self.k_padding_mask = None
        self.attention_mask = None
        self.total_loss = None

    def __reset_properties(self) -> None:
        self.k_encoder_tensor = None
        self.v_encoder_tensor = None
        self.k_padding_mask = None
        self.attention_mask = None
        self.total_loss = None

    def _handle_model_input(self, model_input: dict) -> Tensor:
        q_tensor = model_input.get("q")
        self.k_encoder_tensor = model_input.get("k")
        self.v_encoder_tensor = model_input.get("v")
        self.k_padding_mask = model_input.get("k_padding_mask", None)
        self.attention_mask = model_input.get("attention_mask")
        return q_tensor

    def _handle_model_processing(self, target_token_embeddings: Tensor) -> Tensor:
        model_output, total_loss = self.model(
            q=target_token_embeddings,
            k=self.k_encoder_tensor,
            v=self.v_encoder_tensor,
            k_padding_mask=self.k_padding_mask,
            attention_mask=self.attention_mask,
        )
        self.__update_loss(total_loss)
        return model_output

    def __update_loss(self, loss: Tensor) -> None:
        if self.total_loss is None:
            self.total_loss = loss
        else:
            self.total_loss = self.total_loss + loss

    def _handle_model_output(self, output: Tensor) -> tuple[Tensor, Tensor]:
        total_loss = self.total_loss
        self.__reset_properties()
        return output, total_loss


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
            activation_function=activation_function,
            layer_norm_dim=layer_norm_dim,
            residual_connection_flag=residual_connection_flag,
            is_adaptive_computation=is_adaptive_computation,
            dropout_probability=dropout_probability,
            layer_norm_position=layer_norm_position,
        )

        self.total_loss = None

    def __reset_properties(self) -> None:
        self.total_loss = None

    def _handle_model_processing(self, x: Tensor) -> Tensor:
        model_output, total_loss = self.model(x)
        self.__update_loss(total_loss)
        return model_output

    def __update_loss(self, loss: Tensor) -> None:
        if self.total_loss is None:
            self.total_loss = loss
        else:
            self.total_loss = self.total_loss + loss

    def _handle_model_output(self, output: Tensor) -> tuple[Tensor, Tensor]:
        total_loss = self.total_loss
        self.__reset_properties()
        return output, total_loss

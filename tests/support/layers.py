import math
from dataclasses import dataclass

import torch

from emperor.config import ConfigBase, optional_field
from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    Layer,
    LayerConfig,
    LayerNormPositionOptions,
    LayerStack,
    LayerStackConfig,
    LayerState,
    RecurrentLayerConfig,
    ResidualConfig,
)
from emperor.linears import LinearLayerConfig
from emperor.memory import MemoryPositionOptions, WeightedDynamicMemoryConfig
from emperor.nn import Module


def linear_stack_config(
    dim: int,
    *,
    input_dim: int | None = None,
    output_dim: int | None = None,
    bias_flag: bool = False,
) -> LayerStackConfig:
    resolved_input_dim = dim if input_dim is None else input_dim
    resolved_output_dim = dim if output_dim is None else output_dim
    return LayerStackConfig(
        input_dim=resolved_input_dim,
        hidden_dim=max(resolved_input_dim, resolved_output_dim),
        output_dim=resolved_output_dim,
        num_layers=1,
        last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
        apply_output_pipeline_flag=False,
        layer_config=LayerConfig(
            input_dim=resolved_input_dim,
            output_dim=resolved_output_dim,
            activation=ActivationOptions.DISABLED,
            residual_config=None,
            dropout_probability=0.0,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            gate_config=None,
            halting_config=None,
            memory_config=None,
            layer_model_config=LinearLayerConfig(
                input_dim=resolved_input_dim,
                output_dim=resolved_output_dim,
                bias_flag=bias_flag,
            ),
        ),
    )


def base_layer_config(
    dim: int = 2,
    *,
    activation: ActivationOptions = ActivationOptions.DISABLED,
    memory_config: WeightedDynamicMemoryConfig | None = None,
) -> LayerConfig:
    return LayerConfig(
        input_dim=dim,
        output_dim=dim,
        activation=activation,
        residual_config=None,
        dropout_probability=0.0,
        layer_norm_position=LayerNormPositionOptions.DISABLED,
        gate_config=None,
        halting_config=None,
        memory_config=memory_config,
        layer_model_config=LinearLayerConfig(
            input_dim=dim,
            output_dim=dim,
            bias_flag=False,
        ),
    )


def weighted_memory_config(
    dim: int = 2,
    position: MemoryPositionOptions = MemoryPositionOptions.AFTER_AFFINE,
) -> WeightedDynamicMemoryConfig:
    return WeightedDynamicMemoryConfig(
        input_dim=dim,
        output_dim=dim,
        memory_position_option=position,
        test_time_training_learning_rate=None,
        test_time_training_num_inner_steps=None,
        model_config=linear_stack_config(dim, bias_flag=True),
    )


def _only_layer(model: Layer | LayerStack) -> Layer:
    if isinstance(model, Layer):
        return model
    if len(model) != 1:
        raise AssertionError("Expected a one-layer stack.")
    return model[0]


def _set_affine_parameters(
    model: Layer | LayerStack,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> None:
    layer = _only_layer(model)
    with torch.no_grad():
        layer.model.weight_params.copy_(
            weight.to(
                device=layer.model.weight_params.device,
                dtype=layer.model.weight_params.dtype,
            )
        )
        if layer.model.bias_params is None:
            if bias is not None:
                raise AssertionError("The layer has no bias parameters.")
            return
        if bias is None:
            layer.model.bias_params.zero_()
        else:
            layer.model.bias_params.copy_(
                bias.to(
                    device=layer.model.bias_params.device,
                    dtype=layer.model.bias_params.dtype,
                )
            )


def configure_weighted_memory(memory_model: Module) -> None:
    memory_layer = _only_layer(memory_model.memory_model)
    input_dim, output_dim = memory_layer.model.weight_params.shape
    if input_dim != output_dim:
        raise AssertionError("Scaled identity requires equal dimensions.")
    _set_affine_parameters(
        memory_model.memory_model,
        torch.eye(input_dim) * 2.0,
        torch.zeros(output_dim) if memory_layer.model.bias_params is not None else None,
    )

    weight_layer = _only_layer(memory_model.memory_weight_model)
    _set_affine_parameters(
        memory_model.memory_weight_model,
        torch.zeros_like(weight_layer.model.weight_params),
        torch.tensor([0.0, math.log(3.0)]),
    )


def set_layer_identity(layer: Layer) -> None:
    with torch.no_grad():
        layer.model.weight_params.copy_(
            torch.eye(
                layer.input_dim,
                layer.output_dim,
                device=layer.model.weight_params.device,
                dtype=layer.model.weight_params.dtype,
            )
        )
        if layer.model.bias_params is not None:
            layer.model.bias_params.zero_()


@dataclass
class IdentityBlockConfig(ConfigBase):
    input_dim: int | None = optional_field("Input feature dimension.")
    output_dim: int | None = optional_field("Output feature dimension.")

    def _registry_owner(self) -> type:
        return IdentityBlock


class IdentityBlock(Module):
    def __init__(
        self,
        cfg: IdentityBlockConfig,
        overrides: IdentityBlockConfig | None = None,
    ) -> None:
        super().__init__()
        self.cfg = self._override_config(cfg, overrides)

    def forward(self, state: LayerState) -> LayerState:
        return state


def recurrent_config(
    *,
    recurrent_layer_norm_position: LayerNormPositionOptions = (
        LayerNormPositionOptions.DISABLED
    ),
    residual_connection_option: object | None = None,
    halting_config: object | None = None,
    memory_config: object | None = None,
) -> RecurrentLayerConfig:
    return RecurrentLayerConfig(
        input_dim=2,
        output_dim=2,
        max_steps=1,
        recurrent_layer_norm_position=recurrent_layer_norm_position,
        block_config=IdentityBlockConfig(input_dim=2, output_dim=2),
        gate_config=None,
        residual_config=(
            None
            if residual_connection_option is None
            else ResidualConfig(option=residual_connection_option)
        ),
        halting_config=halting_config,
        memory_config=memory_config,
    )

import copy
import torch.nn as nn

from typing import Self
from torch.types import Tensor
from torch.nn import Linear, Sequential
from dataclasses import dataclass, field
from Emperor.base.enums import ActivationOptions, BaseOptions, LayerNormPositionOptions
from Emperor.base.utils import ConfigBase, Module

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emperor.config import ModelConfig
    from Emperor.linears.options import LinearLayerOptions


@dataclass
class LayerConfig(ConfigBase):
    model_type: "LinearLayerOptions | None" = field(
        default=None,
        metadata={"help": "Linear model module used for output transformation"},
    )
    activation: ActivationOptions | None = field(
        default=None,
        metadata={"help": "Activation function or layer to use"},
    )
    layer_norm_dim: int | None = field(
        default=None,
        metadata={"help": ""},
    )
    residual_flag: bool | None = field(
        default=None,
        metadata={"help": ""},
    )
    adaptive_computation_flag: bool | None = field(
        default=None,
        metadata={"help": ""},
    )
    dropout_probability: float | None = field(
        default=None,
        metadata={"help": ""},
    )
    layer_norm_position: LayerNormPositionOptions | None = field(
        default=None,
        metadata={"help": ""},
    )
    apply_gates_bool: bool | None = field(
        default=None,
        metadata={"help": ""},
    )


class Layer(Module):
    def __init__(
        self,
        model: "Module",
        activation_function: "ActivationOptions | None" = None,
        layer_norm_dim: int | None = None,
        residual_connection_flag: bool = False,
        is_adaptive_computation: bool = False,
        dropout_probability: float = 0.0,
        layer_norm_position: "LayerNormPositionOptions | None" = None,
        apply_gates_bool: bool = False,
    ):
        super().__init__()

        self.model = model
        self.activation_function = activation_function
        self.layer_norm_dim = layer_norm_dim
        self.residual_connection_flag = residual_connection_flag
        self.is_adaptive_computation = is_adaptive_computation
        self.dropout_probability = dropout_probability
        self.layer_norm_position = layer_norm_position
        self.apply_gates_bool = apply_gates_bool

        self.has_activation = self.activation_function is not None
        self.has_dropout = self.dropout_probability > 0.0

        self.dropout_module = self.__init_dropout_module()
        self.layer_norm_module = self.__init_layer_norm_module()

    def __init_dropout_module(self) -> nn.Module | None:
        if self.has_dropout:
            return nn.Dropout(self.dropout_probability)
        return None

    def __init_layer_norm_module(self) -> nn.Module | None:
        if (
            self.layer_norm_position == LayerNormPositionOptions.NONE
            or self.layer_norm_dim is None
        ):
            return None
        assert self.layer_norm_dim > 0, (
            f"expected layer_norm_dim must be greater than 0, received {self.layer_norm_dim}"
        )
        return nn.LayerNorm(self.layer_norm_dim)

    def create_adaptive_computation_module(self):
        # TODO: Create a wrapper that first decides
        # if the current layer should be computed using the
        # adaptive computation from sparse universal transformer
        # paper
        pass

    def forward(
        self,
        main_model_input: Tensor,
        additional_model_inputs: dict = {},
        skip_mask: Tensor | None = None,
    ) -> Tensor | tuple[Tensor | None]:
        # TODO: Ensure that the skip_maks will be used
        # in the future.
        main_model_input = self._handle_model_input(main_model_input)

        if self.layer_norm_position == LayerNormPositionOptions.BEFORE:
            main_model_input = self.layer_norm_module(main_model_input)

        output = self._handle_model_output(main_model_input, additional_model_inputs)

        if self.layer_norm_position == LayerNormPositionOptions.DEFAULT:
            output = self.layer_norm_module(output)
        if self.has_activation:
            # TODO: Add the option to to redirect each sample in the
            # input batch to use a different activation function
            output = self.activation_function(output)
        if self.apply_gates_bool:
            # TODO: Finish the implementation of the gates module
            # output = self.gate_module(output) * output
            pass
        if self.has_dropout:
            output = self.dropout_module(output)
        if self.residual_connection_flag:
            output = output + main_model_input
        if self.layer_norm_position == LayerNormPositionOptions.AFTER:
            output = self.layer_norm_module(output)

        return self._handle_final_output(output)

    def _handle_model_input(self, main_model_input: Tensor):
        return main_model_input

    def _handle_model_output(
        self,
        main_model_input: Tensor,
        additional_model_inputs: dict,
    ) -> Tensor:
        return self.model(main_model_input, **additional_model_inputs)

    def _handle_final_output(self, output: Tensor) -> Tensor:
        return output

    # TODO: In the future instead multiple function inputs
    # use a dataset class to encapsulate the inputs and outputs
    # of the block
    # def forward(
    #     self,
    #     data: LayerData,
    # ) -> LayerData:
    #     output = self.model(data.tensor)


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

    def _handle_model_output(
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

    def _handle_final_output(self, output: Tensor) -> tuple[Tensor, Tensor]:
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

    def _handle_model_output(
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

    def _handle_final_output(self, output: Tensor) -> tuple[Tensor, Tensor]:
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

    def _handle_model_output(
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

    def _handle_final_output(self, output: Tensor) -> tuple[Tensor, Tensor]:
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

    def _handle_model_output(
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

    def _handle_final_output(self, output: Tensor) -> tuple[Tensor, Tensor]:
        return output, self.loss


@dataclass
class LayerStackConfig(LayerConfig):
    input_dim: int | None = field(
        default=None,
        metadata={"help": "Input dimension of the first `Linear` layer"},
    )
    hidden_dim: int | None = field(
        default=None,
        metadata={"help": "Dimension of the hidden `Linear` layers"},
    )
    output_dim: int | None = field(
        default=None,
        metadata={"help": "Output dimension of the output `Linear` layer"},
    )
    num_layers: int | None = field(
        default=None,
        metadata={"help": "Number of layers in the model"},
    )


class LayerStack(Module):
    SHARED_INPUT_OUTPUT_DIM = 1
    SEPARATE_INPUT_OUTPUT_DIM = 2

    def __init__(
        self,
        cfg: "LayerStackConfig | ModelConfig",
        overrides: "LayerStackConfig | None" = None,
    ):
        super().__init__()
        self.main_cfg = cfg
        config = getattr(cfg, "layer_stack_config", cfg)
        self.cfg: "LayerStackConfig" = self._overwrite_config(config, overrides)

        self.model_type = self.cfg.model_type
        self.activation = self.cfg.activation
        self.residual_flag = self.cfg.residual_flag
        self.adaptive_computation_flag = self.cfg.adaptive_computation_flag
        self.dropout_probability = self.cfg.dropout_probability
        self.layer_norm_position = self.cfg.layer_norm_position

        self.input_dim = self.cfg.input_dim
        self.hidden_dim = self.cfg.hidden_dim
        self.output_dim = self.cfg.output_dim
        self.num_layers = self.cfg.num_layers
        self.callback_function = None

        for _name, _val in (
            ("input_dim", self.input_dim),
            ("hidden_dim", self.hidden_dim),
            ("output_dim", self.output_dim),
            ("num_layers", self.num_layers),
        ):
            if not isinstance(_val, int) or _val < 1:
                raise ValueError(f"{_name} must be an integer >= 1, received {_val!r}")

        self.layer_block_model = self.__resolve_layer_block_class()

    def set_callback(self, callback) -> Self:
        self.callback_function = callback
        return self

    def __resolve_layer_block_class(self) -> type[Layer]:
        # TODO: move this somewhere else in the future since it is used in
        # `MixtureOfExperts` as well
        from Emperor.generators.utils.enums import (
            LinearLayerTypes,
            ParameterGeneratorTypes,
        )
        from Emperor.linears.options import LinearLayerOptions

        if (
            isinstance(self.model_type, LinearLayerTypes)
            or isinstance(self.model_type, LinearLayerOptions)
            or self.model_type == Linear
            or isinstance(self.model_type, BaseOptions)
        ):
            return Layer
        elif isinstance(self.model_type, ParameterGeneratorTypes):
            return ParameterGeneratorLayer
        else:
            raise RuntimeError(
                f"Unsupported `model_type` {type(self.model_type)} for `LayerStack`"
            )

    def build_model(self) -> Layer | Sequential:
        layers = []

        layer_adjustment = self.__add_initial_layer(layers)
        self.__add_hidden_layers(layers, layer_adjustment)
        self.__add_output_layer(layers)

        if len(layers) == 1:
            [model] = layers
            self._initialize_parameters(model)
            return model
        model = Sequential(*layers)
        self._initialize_parameters(model)
        return model

    def __add_initial_layer(self, layers: list) -> int:
        if self.input_dim != self.hidden_dim and self.num_layers > 1:
            layer = self.__create_layer(self.input_dim, self.hidden_dim, False)
            layers.append(layer)
            return self.SEPARATE_INPUT_OUTPUT_DIM
        return self.SHARED_INPUT_OUTPUT_DIM

    def __add_hidden_layers(self, layers: list, layer_adjustment: int) -> None:
        for _ in range(self.num_layers - layer_adjustment):
            layer = self.__create_layer(self.hidden_dim, self.hidden_dim)
            if self.callback_function is not None:
                layer = self.callback_function(layer)
            layers.append(layer)

    def __add_output_layer(self, layers: list) -> None:
        layer_input_dim = self.hidden_dim if self.num_layers > 1 else self.input_dim
        config = self.__resolve_model_type_overrides(layer_input_dim, self.output_dim)
        layer = self.layer_block_model(model=self.__get_model_type()(config))
        layers.append(layer)

    def __get_model_type(self):
        return self.model_type.value

    def __create_layer(
        self,
        input_dim: int,
        output_dim: int,
        residual_flag: bool = True,
    ) -> Layer:
        layer_norm_dim = None
        if self.layer_norm_position != LayerNormPositionOptions.NONE:
            layer_norm_dim = output_dim
        config = self.__resolve_model_type_overrides(input_dim, output_dim)
        model = self.__get_model_type()(config)

        return self.layer_block_model(
            model=model,
            layer_norm_dim=layer_norm_dim,
            residual_connection_flag=residual_flag,
            activation_function=self.activation,
            dropout_probability=self.dropout_probability,
            layer_norm_position=self.layer_norm_position,
        )

    def __resolve_model_type_overrides(self, input_dim: int, output_dim: int):
        # TODO: In the future find a way to get rid of this
        # and somehow create a configuration similar to how
        # in can write css in scss
        c = copy.deepcopy(self.main_cfg)
        linears = (
            "LinearLayer",
            "DynamicLinearLayer",
            "DepthMappingLayer",
        )
        if self.__get_model_type().__name__ in linears:
            c.linear_layer_config.input_dim = input_dim
            c.linear_layer_config.output_dim = output_dim
            return c

        generators = (
            "VectorParameterLayer",
            "MatrixParameterLayer",
            "GeneratorParameterLayer",
        )

        if self.__get_model_type().__name__ in generators:
            c.router_model_config.input_dim = input_dim
            c.mixture_model_config.input_dim = input_dim
            c.mixture_model_config.output_dim = output_dim

            return c


class LinearLayerStack(Module):
    def __init__(
        self,
        cfg: "LayerStackConfig | ModelConfig",
        overrides: "LayerStackConfig | None" = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.overrides = overrides
        self.identifier = "layer_stack_config"
        cfg = self.__override_config(overrides)
        self.model = LayerStack(cfg, overrides).build_model()

    def __override_config(
        self, overrides: "LayerStackConfig | None"
    ) -> LayerStackConfig:
        from Emperor.linears.options import LinearLayerOptions

        if overrides is None:
            return LayerStackConfig(model_type=LinearLayerOptions.BASE)
        overrides.model_type = LinearLayerOptions.BASE
        return overrides

    def forward(self, input_batch: Tensor) -> Tensor:
        return self.model(input_batch)

import copy
import torch.nn as nn

from typing import Self
from torch.types import Tensor
from torch.nn import Sequential
from dataclasses import dataclass, field
from Emperor.base.utils import ConfigBase, Module
from Emperor.base.enums import ActivationOptions, BaseOptions, LayerNormPositionOptions

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
        self.last_layer_flag = False

    def mark_as_last_layer(self) -> None:
        self.last_layer_flag = True

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
        return nn.LayerNorm(self.layer_norm_dim, device=self.device)

    def create_adaptive_computation_module(self):
        # TODO: Create a wrapper that first decides
        # if the current layer should be computed using the
        # adaptive computation from sparse universal transformer
        # paper
        pass

    def forward(
        self,
        model_input: Tensor | tuple,
        skip_mask: Tensor | None = None,
    ) -> Tensor | tuple[Tensor | None]:
        # TODO: Ensure that the skip_maks will be used
        # in the future.
        model_input = self._handle_model_input(model_input)
        output = self.__apply_layer_norm_before(model_input)
        output = self._handle_model_processing(output)
        output = self.__apply_layer_norm_default(output)
        output = self.__apply_activation(output)
        output = self.__apply_gates(output)
        output = self.__apply_dropout(output)
        output = self.__apply_residual_connection(output, model_input)
        output = self.__apply_layer_norm_after(output)
        return self._handle_model_output(output)

    def _handle_model_input(self, input: Tensor) -> Tensor:
        return input

    def __apply_layer_norm_before(self, input: Tensor):
        if self.layer_norm_position == LayerNormPositionOptions.BEFORE:
            return self.layer_norm_module(input)
        return input

    def _handle_model_processing(
        self,
        main_model_input: Tensor,
        additional_model_inputs: dict = {},
    ) -> Tensor:
        return self.model(main_model_input, **additional_model_inputs)

    def __apply_layer_norm_default(self, input: Tensor):
        if self.layer_norm_position == LayerNormPositionOptions.DEFAULT:
            return self.layer_norm_module(input)
        return input

    def __apply_activation(self, input: Tensor):
        # TODO: Add the option to to redirect each sample in the
        # input batch to use a different activation function
        if self.has_activation:
            return self.activation_function(input)
        return input

    def __apply_gates(self, input: Tensor):
        # TODO: Implement the gates option
        # if self.apply_gates_bool:
        #     return self.gate_module(output) * output
        return input

    def __apply_dropout(self, input: Tensor):
        if self.has_dropout:
            return self.dropout_module(input)
        return input

    def __apply_residual_connection(self, input: Tensor, prev_input: Tensor):
        if self.residual_connection_flag:
            return input + prev_input
        return input

    def __apply_layer_norm_after(self, input: Tensor):
        if self.layer_norm_position == LayerNormPositionOptions.AFTER:
            return self.layer_norm_module(input)
        return input

    def _handle_model_output(self, output: Tensor) -> Tensor:
        return output

    # TODO: In the future instead multiple function inputs
    # use a dataset class to encapsulate the inputs and outputs
    # of the block
    # def forward(
    #     self,
    #     data: LayerData,
    # ) -> LayerData:
    #     output = self.model(data.tensor)


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
    layer_type: Layer | None = field(
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
        config = getattr(cfg, "layer_stack_config", cfg)
        self.cfg: "LayerStackConfig" = self._overwrite_config(config, overrides)
        self.main_cfg = self._resolve_main_config(self.cfg, cfg)

        self.input_dim = self.cfg.input_dim
        self.hidden_dim = self.cfg.hidden_dim
        self.output_dim = self.cfg.output_dim
        self.num_layers = self.cfg.num_layers
        self.model_type = self.cfg.model_type
        self.activation = self.cfg.activation
        self.residual_flag = self.cfg.residual_flag
        self.adaptive_computation_flag = self.cfg.adaptive_computation_flag
        self.dropout_probability = self.cfg.dropout_probability
        self.layer_norm_position = self.cfg.layer_norm_position
        self.callback_function = None

        self.layer_block_model = self.cfg.layer_type or Layer

    def set_callback(self, callback) -> Self:
        self.callback_function = callback
        return self

    def _override_model_type(
        self,
        overrides: "LayerStackConfig | None",
        model_type: "BaseOptions | None",
        layer_type: "Layer | None" = None,
    ) -> LayerStackConfig:
        if overrides is None:
            return LayerStackConfig(
                model_type=model_type,
                layer_type=layer_type,
            )
        overrides.model_type = model_type
        overrides.layer_type = layer_type
        return overrides

    def build_model(self) -> Layer | Sequential:
        layers = []

        layer_adjustment = self.__add_initial_layer(layers)
        self.__add_hidden_layers(layers, layer_adjustment)
        self.__add_output_layer(layers)

        self.mark_last_layer(layers)

        if len(layers) == 1:
            [model] = layers
            self._initialize_parameters(model)
            return model
        model = Sequential(*layers)
        self._initialize_parameters(model)
        return model

    def mark_last_layer(self, layers: list) -> None:
        layers[-1].mark_as_last_layer()

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
        if isinstance(self.model_type, BaseOptions):
            return self.model_type.value
        return self.model_type

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

        layer_block_model = self.layer_block_model(
            model=model,
            layer_norm_dim=layer_norm_dim,
            residual_connection_flag=residual_flag,
            activation_function=self.activation,
            dropout_probability=self.dropout_probability,
            layer_norm_position=self.layer_norm_position,
        )

        return layer_block_model

    def __resolve_model_type_overrides(self, input_dim: int, output_dim: int):
        c = copy.deepcopy(self.main_cfg)
        c.input_dim = input_dim
        c.output_dim = output_dim
        return c

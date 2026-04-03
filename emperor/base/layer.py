import copy
import torch.nn as nn

from torch.types import Tensor
from torch.nn import Sequential
from dataclasses import dataclass, field
from emperor.base.utils import ConfigBase, Module
from emperor.base.enums import (
    ActivationOptions,
    BaseOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.config import ModelConfig
    from emperor.linears.options import LinearLayerOptions
    from emperor.halting.config import HaltingConfig
    from emperor.halting.utils.options.base import HaltingBase, HaltingStateBase


@dataclass
class LayerState:
    hidden: Tensor
    loss: Tensor | None = None
    halting_state: "HaltingStateBase | None" = None


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
    gate_config: "LayerStackConfig | None" = field(
        default=None,
        metadata={
            "help": "LayerStack config for the gating mechanism; if None gates are skipped"
        },
    )
    halting_config: "HaltingConfig | None" = field(
        default=None,
        metadata={"help": "Optional halting config for adaptive computation per layer"},
    )
    shared_halting_flag: bool | None = field(
        default=None,
        metadata={
            "help": "If True, one halting module is shared across all layers; if False, each layer gets its own"
        },
    )


class Layer(Module):
    def __init__(
        self,
        model: "Module",
        activation_function: "ActivationOptions | None" = None,
        layer_norm_dim: int | None = None,
        residual_connection_flag: bool = False,
        dropout_probability: float = 0.0,
        layer_norm_position: "LayerNormPositionOptions | None" = None,
        gate_config: "LayerStackConfig | None" = None,
        halting_config: "HaltingConfig | None" = None,
        shared_halting_flag: bool = False,
    ):
        super().__init__()

        self.model = model
        self.activation_function = activation_function
        self.layer_norm_dim = layer_norm_dim
        self.residual_connection_flag = residual_connection_flag
        self.dropout_probability = dropout_probability
        self.layer_norm_position = layer_norm_position
        self.gate_config = gate_config
        self.halting_config = halting_config
        self.shared_halting_flag = shared_halting_flag
        self.gate_module = self.__build_gate_module()
        self.halting_module = self.__build_halting_module()

        self.has_activation = self.activation_function is not None
        self.has_dropout = self.dropout_probability > 0.0

        self.dropout_module = self.__init_dropout_module()
        self.layer_norm_module = self.__init_layer_norm_module()
        self.last_layer_flag = False

    def __build_gate_module(self) -> "Module | None":
        if self.gate_config is None:
            return None
        from emperor.linears.options import LinearLayerOptions

        if self.gate_config.model_type != LinearLayerOptions.BASE:
            raise ValueError(
                f"gate_config.model_type must be LinearLayerOptions.BASE, got {self.gate_config.model_type}"
            )
        if self.gate_config.gate_config is not None:
            raise ValueError(
                "gate_config.gate_config must be None, nested gates are not allowed"
            )
        if self.gate_config.halting_config is not None:
            raise ValueError(
                "gate_config.halting_config must be None, halting is not allowed in gates"
            )
        if self.gate_config.shared_halting_flag:
            raise ValueError(
                "gate_config.shared_halting_flag must be False, halting is not allowed in gates"
            )

        return LayerStack(self.gate_config).build()

    def __build_halting_module(self) -> "HaltingBase | None":
        if self.halting_config is None:
            if self.shared_halting_flag:
                raise ValueError(
                    "shared_halting_flag must be False when no halting_config is provided"
                )
            return None
        from emperor.halting.factory import HaltingFactory

        return HaltingFactory(self.halting_config).build()

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
        assert (
            self.layer_norm_dim > 0
        ), f"expected layer_norm_dim must be greater than 0, received {self.layer_norm_dim}"
        return nn.LayerNorm(self.layer_norm_dim)

    def forward(
        self,
        state: "LayerState",
    ) -> "LayerState":
        residual = self._handle_model_input(state.hidden)
        X = self.__maybe_apply_layer_norm_before(residual)
        X = self._handle_model_processing(X)
        X = self.__maybe_apply_layer_norm_default(X)
        X = self.__maybe_apply_activation(X)
        X = self.__maybe_apply_gates(X)
        X = self.__maybe_apply_dropout(X)
        X = self.__maybe_apply_residual_connection(X, residual)
        X = self.__maybe_apply_layer_norm_after(X)
        state = self.__maybe_apply_halting(X, state)
        return self._handle_model_output(state)

    def _handle_model_input(self, input: Tensor) -> Tensor:
        return input

    def __maybe_apply_layer_norm_before(self, input: Tensor):
        if self.layer_norm_position == LayerNormPositionOptions.BEFORE:
            return self.layer_norm_module(input)
        return input

    def _handle_model_processing(
        self,
        main_model_input: Tensor,
        additional_model_inputs: dict = {},
    ) -> tuple[Tensor, tuple | None]:
        return self.model(main_model_input, **additional_model_inputs)

    def __maybe_apply_layer_norm_default(self, input: Tensor):
        if self.layer_norm_position == LayerNormPositionOptions.DEFAULT:
            return self.layer_norm_module(input)
        return input

    def __maybe_apply_activation(self, input: Tensor):
        if self.has_activation:
            return self.activation_function(input)
        return input

    def __maybe_apply_gates(self, input: Tensor) -> Tensor:
        if self.gate_module is not None:
            return self.forward_module(self.gate_module, input) * input
        return input

    @staticmethod
    def forward_module(module: "Module", input: Tensor) -> Tensor:
        return module(LayerState(hidden=input)).hidden

    def __maybe_apply_dropout(self, input: Tensor):
        if self.has_dropout:
            return self.dropout_module(input)
        return input

    def __maybe_apply_residual_connection(self, input: Tensor, prev_input: Tensor):
        if self.residual_connection_flag:
            return input + prev_input
        return input

    def __maybe_apply_layer_norm_after(self, input: Tensor):
        if self.layer_norm_position == LayerNormPositionOptions.AFTER:
            return self.layer_norm_module(input)
        return input

    def __maybe_apply_halting(self, input: Tensor, state: LayerState) -> LayerState:
        if self.halting_module is not None:
            state.halting_state, input = self.halting_module.update_halting_state(
                state.halting_state, input
            )
            if self.last_layer_flag:
                hidden, loss = self.halting_module.finalize_weighted_accumulation(
                    state.halting_state, input
                )
                loss = loss if state.loss is None else state.loss + loss
                state.hidden = hidden
                state.loss = loss

                return state
        state.hidden = input
        return state

    def _handle_model_output(self, layer_state: LayerState) -> LayerState:
        return layer_state


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
    last_layer_bias_option: LastLayerBiasOptions | None = field(
        default=None,
        metadata={"help": "Override bias on the last layer regardless of bias_flag"},
    )
    apply_output_pipeline_flag: bool | None = field(
        default=None,
        metadata={
            "help": "If True, the output layer gets the full processing pipeline (activation, dropout, layer norm, residual, gate)"
        },
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

        self.input_dim: int = self.cfg.input_dim
        self.hidden_dim: int = self.cfg.hidden_dim
        self.output_dim: int = self.cfg.output_dim
        self.num_layers: int = self.cfg.num_layers
        self.model_type: "LinearLayerOptions" = self.cfg.model_type
        self.activation: ActivationOptions = self.cfg.activation
        self.residual_flag: bool = self.cfg.residual_flag
        self.adaptive_computation_flag: bool = self.cfg.adaptive_computation_flag
        self.dropout_probability: float = self.cfg.dropout_probability
        self.layer_norm_position: LayerNormPositionOptions = (
            self.cfg.layer_norm_position
        )
        self.last_layer_bias_option: LastLayerBiasOptions = (
            self.cfg.last_layer_bias_option
        )
        self.gate_config: "LayerStack" = self.cfg.gate_config
        self.halting_config: "HaltingConfig" = self.cfg.halting_config
        self.shared_halting_flag: bool = self.cfg.shared_halting_flag
        self.apply_output_pipeline_flag: bool = self.cfg.apply_output_pipeline_flag

        if self.halting_config is not None and self.num_layers < 2:
            raise ValueError(
                f"num_layers must be at least 2 when halting_config is provided, "
                f"got {self.num_layers}. The halting mechanism requires multiple steps to accumulate "
                f"halting probabilities across layers."
            )

        self.layer_block_model = self.cfg.layer_type or Layer

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

    def build(self) -> Layer | Sequential:
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
            layers.append(layer)

    def __add_output_layer(self, layers: list) -> None:
        layer_input_dim = self.hidden_dim if self.num_layers > 1 else self.input_dim
        if self.apply_output_pipeline_flag:
            layer = self.__create_layer(
                layer_input_dim,
                self.output_dim,
                residual_flag=False,
                is_output_layer=True,
            )
            layers.append(layer)
            return
        config = self.__resolve_model_type_overrides(
            self.main_cfg, layer_input_dim, self.output_dim
        )
        gate_config = None
        if self.gate_config is not None:
            gate_config = self.__resolve_model_type_overrides(
                self.gate_config, self.output_dim, self.output_dim
            )
        self.__apply_last_layer_bias_override(config)
        layer = self.layer_block_model(
            model=self.__get_model_type()(config),
            gate_config=gate_config,
            halting_config=self.halting_config,
            shared_halting_flag=self.shared_halting_flag,
        )
        layers.append(layer)

    def __apply_last_layer_bias_override(self, config) -> None:
        match self.last_layer_bias_option:
            case LastLayerBiasOptions.DISABLED:
                config.bias_flag = False
            case LastLayerBiasOptions.ENABLED:
                config.bias_flag = True

    def __get_model_type(self):
        if isinstance(self.model_type, BaseOptions):
            return self.model_type.value
        return self.model_type

    def __create_layer(
        self,
        input_dim: int,
        output_dim: int,
        residual_flag: bool = True,
        is_output_layer: bool = False,
    ) -> Layer:
        layer_norm_dim = None
        if self.layer_norm_position != LayerNormPositionOptions.NONE:
            layer_norm_dim = output_dim
            if self.layer_norm_position == LayerNormPositionOptions.BEFORE:
                layer_norm_dim = input_dim
        config = self.__resolve_model_type_overrides(
            self.main_cfg, input_dim, output_dim
        )

        gate_config = None
        if self.gate_config is not None:
            gate_config = self.__resolve_model_type_overrides(
                self.gate_config, self.hidden_dim, self.hidden_dim
            )
        if is_output_layer:
            self.__apply_last_layer_bias_override(config)
        model = self.__get_model_type()(config)

        layer_block_model = self.layer_block_model(
            model=model,
            layer_norm_dim=layer_norm_dim,
            residual_connection_flag=residual_flag,
            activation_function=self.activation,
            dropout_probability=self.dropout_probability,
            layer_norm_position=self.layer_norm_position,
            gate_config=gate_config,
            halting_config=self.halting_config,
            shared_halting_flag=self.shared_halting_flag,
        )

        return layer_block_model

    def __resolve_model_type_overrides(
        self,
        cfg: LayerStackConfig,
        input_dim: int,
        output_dim: int,
    ):
        c = copy.deepcopy(cfg)
        c.input_dim = input_dim
        c.output_dim = output_dim
        return c

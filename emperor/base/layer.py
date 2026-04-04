import torch.nn as nn
from copy import deepcopy

from torch.types import Tensor
from torch.nn import Sequential
from dataclasses import dataclass, field
from emperor.base.utils import ConfigBase, Module
from emperor.base.enums import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.config import ModelConfig
    from emperor.halting.config import HaltingConfig
    from emperor.halting.utils.options.base import HaltingBase, HaltingStateBase


@dataclass
class LayerState:
    hidden: Tensor
    loss: Tensor | None = None
    halting_state: "HaltingStateBase | None" = None


@dataclass
class LayerConfig(ConfigBase):
    input_dim: int | None = field(
        default=None,
        metadata={"help": "Input dimension of the first `Linear` layer"},
    )
    output_dim: int | None = field(
        default=None,
        metadata={"help": "Output dimension of the output `Linear` layer"},
    )
    activation: ActivationOptions | None = field(
        default=None,
        metadata={"help": "Activation function or layer to use"},
    )
    residual_flag: bool | None = field(
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
    model_config: ConfigBase | None = field(
        default=None,
        metadata={"help": "Config used to build the model module within the layer"},
    )


class LayerValidator:
    OPTIONAL_FIELDS = {
        "gate_config",
        "halting_config",
        "model_config",
        "override_config",
    }

    @staticmethod
    def validate(cfg: LayerConfig) -> None:
        LayerValidator.__validate_required_fields(cfg)
        LayerValidator.__validate_dimensions(cfg.input_dim, cfg.output_dim)
        LayerValidator.__validate_dropout_probability(cfg.dropout_probability)
        LayerValidator.__validate_model_config(cfg.model_config)
        LayerValidator.__validate_gate_config(cfg.gate_config)
        LayerValidator.__validate_halting_config(
            cfg.halting_config, cfg.shared_halting_flag
        )

    @staticmethod
    def __validate_dimensions(input_dim: int, output_dim: int) -> None:
        if input_dim <= 0:
            raise ValueError(f"input_dim must be greater than 0, received {input_dim}")
        if output_dim <= 0:
            raise ValueError(
                f"output_dim must be greater than 0, received {output_dim}"
            )

    @staticmethod
    def __validate_dropout_probability(dropout_probability: float) -> None:
        if dropout_probability < 0.0 or dropout_probability > 1.0:
            raise ValueError(
                f"dropout_probability must be between 0.0 and 1.0, received {dropout_probability}"
            )

    @staticmethod
    def __validate_required_fields(cfg: LayerConfig) -> None:
        for field_name in cfg.__dataclass_fields__:
            if field_name in LayerValidator.OPTIONAL_FIELDS:
                continue
            if getattr(cfg, field_name) is None:
                raise ValueError(f"{field_name} is required, received None")

    @staticmethod
    def __validate_model_config(model_config: ConfigBase | None) -> None:
        if model_config is None:
            raise ValueError(
                "model_config is required, Layer needs it to build the model"
            )

    @staticmethod
    def __validate_gate_config(gate_config: "LayerStackConfig | None") -> None:
        if gate_config is None:
            return
        layer_config = gate_config.layer_config
        if layer_config is None:
            return

        if layer_config.gate_config is not None:
            raise ValueError(
                "gate_config.layer_config.gate_config must be None, nested gates are not allowed"
            )
        if layer_config.halting_config is not None:
            raise ValueError(
                "gate_config.layer_config.halting_config must be None, halting is not allowed in gates"
            )
        if layer_config.shared_halting_flag:
            raise ValueError(
                "gate_config.layer_config.shared_halting_flag must be False, halting is not allowed in gates"
            )

    @staticmethod
    def __validate_halting_config(
        halting_config: "HaltingConfig | None",
        shared_halting_flag: bool | None,
    ) -> None:
        if halting_config is None and shared_halting_flag:
            raise ValueError(
                "shared_halting_flag must be False when no halting_config is provided"
            )


class Layer(Module):
    def __init__(
        self,
        cfg: LayerConfig,
        overrides: LayerConfig | None = None,
    ):
        super().__init__()
        self.cfg: LayerConfig = self._overwrite_config(cfg, overrides)

        self.input_dim: int = self.cfg.input_dim
        self.output_dim: int = self.cfg.output_dim
        self.layer_norm_position: LayerNormPositionOptions = (
            self.cfg.layer_norm_position
        )
        self.activation_function: ActivationOptions = self.cfg.activation
        self.layer_norm_dim: int | None = self.__resolve_layer_norm_dim()
        self.residual_flag: bool = self.cfg.residual_flag or False
        self.dropout_probability: float = self.cfg.dropout_probability or 0.0
        self.gate_config: "LayerStackConfig | None" = self.cfg.gate_config
        self.halting_config: "HaltingConfig | None" = self.cfg.halting_config
        self.shared_halting_flag: bool = self.cfg.shared_halting_flag or False
        self.model_config: "ConfigBase" = self.cfg.model_config
        LayerValidator.validate(self.cfg)

        self.model = self.__build_model()
        self.gate_model = self.__build_gate_model()
        self.halting_model = self.__build_halting_model()

        self.has_activation = self.activation_function != ActivationOptions.DISABLED
        self.has_dropout = self.dropout_probability > 0.0

        self.dropout_module = self.__init_dropout_module()
        self.layer_norm_module = self.__init_layer_norm_module()
        self.last_layer_flag = False

    def __build_model(self) -> "Module | None":
        if self.model_config is None:
            return None
        return self.model_config.build(self.input_dim, self.output_dim)

    def __build_gate_model(self) -> "Module | None":
        if self.gate_config is None:
            return None
        overrides = LayerStackConfig(
            input_dim=self.output_dim, output_dim=self.output_dim
        )
        return LayerStack(self.gate_config, overrides).build()

    def __build_halting_model(self) -> "HaltingBase | None":
        if self.halting_config is None:
            return None
        from emperor.halting.factory import HaltingFactory

        return HaltingFactory(self.halting_config).build()

    def __init_dropout_module(self) -> nn.Module | None:
        if self.has_dropout:
            return nn.Dropout(self.dropout_probability)
        return None

    def __resolve_layer_norm_dim(self) -> int | None:
        if self.layer_norm_position == LayerNormPositionOptions.DISABLED:
            return None
        if self.layer_norm_position == LayerNormPositionOptions.BEFORE:
            return self.input_dim
        return self.output_dim

    def __init_layer_norm_module(self) -> nn.Module | None:
        if self.layer_norm_dim is None:
            return None
        return nn.LayerNorm(self.layer_norm_dim)

    def mark_as_last_layer(self) -> None:
        self.last_layer_flag = True

    def forward(
        self,
        state: "LayerState",
    ) -> "LayerState":
        residual = self._handle_model_input(state.hidden)
        X = self.__maybe_apply_layer_norm_before(residual)
        X = self._handle_model_processing(X, state)
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
        state: "LayerStack",
    ) -> tuple[Tensor, tuple | None]:
        return self.model(main_model_input)

    def __maybe_apply_layer_norm_default(self, input: Tensor):
        if self.layer_norm_position == LayerNormPositionOptions.DEFAULT:
            return self.layer_norm_module(input)
        return input

    def __maybe_apply_activation(self, input: Tensor):
        if self.has_activation:
            return self.activation_function(input)
        return input

    def __maybe_apply_gates(self, input: Tensor) -> Tensor:
        if self.gate_model is not None:
            return self.forward_module(self.gate_model, input) * input
        return input

    @staticmethod
    def forward_module(module: "Module", input: Tensor) -> Tensor:
        return module(LayerState(hidden=input)).hidden

    def __maybe_apply_dropout(self, input: Tensor):
        if self.has_dropout:
            return self.dropout_module(input)
        return input

    def __maybe_apply_residual_connection(self, input: Tensor, prev_input: Tensor):
        if self.residual_flag:
            return input + prev_input
        return input

    def __maybe_apply_layer_norm_after(self, input: Tensor):
        if self.layer_norm_position == LayerNormPositionOptions.AFTER:
            return self.layer_norm_module(input)
        return input

    def __maybe_apply_halting(self, input: Tensor, state: LayerState) -> LayerState:
        if self.halting_model is not None:
            state.halting_state, input = self.halting_model.update_halting_state(
                state.halting_state, input
            )
            if self.last_layer_flag:
                hidden, loss = self.halting_model.finalize_weighted_accumulation(
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
class LayerStackConfig(ConfigBase):
    input_dim: int | None = field(
        default=None,
        metadata={"help": "Input dimension of the first layer in the stack"},
    )
    hidden_dim: int | None = field(
        default=None,
        metadata={
            "help": "Dimension used for all hidden layers between input and output"
        },
    )
    output_dim: int | None = field(
        default=None,
        metadata={"help": "Output dimension of the last layer in the stack"},
    )
    num_layers: int | None = field(
        default=None,
        metadata={"help": "Total number of layers in the stack"},
    )
    layer_type: Layer | None = field(
        default=None,
        metadata={
            "help": "Layer subclass to use for each layer; defaults to Layer if None"
        },
    )
    last_layer_bias_option: "LastLayerBiasOptions | None" = field(
        default=None,
        metadata={
            "help": "Override bias on the last layer: DEFAULT keeps model_config value, DISABLED removes bias, ENABLED adds bias"
        },
    )
    apply_output_pipeline_flag: bool | None = field(
        default=None,
        metadata={
            "help": "If True, the output layer applies the full pipeline (activation, dropout, layer norm, residual, gate); if False, returns clean model output"
        },
    )
    layer_config: LayerConfig | None = field(
        default=None,
        metadata={
            "help": "LayerConfig shared across all layers in the stack; per-layer overrides are applied on top"
        },
    )


class LayerStackValidator:
    @staticmethod
    def validate(cfg: "LayerStackConfig") -> None:
        LayerStackValidator.__validate_required_fields(cfg)
        LayerStackValidator.__validate_dimensions(cfg)
        LayerStackValidator.__validate_halting_config(cfg)

    @staticmethod
    def __validate_dimensions(cfg: "LayerStackConfig") -> None:
        if cfg.input_dim is not None and cfg.input_dim <= 0:
            raise ValueError(
                f"input_dim must be greater than 0, received {cfg.input_dim}"
            )
        if cfg.hidden_dim is not None and cfg.hidden_dim <= 0:
            raise ValueError(
                f"hidden_dim must be greater than 0, received {cfg.hidden_dim}"
            )
        if cfg.output_dim is not None and cfg.output_dim <= 0:
            raise ValueError(
                f"output_dim must be greater than 0, received {cfg.output_dim}"
            )
        if cfg.num_layers is not None and cfg.num_layers <= 0:
            raise ValueError(
                f"num_layers must be greater than 0, received {cfg.num_layers}"
            )

    @staticmethod
    def __validate_required_fields(cfg: "LayerStackConfig") -> None:
        if cfg.input_dim is None:
            raise ValueError(f"input_dim is required, received {cfg.input_dim}")
        if cfg.hidden_dim is None:
            raise ValueError(f"hidden_dim is required, received {cfg.hidden_dim}")
        if cfg.output_dim is None:
            raise ValueError(f"output_dim is required, received {cfg.output_dim}")
        if cfg.num_layers is None:
            raise ValueError(f"num_layers is required, received {cfg.num_layers}")
        if cfg.last_layer_bias_option is None:
            raise ValueError(
                f"last_layer_bias_option is required, received {cfg.last_layer_bias_option}"
            )
        if cfg.apply_output_pipeline_flag is None:
            raise ValueError(
                f"apply_output_pipeline_flag is required, received {cfg.apply_output_pipeline_flag}"
            )
        if cfg.layer_config is None:
            raise ValueError(f"layer_config is required, received {cfg.layer_config}")

    @staticmethod
    def __validate_halting_config(cfg: "LayerStackConfig") -> None:
        if cfg.layer_config is None:
            return
        if cfg.layer_config.halting_config is not None and cfg.num_layers < 2:
            raise ValueError(
                f"num_layers must be at least 2 when halting_config is provided, "
                f"got {cfg.num_layers}. The halting mechanism requires multiple steps to accumulate "
                f"halting probabilities across layers."
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
        LayerStackValidator.validate(self.cfg)

        self.input_dim: int = self.cfg.input_dim
        self.hidden_dim: int = self.cfg.hidden_dim
        self.output_dim: int = self.cfg.output_dim
        self.num_layers: int = self.cfg.num_layers
        self.last_layer_bias_option: "LastLayerBiasOptions" = (
            self.cfg.last_layer_bias_option
        )
        self.apply_output_pipeline_flag: bool = self.cfg.apply_output_pipeline_flag
        self.layer_config: LayerConfig = self.cfg.layer_config

        self.layer_block_model = self.cfg.layer_type or Layer

    def build(self) -> Layer | Sequential:
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
            layer = self.__create_layer(self.input_dim, self.hidden_dim)
            layers.append(layer)
            return self.SEPARATE_INPUT_OUTPUT_DIM
        return self.SHARED_INPUT_OUTPUT_DIM

    def __add_hidden_layers(self, layers: list, layer_adjustment: int) -> None:
        for _ in range(self.num_layers - layer_adjustment):
            layer = self.__create_layer(self.hidden_dim, self.hidden_dim)
            layers.append(layer)

    def __add_output_layer(self, layers: list) -> None:
        layer_input_dim = self.hidden_dim if self.num_layers > 1 else self.input_dim
        bias_overrides = self.__resolve_last_layer_bias_override()
        if self.apply_output_pipeline_flag:
            layer = self.__create_layer(
                layer_input_dim, self.output_dim, bias_overrides
            )
            layer.mark_as_last_layer()
            layers.append(layer)
            return
        overrides = LayerConfig(
            activation=ActivationOptions.DISABLED,
            residual_flag=False,
            dropout_probability=0.0,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
        )
        if bias_overrides is not None:
            overrides = self._overwrite_config(overrides, bias_overrides)
        layer = self.__create_layer(layer_input_dim, self.output_dim, overrides)
        layer.mark_as_last_layer()
        layers.append(layer)

    def __resolve_last_layer_bias_override(self) -> LayerConfig | None:
        if self.last_layer_bias_option == LastLayerBiasOptions.DEFAULT:
            return None
        if not hasattr(self.layer_config.model_config, "bias_flag"):
            return None

        model_config = deepcopy(self.layer_config.model_config)
        match self.last_layer_bias_option:
            case LastLayerBiasOptions.DISABLED:
                model_config.bias_flag = False
            case LastLayerBiasOptions.ENABLED:
                model_config.bias_flag = True
        return LayerConfig(model_config=model_config)

    def __create_layer(
        self,
        input_dim: int,
        output_dim: int,
        overrides: LayerConfig | None = None,
    ) -> Layer:
        residual_flag = False if input_dim != output_dim else None
        dim_overrides = LayerConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            residual_flag=residual_flag,
        )
        if overrides is not None:
            dim_overrides = self._overwrite_config(dim_overrides, overrides)
        return self.layer_block_model(cfg=self.layer_config, overrides=dim_overrides)

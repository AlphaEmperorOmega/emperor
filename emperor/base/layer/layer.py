import torch.nn as nn

from torch.types import Tensor
from torch.nn import Sequential
from emperor.base.utils import ConfigBase, Module
from emperor.base.enums import (
    ActivationOptions,
    LayerNormPositionOptions,
)
from emperor.base.layer.state import LayerState
from emperor.base.layer.config import LayerConfig, LayerStackConfig
from emperor.base.layer._validator import LayerValidator

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.halting.config import HaltingConfig
    from emperor.halting.utils.options.base import HaltingBase


class Layer(Module):
    def __init__(
        self,
        cfg: LayerConfig,
        overrides: LayerConfig | None = None,
    ):
        super().__init__()
        self.cfg: LayerConfig = self._override_config(cfg, overrides)

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

    def __resolve_layer_norm_dim(self) -> int | None:
        if self.layer_norm_position == LayerNormPositionOptions.DISABLED:
            return None
        if self.layer_norm_position == LayerNormPositionOptions.BEFORE:
            return self.input_dim
        return self.output_dim

    def __build_model(self) -> "Module | None":
        return self.__build_from_config(
            self.model_config, input_dim=self.input_dim, output_dim=self.output_dim
        )

    def __build_gate_model(self) -> "Layer | Sequential | None":
        return self.__build_from_config(
            self.gate_config, input_dim=self.output_dim, output_dim=self.output_dim
        )

    def __build_halting_model(self) -> "HaltingBase | None":
        return self.__build_from_config(self.halting_config, input_dim=self.output_dim)

    def __build_from_config(
        self, config: "ConfigBase | None", **kwargs
    ) -> "Module | None":
        if config is None:
            return None
        return config.build(overrides=type(config)(**kwargs))

    def __init_dropout_module(self) -> nn.Module | None:
        if self.has_dropout:
            return nn.Dropout(self.dropout_probability)
        return None

    def __init_layer_norm_module(self) -> nn.Module | None:
        if self.layer_norm_dim is None:
            return None
        return nn.LayerNorm(self.layer_norm_dim)

    def mark_as_last_layer(self) -> None:
        self.last_layer_flag = True

    @staticmethod
    def forward_with_state(model: "Module", input: Tensor) -> Tensor:
        return model(LayerState(hidden=input)).hidden

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
        state: "LayerState",
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
            return self.forward_with_state(self.gate_model, input) * input
        return input

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
            state.halting_state, halting_output = (
                self.halting_model.update_halting_state(state.halting_state, input)
            )
            if self.last_layer_flag:
                hidden, loss = self.halting_model.finalize_weighted_accumulation(
                    state.halting_state, input
                )
                loss = loss if state.loss is None else state.loss + loss
                state.hidden = hidden
                state.loss = loss
                return state
            state.hidden = halting_output
            return state
        state.hidden = input
        return state

    def _handle_model_output(self, layer_state: LayerState) -> LayerState:
        return layer_state

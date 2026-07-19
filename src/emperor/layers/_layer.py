from typing import TYPE_CHECKING

import torch.nn as nn
from torch import Tensor

from emperor.config import ConfigBase
from emperor.layers._composition.gate import LayerGate
from emperor.layers._composition.residual import ResidualConnection
from emperor.layers._config import GateConfig, LayerConfig, ResidualConfig
from emperor.layers._options import (
    ActivationOptions,
    LayerNormPositionOptions,
)
from emperor.layers._state import LayerState
from emperor.layers._support import LayerModuleBase
from emperor.layers._validation import LayerValidator
from emperor.memory import MemoryPositionOptions

if TYPE_CHECKING:
    from emperor.halting import HaltingBase, HaltingConfig, HaltingStateBase
    from emperor.memory import DynamicMemoryConfig, MemoryInterface
    from emperor.nn import Module


class Layer(LayerModuleBase):
    VALIDATOR = LayerValidator

    def __init__(
        self,
        cfg: LayerConfig,
        overrides: LayerConfig | None = None,
    ):
        super().__init__()
        self.cfg: LayerConfig = self._override_config(cfg, overrides)
        self._validate_configuration()

        self.input_dim: int = self.cfg.input_dim
        self.output_dim: int = self.cfg.output_dim
        self.layer_norm_position: LayerNormPositionOptions = (
            self.cfg.layer_norm_position
        )
        self.activation_function: ActivationOptions = self.cfg.activation
        self.layer_norm_dim: int | None = self.__resolve_layer_norm_dim()
        self.residual_config: ResidualConfig | None = self.cfg.residual_config
        self.dropout_probability: float = self.cfg.dropout_probability
        self.gate_config: GateConfig | None = self.cfg.gate_config
        self.halting_config: HaltingConfig | None = self.cfg.halting_config
        self.memory_config: DynamicMemoryConfig | None = self.cfg.memory_config
        self.layer_model_config: ConfigBase = self.cfg.layer_model_config

        self.model = self.__build_model()
        self.gate_model: LayerGate | None = self.__build_gate()
        self.halting_model = self.__build_halting_model()
        self.memory_model = self.__build_memory_model()
        self.residual_connection = self.__build_residual_connection()
        self.dropout_module = self.__init_dropout_module()
        self.layer_norm_module = self.__init_layer_norm_module()
        self.last_layer_flag = False

    def _validate_configuration(self) -> None:
        self.VALIDATOR.validate(self)

    def __resolve_layer_norm_dim(self) -> int | None:
        if self.layer_norm_position == LayerNormPositionOptions.DISABLED:
            return None
        if self.layer_norm_position == LayerNormPositionOptions.BEFORE:
            return self.input_dim
        return self.output_dim

    def __build_model(self) -> "Module | None":
        return self._build_from_config(
            self.layer_model_config,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
        )

    def __build_gate(self) -> LayerGate | None:
        return self._build_from_config(
            self.gate_config,
            gate_dim=self.output_dim,
        )

    def __build_halting_model(self) -> "HaltingBase | None":
        return self._build_from_config(self.halting_config, input_dim=self.output_dim)

    def __build_memory_model(self) -> "MemoryInterface | None":
        return self._build_from_config(
            self.memory_config, input_dim=self.input_dim, output_dim=self.output_dim
        )

    def __build_residual_connection(self) -> ResidualConnection | None:
        return self._build_from_config(
            self.residual_config,
            residual_dim=self.output_dim,
        )

    def __init_dropout_module(self) -> nn.Module | None:
        if self.__should_apply_dropout():
            return nn.Dropout(self.dropout_probability)
        return None

    def __should_apply_dropout(self) -> bool:
        return self.dropout_probability > 0.0

    def __init_layer_norm_module(self) -> nn.Module | None:
        if self.layer_norm_dim is None:
            return None
        return nn.LayerNorm(self.layer_norm_dim)

    def mark_as_last_layer(self) -> None:
        self.last_layer_flag = True

    @staticmethod
    def run_model_returning_state(model: "Module", X: Tensor) -> "LayerState":
        input_state = LayerState(hidden=X)
        return model(input_state)

    @staticmethod
    def run_model_returning_hidden(model: "Module", X: Tensor) -> Tensor:
        return Layer.run_model_returning_state(model, X).hidden

    def forward(
        self,
        state: "LayerState",
    ) -> "LayerState":
        if self.__should_skip_halted_state(state):
            return state
        residual = self._handle_model_input(state.hidden)
        X = self.__maybe_apply_layer_norm_before(residual)
        X = self.__maybe_apply_memory_before(X)
        X = self._handle_model_processing(X, state)
        X = self.__maybe_apply_memory_after(X)
        X = self.__maybe_apply_layer_norm_default(X)
        X = self.__maybe_apply_activation(X)
        X = self.__maybe_apply_gates(X)
        X = self.__maybe_apply_dropout(X)
        X = self.__maybe_apply_residual_connection(X, residual)
        X = self.__maybe_apply_layer_norm_after(X)
        X, halting_state, loss = self.__maybe_apply_halting(
            X, state.halting_state, state.loss
        )
        state = self.__update_output_state(state, X, halting_state, loss)
        return self._handle_model_output(state)

    def __should_skip_halted_state(self, state: LayerState) -> bool:
        if not self.__has_halting_state(state):
            return False

        return self.__is_halting_state_complete(state.halting_state)

    def __has_halting_state(self, state: LayerState) -> bool:
        return self.halting_model is not None and state.halting_state is not None

    def __is_halting_state_complete(
        self,
        halting_state: "HaltingStateBase | None",
    ) -> bool:
        halt_mask = getattr(halting_state, "halt_mask", None)
        if halt_mask is None:
            return False
        return bool(halt_mask.all().item())

    def _handle_model_input(self, input: Tensor) -> Tensor:
        return input

    def __maybe_apply_layer_norm_before(self, input: Tensor):
        if self.layer_norm_position == LayerNormPositionOptions.BEFORE:
            return self.layer_norm_module(input)
        return input

    def __maybe_apply_memory_before(self, input: Tensor) -> Tensor:
        return self._maybe_apply_memory_by_position(
            input, MemoryPositionOptions.BEFORE_AFFINE
        )

    def _handle_model_processing(
        self,
        main_model_input: Tensor,
        state: "LayerState",
    ) -> tuple[Tensor, tuple | None]:
        return self.model(main_model_input)

    def __maybe_apply_memory_after(self, input: Tensor) -> Tensor:
        return self._maybe_apply_memory_by_position(
            input, MemoryPositionOptions.AFTER_AFFINE
        )

    def __maybe_apply_layer_norm_default(self, input: Tensor):
        if self.layer_norm_position == LayerNormPositionOptions.DEFAULT:
            return self.layer_norm_module(input)
        return input

    def __maybe_apply_activation(self, input: Tensor):
        if self.__should_apply_activation():
            return self.activation_function(input)
        return input

    def __should_apply_activation(self) -> bool:
        return self.activation_function != ActivationOptions.DISABLED

    def __maybe_apply_gates(
        self,
        input: Tensor,
    ) -> Tensor:
        if self.gate_model is None:
            return input
        return self.gate_model(input)

    def __maybe_apply_dropout(self, input: Tensor):
        if self.__should_apply_dropout():
            return self.dropout_module(input)
        return input

    def __maybe_apply_residual_connection(self, input: Tensor, prev_input: Tensor):
        if self.residual_connection is None:
            return input
        return self.residual_connection(input, prev_input)

    def __maybe_apply_layer_norm_after(self, input: Tensor):
        if self.layer_norm_position == LayerNormPositionOptions.AFTER:
            return self.layer_norm_module(input)
        return input

    def __maybe_apply_halting(
        self,
        input: Tensor,
        halting_state: "HaltingStateBase | None",
        loss: Tensor | None,
    ) -> tuple[Tensor, "HaltingStateBase | None", Tensor | None]:
        if self.halting_model is None:
            return input, halting_state, loss

        halting_state, halting_output = self.halting_model.update_halting_state(
            halting_state, input
        )
        if self.last_layer_flag or self.__is_halting_state_complete(halting_state):
            return self.__maybe_finalize_halted_output(input, halting_state, loss)
        return halting_output, halting_state, loss

    def __maybe_finalize_halted_output(
        self,
        input: Tensor,
        halting_state: "HaltingStateBase | None",
        loss: Tensor | None,
    ) -> tuple[Tensor, "HaltingStateBase | None", Tensor | None]:
        if self.halting_model is None or halting_state is None:
            return input, halting_state, loss
        hidden, halting_loss = self.halting_model.finalize_weighted_accumulation(
            halting_state, input
        )
        auxiliary_loss = self._reduce_auxiliary_loss(halting_loss)
        loss = self._accumulate_auxiliary_loss(loss, auxiliary_loss)
        return hidden, halting_state, loss

    def __update_output_state(
        self,
        state: LayerState,
        hidden: Tensor,
        halting_state: "HaltingStateBase | None",
        loss: Tensor | None,
    ) -> LayerState:
        state.hidden = hidden
        state.halting_state = halting_state
        state.loss = loss
        return state

    def _handle_model_output(self, layer_state: LayerState) -> LayerState:
        return layer_state

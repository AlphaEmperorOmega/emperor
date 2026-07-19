from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch.nn as nn
from torch import Tensor

from emperor.config import ConfigBase
from emperor.layers._composition.gate import LayerGate
from emperor.layers._composition.residual import ResidualConnection
from emperor.layers._config import GateConfig, RecurrentLayerConfig, ResidualConfig
from emperor.layers._layer import Layer
from emperor.layers._options import (
    LayerNormPositionOptions,
)
from emperor.layers._state import LayerState
from emperor.layers._support import LayerModuleBase
from emperor.layers._validation import RecurrentLayerValidator
from emperor.memory import MemoryPositionOptions

if TYPE_CHECKING:
    from emperor.halting import HaltingConfig, HaltingInterface, HaltingStateBase
    from emperor.layers._stack import LayerStack
    from emperor.memory import DynamicMemoryConfig, MemoryInterface
    from emperor.nn import Module


@dataclass
class _RecurrentState:
    hidden: Tensor
    loss: Tensor | None
    context_state: LayerState
    halting_state: "HaltingStateBase | None" = None


class RecurrentLayer(LayerModuleBase):
    VALIDATOR = RecurrentLayerValidator

    def __init__(
        self,
        cfg: RecurrentLayerConfig,
        overrides: RecurrentLayerConfig | None = None,
    ):
        super().__init__()
        self.cfg: RecurrentLayerConfig = self._override_config(cfg, overrides)
        self.VALIDATOR.validate(self)

        self.input_dim: int = self.cfg.input_dim
        self.output_dim: int = self.cfg.output_dim
        self.max_steps: int = self.cfg.max_steps
        self.recurrent_layer_norm_position: LayerNormPositionOptions = (
            self.cfg.recurrent_layer_norm_position
        )
        self.block_config: ConfigBase | None = self.cfg.block_config
        self.gate_config: GateConfig | None = self.cfg.gate_config
        self.residual_config: ResidualConfig | None = self.cfg.residual_config
        self.halting_config: HaltingConfig | None = self.cfg.halting_config
        self.memory_config: DynamicMemoryConfig | None = self.cfg.memory_config

        self.block_model = self.__build_block_model()
        self.recurrent_gate = self.__build_recurrent_gate()
        self.residual_connection = self.__build_residual_connection()
        self.halting_model = self.__build_halting_model()
        self.memory_model = self.__build_memory_model()
        self.recurrent_layer_norm_module = self.__init_layer_norm_module()

    def __build_block_model(self) -> "Layer | LayerStack | Module":
        return self._build_from_config(
            self.block_config,
            input_dim=self.output_dim,
            output_dim=self.output_dim,
        )

    def __build_recurrent_gate(self) -> LayerGate | None:
        return self._build_from_config(
            self.gate_config,
            gate_dim=self.output_dim,
        )

    def __build_residual_connection(self) -> ResidualConnection | None:
        return self._build_from_config(
            self.residual_config,
            residual_dim=self.output_dim,
        )

    def __build_halting_model(self) -> "HaltingInterface | None":
        return self._build_from_config(
            self.halting_config,
            input_dim=self.output_dim,
        )

    def __build_memory_model(self) -> "MemoryInterface | None":
        return self._build_from_config(
            self.memory_config,
            input_dim=self.output_dim,
            output_dim=self.output_dim,
        )

    def __init_layer_norm_module(self) -> nn.Module | None:
        if self.recurrent_layer_norm_position == LayerNormPositionOptions.DISABLED:
            return None
        return nn.LayerNorm(self.output_dim)

    def forward(self, state: LayerState) -> LayerState:
        self.VALIDATOR.validate_state(state, self.input_dim)

        owner_halting_state = state.halting_state
        recurrent_state = self.__run_recurrent_steps(state)
        recurrent_state = self.__maybe_finalize_recurrent_halting(recurrent_state)
        return self.__reconstruct_layer_state(
            state, recurrent_state, owner_halting_state
        )

    def __run_recurrent_steps(
        self,
        layer_state: LayerState,
    ) -> _RecurrentState:
        recurrent_state = _RecurrentState(
            hidden=layer_state.hidden,
            loss=layer_state.loss,
            context_state=layer_state,
        )
        for _ in range(self.max_steps):
            previous_step_hidden = recurrent_state.hidden
            recurrent_state = self.__run_recurrent_block_step(recurrent_state)
            recurrent_state = self.__run_controllers(
                recurrent_state,
                previous_step_hidden,
            )

            if self.__all_items_halted(recurrent_state.halting_state):
                break

        return recurrent_state

    def __run_recurrent_block_step(
        self,
        recurrent_state: _RecurrentState,
    ) -> _RecurrentState:
        block_input = self.__maybe_apply_layer_norm_before(recurrent_state.hidden)
        block_input = self.__maybe_apply_memory_before_block(block_input)
        block_output_state = self.__process_block(recurrent_state, block_input)
        candidate_hidden = self.__maybe_apply_memory_after_block(
            block_output_state.hidden
        )
        candidate_hidden = self.__maybe_apply_layer_norm_default(candidate_hidden)
        return _RecurrentState(
            hidden=candidate_hidden,
            loss=block_output_state.loss,
            context_state=recurrent_state.context_state,
            halting_state=recurrent_state.halting_state,
        )

    def __maybe_apply_layer_norm_before(self, hidden: Tensor) -> Tensor:
        if self.recurrent_layer_norm_position == LayerNormPositionOptions.BEFORE:
            return self.recurrent_layer_norm_module(hidden)
        return hidden

    def __maybe_apply_memory_before_block(self, hidden: Tensor) -> Tensor:
        return self._maybe_apply_memory_by_position(
            hidden,
            MemoryPositionOptions.BEFORE_AFFINE,
        )

    def __process_block(
        self,
        recurrent_state: _RecurrentState,
        block_input: Tensor,
    ) -> LayerState:
        block_input_state = replace(
            recurrent_state.context_state,
            hidden=block_input,
            loss=recurrent_state.loss,
            halting_state=None,
        )
        block_output_state = self.block_model(block_input_state)
        self.VALIDATOR.validate_candidate(
            block_output_state.hidden,
            recurrent_state.hidden,
            self.output_dim,
        )
        return block_output_state

    def __maybe_apply_memory_after_block(self, hidden: Tensor) -> Tensor:
        return self._maybe_apply_memory_by_position(
            hidden,
            MemoryPositionOptions.AFTER_AFFINE,
        )

    def __maybe_apply_layer_norm_default(self, hidden: Tensor) -> Tensor:
        if self.recurrent_layer_norm_position == LayerNormPositionOptions.DEFAULT:
            return self.recurrent_layer_norm_module(hidden)
        return hidden

    def __run_controllers(
        self,
        recurrent_state: _RecurrentState,
        previous_hidden: Tensor,
    ) -> _RecurrentState:
        gated_hidden = self.__maybe_apply_gate(recurrent_state.hidden)
        residual_hidden = self.__maybe_apply_residual(gated_hidden, previous_hidden)
        normalized_hidden = self.__maybe_apply_layer_norm_after(residual_hidden)
        updated_halting_state, next_recurrent_hidden = (
            self.__maybe_update_halting_state(
                recurrent_state.halting_state,
                normalized_hidden,
                previous_hidden,
            )
        )
        return _RecurrentState(
            hidden=next_recurrent_hidden,
            loss=recurrent_state.loss,
            context_state=recurrent_state.context_state,
            halting_state=updated_halting_state,
        )

    def __maybe_apply_gate(
        self,
        candidate_hidden: Tensor,
    ) -> Tensor:
        if self.recurrent_gate is None:
            return candidate_hidden
        return self.recurrent_gate(candidate_hidden)

    def __maybe_apply_residual(
        self,
        candidate_hidden: Tensor,
        previous_hidden: Tensor,
    ) -> Tensor:
        if self.residual_connection is None:
            return candidate_hidden
        return self.residual_connection(candidate_hidden, previous_hidden)

    def __maybe_apply_layer_norm_after(self, hidden: Tensor) -> Tensor:
        if self.recurrent_layer_norm_position == LayerNormPositionOptions.AFTER:
            return self.recurrent_layer_norm_module(hidden)
        return hidden

    def __maybe_update_halting_state(
        self,
        previous_halting_state: "HaltingStateBase | None",
        candidate_hidden: Tensor,
        previous_step_hidden: Tensor,
    ) -> tuple["HaltingStateBase | None", Tensor]:
        if self.halting_model is None:
            return previous_halting_state, candidate_hidden

        updated_halting_state, halting_output_hidden = (
            self.halting_model.update_halting_state(
                previous_halting_state,
                candidate_hidden,
            )
        )
        self.VALIDATOR.validate_candidate(
            halting_output_hidden,
            previous_step_hidden,
            self.output_dim,
        )
        return updated_halting_state, halting_output_hidden

    def __maybe_finalize_recurrent_halting(
        self,
        recurrent_state: _RecurrentState,
    ) -> _RecurrentState:
        halting_state = recurrent_state.halting_state
        if self.halting_model is None or halting_state is None:
            return recurrent_state

        finalized_hidden, halting_loss = (
            self.halting_model.finalize_weighted_accumulation(
                halting_state,
                recurrent_state.hidden,
            )
        )
        reduced_halting_loss = self._reduce_auxiliary_loss(halting_loss)
        accumulated_loss = self._accumulate_auxiliary_loss(
            recurrent_state.loss,
            reduced_halting_loss,
        )
        return _RecurrentState(
            hidden=finalized_hidden,
            loss=accumulated_loss,
            context_state=recurrent_state.context_state,
            halting_state=halting_state,
        )

    def __reconstruct_layer_state(
        self,
        layer_state: LayerState,
        recurrent_state: _RecurrentState,
        owner_halting_state: "HaltingStateBase | None",
    ) -> LayerState:
        layer_state.hidden = recurrent_state.hidden
        layer_state.loss = recurrent_state.loss
        layer_state.halting_state = owner_halting_state
        return layer_state

    def __all_items_halted(self, halting_state: "HaltingStateBase | None") -> bool:
        if halting_state is None or halting_state.halt_mask is None:
            return False
        all_items_halted = bool(halting_state.halt_mask.all().item())
        return all_items_halted

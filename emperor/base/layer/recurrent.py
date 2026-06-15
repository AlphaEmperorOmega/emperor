import torch
import torch.nn as nn

from dataclasses import dataclass, replace
from torch import Tensor
from emperor.base.options import LayerNormPositionOptions
from emperor.base.utils import ConfigBase

from .layer import Layer
from .state import LayerState
from .base import LayerModuleBase
from ._validator import RecurrentLayerValidator
from .gate import GateConfig, LayerGate
from .config import RecurrentLayerConfig
from .residual import ResidualConnection, ResidualConnectionOptions

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emperor.base.utils import Module
    from emperor.halting.config import HaltingConfig
    from emperor.halting.utils.options.base import HaltingBase, HaltingStateBase
    from emperor.memory.config import DynamicMemoryConfig
    from emperor.memory.core.base import DynamicMemoryAbstract

    from .stack import LayerStack


@dataclass
class _RecurrentState:
    hidden: Tensor
    loss: Tensor | None
    context_state: LayerState
    halting_state: "HaltingStateBase | None" = None


class RecurrentLayer(LayerModuleBase):
    def __init__(
        self,
        cfg: RecurrentLayerConfig,
        overrides: RecurrentLayerConfig | None = None,
    ):
        super().__init__()
        self.cfg: RecurrentLayerConfig = self._override_config(cfg, overrides)
        RecurrentLayerValidator.validate(self.cfg)

        self.input_dim: int = self.cfg.input_dim
        self.output_dim: int = self.cfg.output_dim
        self.max_steps: int = self.cfg.max_steps
        self.recurrent_layer_norm_position: LayerNormPositionOptions = (
            self.cfg.recurrent_layer_norm_position
        )
        self.block_config: ConfigBase | None = self.cfg.block_config
        self.gate_config: GateConfig | None = self.cfg.gate_config
        self.residual_connection_option: ResidualConnectionOptions = (
            self.cfg.residual_connection_option
        )
        self.halting_config: "HaltingConfig | None" = self.cfg.halting_config
        self.memory_config: "DynamicMemoryConfig | None" = self.cfg.memory_config

        self.block_model = self.__build_block_model()
        self.recurrent_gate = self.__build_recurrent_gate()
        self.residual_connection = self.__build_residual_connection()
        self.halting_model = self.__build_halting_model()
        self.memory_model = self.__build_memory_model()
        self.recurrent_layer_norm_module = self.__init_recurrent_layer_norm_module()

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
        if self.residual_connection_option == ResidualConnectionOptions.DISABLED:
            return None
        return ResidualConnection(self.residual_connection_option)

    def __build_halting_model(self) -> "HaltingBase | None":
        return self._build_from_config(
            self.halting_config,
            input_dim=self.output_dim,
        )

    def __build_memory_model(self) -> "DynamicMemoryAbstract | None":
        return self._build_from_config(
            self.memory_config,
            input_dim=self.output_dim,
            output_dim=self.output_dim,
        )

    def __init_recurrent_layer_norm_module(self) -> nn.Module | None:
        if self.recurrent_layer_norm_position == LayerNormPositionOptions.DISABLED:
            return None
        return nn.LayerNorm(self.output_dim)

    def forward(self, state: LayerState) -> LayerState:
        RecurrentLayerValidator.validate_state(state, self.input_dim)

        original_halting_state = state.halting_state
        recurrent_state = self.__run_recurrent_steps(state)
        recurrent_state = self.__maybe_finalize_recurrent_halting(recurrent_state)
        return self.__reconstruct_layer_state(
            state, recurrent_state, original_halting_state
        )

    def __run_recurrent_steps(
        self,
        state: LayerState,
    ) -> _RecurrentState:
        recurrent_state = _RecurrentState(
            hidden=state.hidden, loss=state.loss, context_state=state
        )
        for _ in range(self.max_steps):
            previous_hidden = recurrent_state.hidden
            already_halted_mask = self.__get_halt_mask(recurrent_state.halting_state)
            recurrent_state = self.__run_recurrent_block_step(recurrent_state)
            recurrent_state = self.__run_recurrent_controllers(
                recurrent_state, previous_hidden, already_halted_mask
            )

            if self.__all_items_halted(recurrent_state.halting_state):
                break

        return recurrent_state

    def __get_halt_mask(
        self,
        halting_state: "HaltingStateBase | None",
    ) -> Tensor | None:
        if halting_state is None:
            return None
        return halting_state.halt_mask

    def __run_recurrent_block_step(
        self,
        recurrent_state: _RecurrentState,
    ) -> _RecurrentState:
        block_input = self.__maybe_apply_recurrent_layer_norm_before(
            recurrent_state.hidden
        )
        block_input = self.__maybe_apply_memory_before_block(block_input)
        block_output_state = self.__process_block(recurrent_state, block_input)
        candidate = self.__maybe_apply_memory_after_block(block_output_state.hidden)
        candidate = self.__maybe_apply_recurrent_layer_norm_default(candidate)
        return _RecurrentState(
            hidden=candidate,
            loss=block_output_state.loss,
            context_state=recurrent_state.context_state,
            halting_state=recurrent_state.halting_state,
        )

    def __maybe_apply_recurrent_layer_norm_before(self, hidden: Tensor) -> Tensor:
        if self.recurrent_layer_norm_position == LayerNormPositionOptions.BEFORE:
            return self.recurrent_layer_norm_module(hidden)
        return hidden

    def __maybe_apply_memory_before_block(self, hidden: Tensor) -> Tensor:
        from emperor.memory.options import MemoryPositionOptions

        return self._maybe_apply_memory_by_position(
            hidden, MemoryPositionOptions.BEFORE_AFFINE
        )

    def __process_block(
        self, recurrent_state: _RecurrentState, hidden: Tensor
    ) -> LayerState:
        block_state = self.__build_block_state(recurrent_state, hidden)
        block_output_state = self.block_model(block_state)
        RecurrentLayerValidator.validate_candidate(
            block_output_state.hidden,
            recurrent_state.hidden,
            self.output_dim,
        )
        return block_output_state

    def __build_block_state(
        self, recurrent_state: _RecurrentState, hidden: Tensor
    ) -> LayerState:
        return replace(
            recurrent_state.context_state,
            hidden=hidden,
            loss=recurrent_state.loss,
            halting_state=None,
        )

    def __maybe_apply_memory_after_block(self, hidden: Tensor) -> Tensor:
        from emperor.memory.options import MemoryPositionOptions

        return self._maybe_apply_memory_by_position(
            hidden, MemoryPositionOptions.AFTER_AFFINE
        )

    def __maybe_apply_recurrent_layer_norm_default(self, hidden: Tensor) -> Tensor:
        if self.recurrent_layer_norm_position == LayerNormPositionOptions.DEFAULT:
            return self.recurrent_layer_norm_module(hidden)
        return hidden

    def __run_recurrent_controllers(
        self,
        recurrent_state: _RecurrentState,
        previous_hidden: Tensor,
        already_halted_mask: Tensor | None,
    ) -> _RecurrentState:
        hidden = self.__maybe_apply_recurrent_gate(recurrent_state.hidden)
        hidden = self.__maybe_apply_recurrent_residual(hidden, previous_hidden)
        hidden = self.__preserve_halted_hidden(
            previous_hidden, hidden, already_halted_mask
        )
        halting_state, hidden = self.__maybe_update_recurrent_halting_state(
            recurrent_state.halting_state, hidden, previous_hidden
        )
        hidden = self.__maybe_apply_recurrent_layer_norm_after(hidden)
        hidden = self.__preserve_halted_hidden(
            previous_hidden, hidden, already_halted_mask
        )
        return _RecurrentState(
            hidden=hidden,
            loss=recurrent_state.loss,
            context_state=recurrent_state.context_state,
            halting_state=halting_state,
        )

    def __maybe_apply_recurrent_layer_norm_after(self, hidden: Tensor) -> Tensor:
        if self.recurrent_layer_norm_position == LayerNormPositionOptions.AFTER:
            return self.recurrent_layer_norm_module(hidden)
        return hidden

    def __maybe_update_recurrent_halting_state(
        self,
        halting_state: "HaltingStateBase | None",
        hidden: Tensor,
        previous_hidden: Tensor,
    ) -> tuple["HaltingStateBase | None", Tensor]:
        if self.halting_model is None:
            return halting_state, hidden

        halting_state, halting_hidden = self.halting_model.update_halting_state(
            halting_state,
            hidden,
        )
        RecurrentLayerValidator.validate_candidate(
            halting_hidden,
            previous_hidden,
            self.output_dim,
        )
        return halting_state, halting_hidden

    def __maybe_apply_recurrent_gate(
        self,
        candidate: Tensor,
    ) -> Tensor:
        if self.recurrent_gate is None:
            return candidate
        return self.recurrent_gate(candidate)

    def __maybe_apply_recurrent_residual(
        self,
        candidate: Tensor,
        previous_hidden: Tensor,
    ) -> Tensor:
        if self.residual_connection is None:
            return candidate
        return self.residual_connection(candidate, previous_hidden)

    def __preserve_halted_hidden(
        self,
        previous_hidden: Tensor,
        candidate_hidden: Tensor,
        halt_mask: Tensor | None,
    ) -> Tensor:
        if halt_mask is None:
            return candidate_hidden
        while halt_mask.dim() < candidate_hidden.dim():
            halt_mask = halt_mask.unsqueeze(-1)
        return torch.where(halt_mask, previous_hidden, candidate_hidden)

    def __maybe_finalize_recurrent_halting(
        self,
        run_state: _RecurrentState,
    ) -> _RecurrentState:
        halting_state = run_state.halting_state
        if self.halting_model is None or halting_state is None:
            return run_state

        finalized_hidden, recurrent_loss = (
            self.halting_model.finalize_weighted_accumulation(
                halting_state,
                run_state.hidden,
            )
        )
        recurrent_loss = self._reduce_auxiliary_loss(recurrent_loss)
        return _RecurrentState(
            hidden=finalized_hidden,
            loss=self._accumulate_auxiliary_loss(
                run_state.loss,
                recurrent_loss,
            ),
            context_state=run_state.context_state,
            halting_state=halting_state,
        )

    def __reconstruct_layer_state(
        self,
        state: LayerState,
        run_state: _RecurrentState,
        original_halting_state: "HaltingStateBase | None",
    ) -> LayerState:
        state.hidden = run_state.hidden
        state.loss = run_state.loss
        state.halting_state = original_halting_state
        return state

    def __all_items_halted(self, halting_state: "HaltingStateBase | None") -> bool:
        halt_mask = self.__get_halt_mask(halting_state)
        if halt_mask is None:
            return False
        return bool(halt_mask.all().item())

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import Tensor

from emperor.layers._config import LayerConfig
from emperor.layers._layer.pipeline import (
    LayerHaltingDelegate,
    LayerMemoryDelegate,
    LayerNormalizationDelegate,
    LayerPostprocessingDelegate,
    LayerResidualDelegate,
)
from emperor.layers._layer.validation import LayerValidator
from emperor.layers._state import LayerState
from emperor.layers._support import LayerModuleBase, RowLayoutAwareModule

if TYPE_CHECKING:
    from emperor.halting import HaltingInterface, HaltingStateBase
    from emperor.layers._composition.gate import LayerGate
    from emperor.layers._composition.residual.base import ResidualState
    from emperor.layers._config import GateConfig
    from emperor.layers._row_layout import RowLayout
    from emperor.memory import MemoryInterface
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
        self.VALIDATOR.validate(self)

        self.input_dim: int = self.cfg.input_dim
        self.output_dim: int = self.cfg.output_dim
        self.model = self.__build_model()
        self.postprocessing = LayerPostprocessingDelegate(self.cfg)
        self.halting = LayerHaltingDelegate(self.cfg)
        self.memory = LayerMemoryDelegate(self.cfg)
        self.residual = LayerResidualDelegate(self.cfg)
        self.normalization = LayerNormalizationDelegate(self.cfg)

    def __build_model(self) -> Module:
        model = self._build_from_config(
            self.cfg.layer_model_config,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
        )
        return self.VALIDATOR.validate_layer_model(model)

    @staticmethod
    def run_model_from_hidden(
        model: Module,
        hidden: Tensor,
        *,
        row_layout: RowLayout | None = None,
    ) -> LayerState:
        input_state = LayerState(hidden=hidden, row_layout=row_layout)
        return model(input_state)

    def forward(
        self,
        state: LayerState,
    ) -> LayerState:
        if self.halting.should_skip(state):
            return state
        state = self._handle_layer_input(state)
        state = self._handle_layer_processing(state)
        return self._handle_layer_output(state)

    def _handle_layer_input(self, state: LayerState) -> LayerState:
        return state

    def _handle_layer_processing(self, state: LayerState) -> LayerState:
        state, saved_layout, residual = self.__setup_pipeline(state)
        state = self.normalization.before_model(state)
        state = self.memory.before_model(state)
        state = self._handle_model_processing(state)
        state = self.memory.after_model(state)
        state = self.normalization.after_model(state)
        state = self.postprocessing.process(state)
        state = self.residual.apply_residual(state, residual)
        state = self.normalization.after_residual(state)
        state = self.halting.apply_halting(state)
        return self.__finalize_pipeline(state, saved_layout)

    def __setup_pipeline(
        self,
        state: LayerState,
    ) -> tuple[LayerState, RowLayout | None, Tensor]:
        saved_layout = state.row_layout
        state.row_layout = self.halting.restrict_row_layout(state.row_layout)
        residual = state.hidden
        return state, saved_layout, residual

    def _handle_model_processing(
        self,
        state: LayerState,
    ) -> LayerState:
        if isinstance(self.model, RowLayoutAwareModule):
            state.hidden = self.model(state.hidden, row_layout=state.row_layout)
            return state
        state.hidden = self.model(state.hidden)
        return state

    def __finalize_pipeline(
        self,
        state: LayerState,
        saved_layout: RowLayout | None,
    ) -> LayerState:
        state.row_layout = saved_layout
        return state

    def _handle_layer_output(self, state: LayerState) -> LayerState:
        return state

    def _bind_shared_gate(self, config: GateConfig, model: LayerGate) -> None:
        self.postprocessing.bind_shared_gate(config, model)

    def _bind_shared_halting(
        self,
        model: HaltingInterface[HaltingStateBase],
    ) -> None:
        self.halting.bind_shared(model)

    def _bind_shared_memory(self, model: MemoryInterface) -> None:
        self.memory.bind_shared(model)

    def _mark_as_last_layer(self) -> None:
        self.halting.mark_as_terminal_layer()

    def _new_residual_state(
        self,
        initial_source: Tensor,
    ) -> ResidualState | None:
        return self.residual.new_state(initial_source)

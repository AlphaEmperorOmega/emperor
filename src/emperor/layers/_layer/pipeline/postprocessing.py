from __future__ import annotations

from typing import TYPE_CHECKING

from torch import Tensor, nn

from emperor.layers._composition.gate import LayerGate
from emperor.layers._layer.validation import LayerPostprocessingDelegateValidator
from emperor.layers._options import ActivationOptions
from emperor.nn import Module

if TYPE_CHECKING:
    from emperor.layers._config import GateConfig, LayerConfig
    from emperor.layers._row_layout import RowLayout
    from emperor.layers._state import LayerState


class LayerPostprocessingDelegate(Module):
    """Own activation, gate, and dropout construction and ordering."""

    VALIDATOR = LayerPostprocessingDelegateValidator

    def __init__(
        self,
        cfg: LayerConfig,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.VALIDATOR.validate(self)
        self.__initialize_from_config()
        self.gate = self.__build_gate()
        self.dropout = self.__build_dropout()

    def __initialize_from_config(self) -> None:
        self.activation_function: ActivationOptions = self.cfg.activation
        self.output_dim: int = self.cfg.output_dim
        self.gate_config = self.cfg.gate_config
        self.dropout_probability: float = self.cfg.dropout_probability

    def __build_gate(self) -> LayerGate | None:
        gate = self._build_from_config(
            self.gate_config,
            gate_dim=self.output_dim,
        )
        return self.VALIDATOR.validate_built_gate_type(gate)

    def __build_dropout(self) -> nn.Dropout | None:
        dropout_is_enabled = self.dropout_probability > 0.0
        if dropout_is_enabled:
            return nn.Dropout(self.dropout_probability)
        return None

    def bind_shared_gate(self, config: GateConfig, gate: LayerGate) -> None:
        self.gate_config = config
        self.gate = gate

    def process(self, state: LayerState) -> LayerState:
        hidden = self.__maybe_apply_activation(state.hidden)
        hidden = self.__maybe_apply_gate(hidden, row_layout=state.row_layout)
        hidden = self.__maybe_apply_dropout(hidden)
        state.hidden = hidden
        return state

    def __maybe_apply_activation(self, hidden: Tensor) -> Tensor:
        if self.activation_function != ActivationOptions.DISABLED:
            return self.activation_function(hidden)
        return hidden

    def __maybe_apply_gate(
        self,
        hidden: Tensor,
        row_layout: RowLayout | None,
    ) -> Tensor:
        if self.gate is not None:
            return self.gate(hidden, row_layout=row_layout)
        return hidden

    def __maybe_apply_dropout(self, hidden: Tensor) -> Tensor:
        if self.dropout is not None:
            return self.dropout(hidden)
        return hidden

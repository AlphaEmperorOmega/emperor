from __future__ import annotations

from typing import TYPE_CHECKING

from torch import Tensor, nn

from emperor.layers._composition.gate import LayerGate
from emperor.layers._options import ActivationOptions
from emperor.nn import Module

if TYPE_CHECKING:
    from emperor.layers._config import GateConfig, LayerConfig
    from emperor.layers._row_layout import RowLayout
    from emperor.layers._state import LayerState


class LayerPostprocessingDelegate(Module):
    """Own activation, gate, and dropout construction and ordering."""

    def __init__(
        self,
        layer_config: LayerConfig,
    ) -> None:
        super().__init__()
        activation_function = layer_config.activation
        output_dim = layer_config.output_dim
        dropout_probability = layer_config.dropout_probability
        if activation_function is None:
            raise ValueError("Layer postprocessing requires a resolved activation.")
        if output_dim is None:
            raise ValueError("Layer postprocessing requires a resolved output_dim.")
        if dropout_probability is None:
            raise ValueError(
                "Layer postprocessing requires a resolved dropout_probability."
            )
        self.activation_function = activation_function
        self.output_dim = output_dim
        self.gate_config = layer_config.gate_config
        self.gate = self.__build_gate()
        self.dropout_probability = dropout_probability
        self.dropout = self.__build_dropout()

    def __build_gate(self) -> LayerGate | None:
        gate = self._build_from_config(
            self.gate_config,
            gate_dim=self.output_dim,
        )
        if gate is None:
            return None
        if not isinstance(gate, LayerGate):
            raise TypeError("gate_config must build a LayerGate.")
        return gate

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

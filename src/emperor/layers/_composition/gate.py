from typing import TYPE_CHECKING

from torch import Tensor

from emperor.layers._config import GateConfig
from emperor.layers._options import ActivationOptions, LayerGateOptions
from emperor.layers._validation import LayerGateValidator
from emperor.nn import Module

if TYPE_CHECKING:
    from emperor.layers._state import LayerState
    from emperor.nn import Module as EmperorModule


class LayerGate(Module):
    VALIDATOR = LayerGateValidator

    def __init__(
        self,
        cfg: GateConfig,
        overrides: GateConfig | None = None,
    ):
        super().__init__()
        self.cfg: GateConfig = self._override_config(cfg, overrides)

        self.VALIDATOR.validate(self)
        self.option: LayerGateOptions = self.cfg.option
        self.activation: ActivationOptions | None = self.cfg.activation
        self.gate_dim: int | None = self.cfg.gate_dim
        self.model = self.__build_model()

    def __build_model(self) -> "EmperorModule":
        return self._build_from_config(
            self.cfg.model_config,
            input_dim=self.gate_dim,
            output_dim=self.gate_dim,
        )

    def effective_values(self, gate_output: Tensor) -> Tensor:
        if self.activation is None or self.activation == ActivationOptions.DISABLED:
            return gate_output
        return self.activation(gate_output)

    def forward(self, current: Tensor) -> Tensor:
        gate_output = self.__run_gate_model(current)
        self.VALIDATOR.validate_gate_output(gate_output, current, self.option)
        gate = self.effective_values(gate_output)
        if self.option == LayerGateOptions.MULTIPLIER:
            return gate * current
        if self.option == LayerGateOptions.ADDITION:
            return current + gate
        raise ValueError(f"Unsupported gate option {self.option} for LayerGate.")

    def __run_gate_model(self, current: Tensor) -> Tensor:
        self.VALIDATOR.validate_gate_model(self.model)
        gate_state = self.__gate_state(current)
        output = self.model(gate_state)
        return output.hidden if hasattr(output, "hidden") else output

    @staticmethod
    def __gate_state(current: Tensor) -> "LayerState":
        from emperor.layers._state import LayerState

        return LayerState(hidden=current)

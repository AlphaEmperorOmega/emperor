from typing import TYPE_CHECKING

from torch import Tensor

from emperor.neuron._axons.validation import AxonsValidator
from emperor.nn import Module

if TYPE_CHECKING:
    from emperor.neuron._config import AxonsConfig


class Axons(Module):
    VALIDATOR = AxonsValidator

    def __init__(
        self,
        cfg: "AxonsConfig",
        overrides: "AxonsConfig | None" = None,
    ):
        super().__init__()
        axons_config = getattr(cfg, "axons_config", cfg)
        self.cfg: AxonsConfig = self._override_config(axons_config, overrides)
        self.memory_config = self.cfg.memory_config
        self.VALIDATOR.validate(self)
        self.memory_model = self.__maybe_build_memory_model()

    def __maybe_build_memory_model(self) -> Module | None:
        if self.memory_config is None:
            return None
        return self._build_from_config(
            self.memory_config,
            input_dim=self.memory_config.input_dim,
            output_dim=self.memory_config.input_dim,
        )

    def forward(self, input: Tensor) -> Tensor:
        self.VALIDATOR.validate_forward_input(input)
        if self.memory_model is None:
            return input
        return self.memory_model(input)

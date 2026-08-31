from typing import TYPE_CHECKING

from torch import Tensor

from emperor.neuron._nucleus.validation import NucleusValidator
from emperor.nn import Module

if TYPE_CHECKING:
    from emperor.neuron._config import NucleusConfig


class Nucleus(Module):
    VALIDATOR = NucleusValidator

    def __init__(
        self,
        cfg: "NucleusConfig",
        overrides: "NucleusConfig | None" = None,
    ):
        super().__init__()
        self.cfg: NucleusConfig = self._override_config(cfg, overrides)
        self.model_config = self.cfg.model_config
        self.VALIDATOR.validate(self)
        self.model = self.model_config.build()

    def forward(self, input: Tensor) -> Tensor:
        self.VALIDATOR.validate_forward_input(input)
        return self.model(input)

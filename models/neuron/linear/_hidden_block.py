from dataclasses import dataclass, fields

from emperor.base.layer import Layer
from emperor.base.utils import ConfigBase, Module, optional_field
from torch import Tensor


@dataclass
class HiddenBlockConfig(ConfigBase):
    input_dim: int | None = optional_field(
        "Input feature dimension for the package-local hidden block Adapter."
    )
    output_dim: int | None = optional_field(
        "Output feature dimension for the package-local hidden block Adapter."
    )
    model_config: ConfigBase | None = optional_field(
        "Package-local hidden block config wrapped by the neuron nucleus."
    )

    def _registry_owner(self) -> type:
        return HiddenBlockAdapter


class HiddenBlockAdapter(Module):
    def __init__(
        self,
        cfg: HiddenBlockConfig,
        overrides: HiddenBlockConfig | None = None,
    ) -> None:
        super().__init__()
        self.cfg: HiddenBlockConfig = self._override_config(cfg, overrides)
        self.input_dim = self.cfg.input_dim
        self.output_dim = self.cfg.output_dim
        self.model_config: ConfigBase = self.cfg.model_config
        self.model = self.model_config.build(
            overrides=self._dimension_overrides(self.model_config)
        )

    def _dimension_overrides(self, model_config: ConfigBase):
        declared_fields = {field.name for field in fields(model_config)}
        overrides = {}
        if "input_dim" in declared_fields:
            overrides["input_dim"] = self.input_dim
        if "output_dim" in declared_fields:
            overrides["output_dim"] = self.output_dim
        if not overrides:
            return None
        return type(model_config)(**overrides)

    def forward(self, input: Tensor) -> Tensor:
        state = Layer.run_model_returning_state(self.model, input)
        return state.hidden

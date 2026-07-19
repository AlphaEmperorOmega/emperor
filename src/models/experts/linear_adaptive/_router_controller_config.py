from __future__ import annotations

from dataclasses import dataclass

from emperor.base.layer.state import LayerState
from emperor.base.config import ConfigBase, optional_field
from emperor.base.module import Module


@dataclass
class RouterControllerModelConfig(ConfigBase):
    input_dim: int | None = optional_field("External router input dimension.")
    hidden_dim: int | None = optional_field("Stable router trunk hidden dimension.")
    output_dim: int | None = optional_field("Router logits output dimension.")
    adapter_config: ConfigBase | None = optional_field(
        "Optional projection used when input_dim differs from hidden_dim."
    )
    trunk_config: ConfigBase | None = optional_field(
        "Stable router trunk config with optional controllers."
    )
    head_config: ConfigBase | None = optional_field("Final router logits head config.")

    def _registry_owner(self) -> type:
        return RouterControllerModel


class RouterControllerModel(Module):
    def __init__(
        self,
        cfg: RouterControllerModelConfig,
        overrides: RouterControllerModelConfig | None = None,
    ) -> None:
        super().__init__()
        self.cfg: RouterControllerModelConfig = self._override_config(cfg, overrides)
        self.__validate_config()

        self.input_dim = self.cfg.input_dim
        self.hidden_dim = self.cfg.hidden_dim
        self.output_dim = self.cfg.output_dim
        self.adapter_config = self.cfg.adapter_config
        self.trunk_config = self.cfg.trunk_config
        self.head_config = self.cfg.head_config

        self.adapter = self.__build_adapter()
        self.trunk = self.__build_trunk()
        self.head = self.__build_head()

    def __validate_config(self) -> None:
        for field_name in ("input_dim", "hidden_dim", "output_dim"):
            value = getattr(self.cfg, field_name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(
                    f"{field_name} must be an int for RouterControllerModelConfig, "
                    f"got {type(value).__name__}."
                )
            if value <= 0:
                raise ValueError(
                    f"{field_name} must be a positive integer, received {value!r}."
                )

        for field_name in ("trunk_config", "head_config"):
            value = getattr(self.cfg, field_name)
            if not isinstance(value, ConfigBase):
                raise TypeError(
                    f"{field_name} must be a ConfigBase for "
                    f"RouterControllerModelConfig, got {type(value).__name__}."
                )

        if self.cfg.input_dim != self.cfg.hidden_dim and not isinstance(
            self.cfg.adapter_config,
            ConfigBase,
        ):
            raise TypeError(
                "adapter_config must be a ConfigBase when router input_dim differs "
                "from hidden_dim."
            )

    def __build_adapter(self) -> Module | None:
        if self.input_dim == self.hidden_dim:
            return None
        return self._build_from_config(
            self.adapter_config,
            input_dim=self.input_dim,
            output_dim=self.hidden_dim,
        )

    def __build_trunk(self) -> Module:
        return self._build_from_config(
            self.trunk_config,
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
        )

    def __build_head(self) -> Module:
        return self._build_from_config(
            self.head_config,
            input_dim=self.hidden_dim,
            output_dim=self.output_dim,
        )

    def forward(self, state: LayerState) -> LayerState:
        if self.adapter is not None:
            state = self.adapter(state)
        state = self.trunk(state)
        return self.head(state)

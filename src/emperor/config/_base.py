from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from emperor.nn import Module


def optional_field(help: str):
    return field(default=None, metadata={"help": help})


@dataclass
class ConfigBase:
    def _registry_owner(self) -> type:
        raise NotImplementedError(
            f"{type(self).__name__} must implement `_registry_owner` "
            f"or override `build`"
        )

    def registry_owner(self) -> type:
        return self._registry_owner()

    def build(self, overrides: ConfigBase | None = None) -> Module:
        owner = self.registry_owner()
        if hasattr(self, "model_type"):
            return owner.build_from_config(self, overrides)
        return owner(self, overrides)

    def get(self, key: str, default=None) -> Any:
        if not hasattr(self, key):
            return default

        return getattr(self, key, default)

    def __post_init__(self):
        self._passed_args: dict[str, Any] = {}
        for config_field in fields(self):
            if config_field.name == "passed_args":
                continue
            value = getattr(self, config_field.name)
            if (
                config_field.default is not None and value != config_field.default
            ) or isinstance(value, bool):
                self._passed_args[config_field.name] = value

    def get_custom_parameters(self) -> dict[str, Any]:
        return self._passed_args

    def update(self, other: ConfigBase) -> ConfigBase:
        for config_field in fields(other):
            key = config_field.name
            value = getattr(other, key)
            if value is not None:
                setattr(self, key, value)
        return self

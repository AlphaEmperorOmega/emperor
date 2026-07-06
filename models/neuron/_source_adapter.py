from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any


@dataclass(frozen=True)
class SourcePackageAdapter:
    config_module: ModuleType
    builder_type: type
    experiment_preset_type: type
    experiment_presets_type: type
    builder_kwargs_from_flat_fn: Callable[[dict[str, Any], ModuleType], dict[str, Any]]
    flat_defaults_fn: Callable[[ModuleType], dict[str, Any]] | None = None
    kwarg_aliases: dict[str, str] = field(default_factory=dict)

    def canonical_kwarg_aliases(self) -> dict[str, str]:
        return dict(self.kwarg_aliases)

    def normalize_source_kwargs(self, source_kwargs: dict[str, Any]) -> dict[str, Any]:
        return {
            self.kwarg_aliases.get(key, key): value
            for key, value in source_kwargs.items()
        }

    def builder_kwargs_from_flat(
        self,
        source_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        return self.builder_kwargs_from_flat_fn(
            self.normalize_source_kwargs(source_kwargs),
            self.config_module,
        )

    def build_source_config(self, source_kwargs: dict[str, Any]):
        builder_kwargs = self.builder_kwargs_from_flat(source_kwargs)
        return self.builder_type(**builder_kwargs).build()

    def source_default_kwargs(self) -> dict[str, Any]:
        if self.flat_defaults_fn is None:
            return {}
        return self.flat_defaults_fn(self.config_module)

    def source_preset(self, preset):
        return self.experiment_preset_type[preset.name]

    def source_values_for_preset(self, preset) -> dict[str, object]:
        source_presets = self.experiment_presets_type()
        return source_presets.overrides_for_preset(self.source_preset(preset))

    def source_locks_for_preset(self, preset) -> dict[str, object]:
        source_presets = self.experiment_presets_type()
        return dict(source_presets.locks_for_preset(self.source_preset(preset)))

    def source_description_for_preset(self, preset) -> str:
        source_presets = self.experiment_presets_type()
        return source_presets.description_for_preset(self.source_preset(preset))

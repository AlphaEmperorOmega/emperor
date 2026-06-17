from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from lightning.pytorch.callbacks import Callback


MonitorKind = str


@dataclass(frozen=True)
class MonitorOption:
    name: str
    label: str
    description: str
    kinds: Sequence[MonitorKind]
    callback_factory: Callable[[], Callback]
    default_enabled: bool = False

    def build_callback(self) -> Callback:
        return self.callback_factory()

    def to_api(self) -> dict[str, object]:
        return {
            "name": self.name,
            "label": self.label,
            "description": self.description,
            "kinds": list(self.kinds),
            "defaultEnabled": self.default_enabled,
        }

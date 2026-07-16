from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lightning.pytorch.callbacks import Callback

MonitorKind = str


@dataclass(frozen=True)
class MonitorSettings:
    log_every_n_steps: int = 100

    def __post_init__(self) -> None:
        if self.log_every_n_steps < 1:
            raise ValueError("monitor log cadence must be at least one step.")


@dataclass(frozen=True)
class MonitorOption:
    name: str
    label: str
    description: str
    kinds: Sequence[MonitorKind]
    callback_factory: Callable[[MonitorSettings], Callback]
    default_enabled: bool = False

    def build_callback(self, settings: MonitorSettings | None = None) -> Callback:
        return self.callback_factory(settings or MonitorSettings())

    def to_api(self) -> dict[str, object]:
        return {
            "name": self.name,
            "label": self.label,
            "description": self.description,
            "kinds": list(self.kinds),
            "defaultEnabled": self.default_enabled,
        }

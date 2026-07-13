from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from workbench.backend.training_jobs.service import TrainingJobService

__all__ = ["TrainingJobService"]


def __getattr__(name: str) -> Any:
    if name != "TrainingJobService":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from workbench.backend.training_jobs.service import TrainingJobService

    globals()[name] = TrainingJobService
    return TrainingJobService

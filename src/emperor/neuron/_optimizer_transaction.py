"""Atomic optimizer mutations used by current Neuron checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch.optim import Optimizer


@dataclass(frozen=True)
class _OptimizerGroupSnapshot:
    group: dict[str, Any]
    original_values: dict[str, Any]
    original_lists: dict[str, tuple[list[Any], tuple[Any, ...]]]

    @classmethod
    def capture(cls, group: dict[str, Any]) -> _OptimizerGroupSnapshot:
        return cls(
            group=group,
            original_values=dict(group),
            original_lists={
                name: (value, tuple(value))
                for name, value in group.items()
                if isinstance(value, list)
            },
        )

    def restore(self) -> None:
        self.group.clear()
        for name, value in self.original_values.items():
            list_snapshot = self.original_lists.get(name)
            if list_snapshot is None:
                self.group[name] = value
                continue
            original_list, original_items = list_snapshot
            original_list[:] = original_items
            self.group[name] = original_list


@dataclass
class _OptimizerLoadSnapshot:
    optimizer: Optimizer
    original_param_groups: list[dict[str, Any]]
    groups: tuple[_OptimizerGroupSnapshot, ...]
    original_state: dict[Any, Any]
    original_state_items: tuple[tuple[Any, Any], ...]
    optimizer_loaded: bool = False

    @classmethod
    def capture(cls, optimizer: Optimizer) -> _OptimizerLoadSnapshot:
        return cls(
            optimizer=optimizer,
            original_param_groups=optimizer.param_groups,
            groups=tuple(
                _OptimizerGroupSnapshot.capture(group)
                for group in optimizer.param_groups
            ),
            original_state=optimizer.state,
            original_state_items=tuple(optimizer.state.items()),
        )

    def restore(self) -> None:
        for group in self.groups:
            group.restore()
        self.original_param_groups[:] = [group.group for group in self.groups]
        self.optimizer.param_groups = self.original_param_groups
        self.original_state.clear()
        self.original_state.update(self.original_state_items)
        self.optimizer.state = self.original_state


class NeuronOptimizerLoadTransaction:
    """Retain the complete pre-load optimizer state until fit can start."""

    def __init__(self) -> None:
        self._snapshots: list[_OptimizerLoadSnapshot] = []

    def prepare_for_load(self, optimizers: list[Optimizer]) -> None:
        self.clear()
        self._snapshots = [
            _OptimizerLoadSnapshot.capture(optimizer) for optimizer in optimizers
        ]

    def optimizer_requires_completion(self, optimizer: Optimizer) -> bool:
        return any(snapshot.optimizer is optimizer for snapshot in self._snapshots)

    def mark_optimizer_loaded(self, optimizer: Optimizer) -> None:
        for snapshot in self._snapshots:
            if snapshot.optimizer is optimizer:
                snapshot.optimizer_loaded = True

    def commit_loaded(self) -> None:
        if self._snapshots and not all(
            snapshot.optimizer_loaded for snapshot in self._snapshots
        ):
            raise RuntimeError(
                "Cannot commit a partial Neuron optimizer checkpoint load."
            )
        self._snapshots.clear()

    def commit(self) -> None:
        """Keep current optimizer mutations and discard rollback snapshots."""

        self._snapshots.clear()

    def clear(self) -> None:
        for snapshot in reversed(self._snapshots):
            snapshot.restore()
        self._snapshots.clear()


__all__ = ["NeuronOptimizerLoadTransaction"]

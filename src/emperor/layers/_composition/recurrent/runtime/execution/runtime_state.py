"""Transactional runtime-state isolation for recurrent execution."""

# PyTorch's public buffer iteration omits registered None buffers and persistence
# metadata. Exact identity-preserving rollback therefore uses Module registries.
# pyright: reportPrivateUsage=false

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from collections.abc import Generator


@dataclass(frozen=True)
class _BufferSnapshot:
    buffer: Tensor
    value: Tensor


@dataclass(frozen=True)
class _NamedBufferSnapshot:
    name: str
    snapshot: _BufferSnapshot | None


@dataclass(frozen=True)
class _ModuleBufferSnapshot:
    module: nn.Module
    buffers: tuple[_NamedBufferSnapshot, ...]
    non_persistent_buffer_names: frozenset[str]


@dataclass(frozen=True)
class _RecurrentRuntimeStateSnapshot:
    module_buffers: tuple[_ModuleBufferSnapshot, ...]
    cpu_rng_state: Tensor
    cuda_rng_states: tuple[Tensor, ...] | None


class RecurrentRuntimeStateGuard:
    """Isolate mutable module and RNG state across recurrent branch execution."""

    @contextmanager
    def isolate_provisional_branch(
        self,
        recurrent_module: nn.Module,
    ) -> Generator[None, None, None]:
        """Restore runtime state after a provisional branch finishes or fails."""
        runtime_state_snapshot = self.__snapshot_runtime_state(recurrent_module)
        try:
            yield
        finally:
            self.__restore_runtime_state(runtime_state_snapshot)

    @contextmanager
    def rollback_handoff_on_failure(
        self,
        recurrent_module: nn.Module,
    ) -> Generator[None, None, None]:
        """Restore pre-handoff runtime state only when the handoff fails."""
        runtime_state_snapshot = self.__snapshot_runtime_state(recurrent_module)
        try:
            yield
        except BaseException:
            self.__restore_runtime_state(runtime_state_snapshot)
            raise

    def __snapshot_runtime_state(
        self,
        recurrent_module: nn.Module,
    ) -> _RecurrentRuntimeStateSnapshot:
        return _RecurrentRuntimeStateSnapshot(
            module_buffers=self.__snapshot_module_buffers(recurrent_module),
            cpu_rng_state=torch.get_rng_state(),
            cuda_rng_states=(
                tuple(torch.cuda.get_rng_state_all())
                if torch.cuda.is_initialized()
                else None
            ),
        )

    @staticmethod
    def __restore_runtime_state(snapshot: _RecurrentRuntimeStateSnapshot) -> None:
        RecurrentRuntimeStateGuard.__restore_module_buffers(snapshot.module_buffers)
        torch.set_rng_state(snapshot.cpu_rng_state)
        if snapshot.cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(list(snapshot.cuda_rng_states))

    @staticmethod
    def __snapshot_module_buffers(
        recurrent_module: nn.Module,
    ) -> tuple[_ModuleBufferSnapshot, ...]:
        return tuple(
            _ModuleBufferSnapshot(
                module=module,
                buffers=tuple(
                    _NamedBufferSnapshot(
                        name=name,
                        snapshot=(
                            None
                            if buffer is None
                            else _BufferSnapshot(
                                buffer=buffer,
                                value=buffer.detach().clone(),
                            )
                        ),
                    )
                    for name, buffer in module._buffers.items()
                ),
                non_persistent_buffer_names=frozenset(
                    module._non_persistent_buffers_set
                ),
            )
            for module in recurrent_module.modules()
        )

    @staticmethod
    def __restore_module_buffers(
        snapshots: tuple[_ModuleBufferSnapshot, ...],
    ) -> None:
        for snapshot in snapshots:
            snapshot.module._buffers.clear()
            snapshot.module._non_persistent_buffers_set.clear()
            snapshot.module._non_persistent_buffers_set.update(
                snapshot.non_persistent_buffer_names
            )
            for named_buffer_snapshot in snapshot.buffers:
                buffer_snapshot = named_buffer_snapshot.snapshot
                if buffer_snapshot is None:
                    snapshot.module._buffers[named_buffer_snapshot.name] = None
                else:
                    with torch.no_grad():
                        buffer_snapshot.buffer.copy_(buffer_snapshot.value)
                    snapshot.module._buffers[named_buffer_snapshot.name] = (
                        buffer_snapshot.buffer
                    )

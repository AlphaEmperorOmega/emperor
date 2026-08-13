from __future__ import annotations

from pathlib import Path
from typing import cast

from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint


class _CheckpointWriter(ModelCheckpoint):
    """Typed access to Lightning's pinned checkpoint-writing operation."""

    @staticmethod
    def save(callback: ModelCheckpoint, trainer: Trainer, target: str) -> None:
        cast(_CheckpointWriter, callback)._save_checkpoint(trainer, target)


def save_terminal_last_checkpoint(
    trainer: Trainer,
    checkpoint_callback: ModelCheckpoint | None,
) -> str | None:
    """Force one full-state, exact-path last checkpoint after successful fit."""

    if checkpoint_callback is None:
        return None
    if checkpoint_callback.save_last is not True:
        raise RuntimeError(
            "Terminal checkpoint saving requires ModelCheckpoint.save_last=True."
        )
    if checkpoint_callback.save_weights_only:
        raise RuntimeError(
            "Terminal checkpoint saving requires a full-state ModelCheckpoint."
        )
    if checkpoint_callback.dirpath is None:
        raise RuntimeError(
            "Terminal checkpoint saving requires Lightning to resolve the "
            "ModelCheckpoint directory."
        )

    target = checkpoint_callback.format_checkpoint_name(
        {},
        checkpoint_callback.CHECKPOINT_NAME_LAST,
    )
    expected_target = Path(checkpoint_callback.dirpath) / "last.ckpt"
    if Path(target) != expected_target:
        raise RuntimeError(
            "Terminal checkpoint naming must resolve to the exact configured "
            "path 'checkpoints/last.ckpt'."
        )

    checkpoint_callback.last_model_path = target
    _CheckpointWriter.save(checkpoint_callback, trainer, target)
    return target


__all__ = ["save_terminal_last_checkpoint"]

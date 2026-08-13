from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from lightning.pytorch.callbacks import ModelCheckpoint

from model_runtime.runs._terminal_checkpoint import save_terminal_last_checkpoint


class _CheckpointContractTrainer:
    def __init__(self, global_step: int = 7) -> None:
        self.global_step = global_step
        self.is_global_zero = False
        self.loggers: tuple[object, ...] = ()
        self.save_calls: list[tuple[str, bool]] = []

    def save_checkpoint(self, path: str, weights_only: bool) -> None:
        self.save_calls.append((path, weights_only))


class TerminalCheckpointTests(unittest.TestCase):
    def test_none_callback_is_a_no_op(self) -> None:
        trainer = _CheckpointContractTrainer()

        self.assertIsNone(save_terminal_last_checkpoint(trainer, None))
        self.assertEqual(trainer.save_calls, [])

    def test_non_terminal_callback_contracts_fail_closed(self) -> None:
        cases = (
            (ModelCheckpoint(save_last=False), "save_last=True"),
            (ModelCheckpoint(save_last="link"), "save_last=True"),
            (
                ModelCheckpoint(save_last=True, save_weights_only=True),
                "full-state",
            ),
        )

        for callback, message in cases:
            with (
                self.subTest(save_last=callback.save_last),
                self.assertRaisesRegex(RuntimeError, message),
            ):
                save_terminal_last_checkpoint(
                    _CheckpointContractTrainer(),
                    callback,
                )

    def test_unresolved_checkpoint_directory_fails_closed(self) -> None:
        callback = ModelCheckpoint(save_last=True)

        with self.assertRaisesRegex(RuntimeError, "resolve.*directory"):
            save_terminal_last_checkpoint(_CheckpointContractTrainer(), callback)

    def test_exact_target_and_callback_state_are_set_before_serialization(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp) / "checkpoints"
            callback = ModelCheckpoint(
                dirpath=checkpoint_dir,
                save_last=True,
            )
            trainer = _CheckpointContractTrainer()
            observed_paths: list[str] = []

            def serialize(_trainer: object, path: str) -> None:
                self.assertEqual(callback.last_model_path, path)
                observed_paths.append(path)

            with patch.object(
                callback,
                "_save_checkpoint",
                side_effect=serialize,
            ) as save:
                target = save_terminal_last_checkpoint(trainer, callback)

            expected = str(checkpoint_dir / "last.ckpt")
            self.assertEqual(target, expected)
            self.assertEqual(observed_paths, [expected])
            save.assert_called_once_with(trainer, expected)

    def test_nonstandard_last_name_fails_before_serialization(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            callback = ModelCheckpoint(dirpath=tmp, save_last=True)
            with (
                patch.object(
                    callback,
                    "format_checkpoint_name",
                    return_value=str(Path(tmp) / "terminal.ckpt"),
                ),
                patch.object(callback, "_save_checkpoint") as save,
                self.assertRaisesRegex(RuntimeError, "exact configured path"),
            ):
                save_terminal_last_checkpoint(
                    _CheckpointContractTrainer(),
                    callback,
                )

            save.assert_not_called()

    def test_lightning_save_contract_runs_on_nonzero_rank_and_updates_state(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            callback = ModelCheckpoint(
                dirpath=Path(tmp) / "checkpoints",
                save_last=True,
            )
            trainer = _CheckpointContractTrainer(global_step=13)

            target = save_terminal_last_checkpoint(trainer, callback)

            self.assertIsNotNone(target)
            self.assertEqual(trainer.save_calls, [(target, False)])
            self.assertEqual(callback.last_model_path, target)
            self.assertEqual(callback._last_checkpoint_saved, target)
            self.assertEqual(callback._last_global_step_saved, 13)

    def test_serialization_failure_propagates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            callback = ModelCheckpoint(dirpath=tmp, save_last=True)
            with (
                patch.object(
                    callback,
                    "_save_checkpoint",
                    side_effect=OSError("disk full"),
                ),
                self.assertRaisesRegex(OSError, "disk full"),
            ):
                save_terminal_last_checkpoint(
                    _CheckpointContractTrainer(),
                    callback,
                )


if __name__ == "__main__":
    unittest.main()

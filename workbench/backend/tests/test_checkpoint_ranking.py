from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from workbench.backend.historical_inspection._checkpoint_ranking import (
    rank_historical_checkpoints,
)
from workbench.backend.historical_inspection._checkpoint_shapes import (
    load_checkpoint_graph_shapes,
)
from workbench.backend.run_history.contracts import HistoricalCheckpointCandidate


def _candidate(path: Path, *, modified_at_ns: int) -> HistoricalCheckpointCandidate:
    return HistoricalCheckpointCandidate(
        path=path,
        size_bytes=path.stat().st_size if path.exists() else 1,
        modified_at_ns=modified_at_ns,
    )


class HistoricalCheckpointRankingTests(unittest.TestCase):
    def test_last_then_step_then_epoch_then_mtime_and_path(self) -> None:
        root = Path("/run/checkpoints")
        candidates = (
            _candidate(root / "epoch=10-step=99.ckpt", modified_at_ns=8),
            _candidate(root / "epoch=1-step=100.ckpt", modified_at_ns=1),
            _candidate(root / "epoch=9-step=99.ckpt", modified_at_ns=9),
            _candidate(root / "epoch=10-step=99-b.ckpt", modified_at_ns=8),
            _candidate(root / "last.ckpt", modified_at_ns=0),
        )

        ranked = rank_historical_checkpoints(candidates)

        self.assertEqual(
            [candidate.path.name for candidate in ranked],
            [
                "last.ckpt",
                "epoch=1-step=100.ckpt",
                "epoch=10-step=99-b.ckpt",
                "epoch=10-step=99.ckpt",
                "epoch=9-step=99.ckpt",
            ],
        )

    def test_corrupt_last_checkpoint_falls_through_to_ranked_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            last = root / "last.ckpt"
            step_99 = root / "epoch=9-step=99.ckpt"
            step_100 = root / "epoch=1-step=100.ckpt"
            for checkpoint in (last, step_99, step_100):
                checkpoint.write_bytes(b"x")
            ranked = rank_historical_checkpoints(
                tuple(
                    _candidate(path, modified_at_ns=path.stat().st_mtime_ns)
                    for path in (step_99, step_100, last)
                )
            )

            def load(file, **_kwargs):
                if Path(file.name).name == "last.ckpt":
                    raise RuntimeError("corrupt")
                return {"state_dict": {"model.weight": torch.zeros((2, 3))}}

            with patch(
                "workbench.backend.historical_inspection._checkpoint_shapes.torch.load",
                side_effect=load,
            ) as torch_load:
                shapes = load_checkpoint_graph_shapes(ranked)

        self.assertIsNotNone(shapes)
        self.assertEqual(
            [Path(call.args[0].name).name for call in torch_load.call_args_list],
            ["last.ckpt", "epoch=1-step=100.ckpt"],
        )


if __name__ == "__main__":
    unittest.main()

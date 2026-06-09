from __future__ import annotations

import math
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

try:
    from viewer.backend import tensorboard_reader
except ModuleNotFoundError as exc:
    if exc.name != "tensorboard":
        raise
    tensorboard_reader = None


@dataclass(frozen=True)
class FakeScalarEvent:
    step: int
    wall_time: float
    value: float


class FakeScalarAccumulator:
    def __init__(self, events_by_tag: dict[str, list[FakeScalarEvent]]) -> None:
        self.events_by_tag = events_by_tag

    def Scalars(self, tag: str) -> list[FakeScalarEvent]:
        return self.events_by_tag[tag]


@unittest.skipIf(tensorboard_reader is None, "tensorboard is not installed")
class TensorBoardReaderTests(unittest.TestCase):
    def test_finite_float_coerces_values_and_replaces_non_finite_values(self) -> None:
        assert tensorboard_reader is not None

        self.assertEqual(tensorboard_reader.finite_float(1), 1.0)
        self.assertEqual(tensorboard_reader.finite_float("2.5"), 2.5)
        self.assertEqual(tensorboard_reader.finite_float(math.nan), 0.0)
        self.assertEqual(tensorboard_reader.finite_float(math.inf), 0.0)
        self.assertEqual(tensorboard_reader.finite_float(-math.inf), 0.0)

    def test_scalar_points_projects_events_and_applies_limit(self) -> None:
        assert tensorboard_reader is not None
        accumulator = FakeScalarAccumulator(
            {
                "train/loss": [
                    FakeScalarEvent(step=1, wall_time=10.0, value=0.5),
                    FakeScalarEvent(step=2, wall_time=math.inf, value=math.nan),
                    FakeScalarEvent(step=3, wall_time=30.0, value=0.25),
                ],
            }
        )

        self.assertEqual(
            tensorboard_reader.scalar_points(accumulator, "train/loss", limit=2),
            [
                {"step": 2, "wallTime": 0.0, "value": 0.0},
                {"step": 3, "wallTime": 30.0, "value": 0.25},
            ],
        )

    def test_event_dirs_returns_sorted_deduplicated_event_parent_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "a" / "nested"
            second = root / "z"
            first.mkdir(parents=True)
            second.mkdir(parents=True)
            (first / "events.out.tfevents.first").write_text("", encoding="utf-8")
            (first / "events.out.tfevents.second").write_text("", encoding="utf-8")
            (second / "events.out.tfevents.third").write_text("", encoding="utf-8")
            (root / "events.out.other").write_text("", encoding="utf-8")

            assert tensorboard_reader is not None
            self.assertEqual(tensorboard_reader.event_dirs(root), [first, second])

    def test_event_dirs_returns_empty_for_missing_or_empty_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            assert tensorboard_reader is not None
            self.assertEqual(tensorboard_reader.event_dirs(root), [])
            self.assertEqual(tensorboard_reader.event_dirs(root / "missing"), [])

    def test_load_event_accumulator_reloads_and_returns_accumulator(self) -> None:
        assert tensorboard_reader is not None
        created: list[object] = []

        class ReloadingAccumulator:
            def __init__(self, path: str, size_guidance: dict[int, int]) -> None:
                self.path = path
                self.size_guidance = size_guidance
                self.reloaded = False
                created.append(self)

            def Reload(self) -> None:
                self.reloaded = True

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            with patch(
                "viewer.backend.tensorboard_reader.event_accumulator.EventAccumulator",
                ReloadingAccumulator,
            ):
                loaded = tensorboard_reader.load_event_accumulator(run_dir)

        self.assertIs(loaded, created[0])
        self.assertEqual(created[0].path, str(run_dir))
        self.assertTrue(created[0].reloaded)
        self.assertTrue(all(value == 0 for value in created[0].size_guidance.values()))

    def test_load_event_accumulator_returns_none_when_load_or_reload_fails(
        self,
    ) -> None:
        assert tensorboard_reader is not None

        class FailingAccumulator:
            def __init__(self, path: str, size_guidance: dict[int, int]) -> None:
                raise RuntimeError("load failed")

        class ReloadFailingAccumulator:
            def __init__(self, path: str, size_guidance: dict[int, int]) -> None:
                pass

            def Reload(self) -> None:
                raise RuntimeError("reload failed")

        with patch(
            "viewer.backend.tensorboard_reader.event_accumulator.EventAccumulator",
            FailingAccumulator,
        ):
            self.assertIsNone(
                tensorboard_reader.load_event_accumulator(Path("load-failure"))
            )

        with patch(
            "viewer.backend.tensorboard_reader.event_accumulator.EventAccumulator",
            ReloadFailingAccumulator,
        ):
            self.assertIsNone(
                tensorboard_reader.load_event_accumulator(Path("reload-failure"))
            )


if __name__ == "__main__":
    unittest.main()

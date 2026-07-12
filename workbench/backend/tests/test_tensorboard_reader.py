from __future__ import annotations

import math
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

try:
    import workbench.backend.tensorboard.events as tensorboard_reader
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


class FakeTagsAccumulator:
    def Tags(self) -> dict[str, list[str]]:
        return {
            "scalars": [],
            "histograms": [],
            "images": [],
            "tensors": [],
        }


@unittest.skipIf(tensorboard_reader is None, "tensorboard is not installed")
class TensorBoardReaderTests(unittest.TestCase):
    def test_monitor_read_reuses_one_shared_event_file_observation(self) -> None:
        from workbench.backend.tensorboard.readers import TensorBoardMonitorReader

        assert tensorboard_reader is not None
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            root.joinpath("events.out.tfevents.test").write_bytes(b"events")
            with (
                patch.object(
                    tensorboard_reader,
                    "event_file_index",
                    wraps=tensorboard_reader.event_file_index,
                ) as observe,
                patch.object(
                    tensorboard_reader,
                    "load_event_accumulator",
                    return_value=FakeTagsAccumulator(),
                ) as load,
            ):
                TensorBoardMonitorReader().read(
                    job_id="job-1",
                    node_path="main",
                    dataset="Mnist",
                    log_dir=str(root),
                )

        self.assertEqual(observe.call_count, 1)
        load.assert_called_once_with(root)

    def test_parameter_read_reuses_one_shared_event_file_observation(self) -> None:
        from workbench.backend.tensorboard.readers import (
            TensorBoardParameterStatusReader,
        )

        assert tensorboard_reader is not None
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            root.joinpath("events.out.tfevents.test").write_bytes(b"events")
            reader = TensorBoardParameterStatusReader()
            with (
                patch.object(
                    tensorboard_reader,
                    "event_file_index",
                    wraps=tensorboard_reader.event_file_index,
                ) as observe,
                patch.object(
                    tensorboard_reader,
                    "load_event_accumulator",
                    return_value=FakeTagsAccumulator(),
                ) as load,
            ):
                reader.read(
                    source_id="run-1",
                    preset="baseline",
                    dataset="Mnist",
                    log_dir=str(root),
                )

        self.assertEqual(observe.call_count, 1)
        load.assert_called_once_with(root, size_guidance=reader._size_guidance)

    def test_tag_read_reuses_one_shared_event_file_observation(self) -> None:
        from workbench.backend.run_history.query import LogRunQueryService
        from workbench.backend.run_history.scanner import LogRunScanner

        assert tensorboard_reader is not None
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            root.joinpath("events.out.tfevents.test").write_bytes(b"events")
            query = LogRunQueryService(scanner=LogRunScanner())
            with (
                patch.object(
                    tensorboard_reader,
                    "event_file_index",
                    wraps=tensorboard_reader.event_file_index,
                ) as observe,
                patch.object(
                    tensorboard_reader,
                    "load_event_accumulator",
                    return_value=FakeTagsAccumulator(),
                ) as load,
            ):
                query.read_tags(root)

        self.assertEqual(observe.call_count, 1)
        load.assert_called_once_with(
            root,
            size_guidance=tensorboard_reader.TENSORBOARD_TAG_SIZE_GUIDANCE,
        )

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

    def test_event_file_index_observes_dirs_fingerprint_and_size_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "a" / "nested"
            second = root / "z"
            first.mkdir(parents=True)
            second.mkdir(parents=True)
            (first / "events.out.tfevents.first").write_text("a", encoding="utf-8")
            (first / "events.out.tfevents.second").write_text("bb", encoding="utf-8")
            (second / "events.out.tfevents.third").write_text("ccc", encoding="utf-8")
            (root / "events.out.other").write_text("", encoding="utf-8")

            assert tensorboard_reader is not None
            index = tensorboard_reader.event_file_index(root)

        self.assertEqual(index.root, root)
        self.assertEqual(index.dirs, (first, second))
        self.assertEqual(
            index.files,
            (
                first / "events.out.tfevents.first",
                first / "events.out.tfevents.second",
                second / "events.out.tfevents.third",
            ),
        )
        self.assertEqual(index.total_size, 6)
        self.assertEqual(
            [(path, size) for path, size, _modified_at in index.fingerprint],
            [
                ((first / "events.out.tfevents.first").as_posix(), 1),
                ((first / "events.out.tfevents.second").as_posix(), 2),
                ((second / "events.out.tfevents.third").as_posix(), 3),
            ],
        )

    def test_event_file_index_returns_empty_for_missing_or_empty_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            assert tensorboard_reader is not None
            empty = tensorboard_reader.event_file_index(root)
            missing = tensorboard_reader.event_file_index(root / "missing")

        self.assertEqual(empty.dirs, ())
        self.assertEqual(empty.fingerprint, ())
        self.assertEqual(empty.total_size, 0)
        self.assertEqual(missing.dirs, ())
        self.assertEqual(missing.fingerprint, ())
        self.assertEqual(missing.total_size, 0)

    def test_event_file_index_excludes_an_entire_mixed_escape_directory(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mixed = root / "mixed"
            trusted = root / "trusted"
            outside = root.parent / f"{root.name}-outside-event"
            mixed.mkdir()
            trusted.mkdir()
            mixed.joinpath("events.out.tfevents.safe").write_bytes(b"safe")
            outside.write_bytes(b"outside")
            mixed.joinpath("events.out.tfevents.escape").symlink_to(outside)
            trusted.joinpath("events.out.tfevents.safe").write_bytes(b"trusted")

            try:
                assert tensorboard_reader is not None
                index = tensorboard_reader.event_file_index(root)
            finally:
                outside.unlink(missing_ok=True)

        self.assertEqual(index.dirs, (trusted,))
        self.assertEqual(index.total_size, len(b"trusted"))
        self.assertEqual(len(index.fingerprint), 1)

    def test_event_cache_lru_and_generation_reject_stale_publication(self) -> None:
        assert tensorboard_reader is not None
        cache = tensorboard_reader.TensorBoardEventCache({"payload": 2})
        generation = cache.token()
        first = ("/runs/first", (), "value")
        nested = ("/runs/first/nested", (), "accumulator")
        second = ("/runs/second", (), "value")
        third = ("/runs/third", (), "value")
        cache.publish("payload", first, 1, generation=generation)
        cache.publish("payload", nested, "nested", generation=generation)
        self.assertEqual(cache.get("payload", first), 1)
        cache.publish("payload", second, 2, generation=generation)

        self.assertIsNone(cache.get("payload", nested))
        cache.clear_roots({"/runs/first"})
        cache.publish("payload", first, 4, generation=generation)
        current_generation = cache.token()
        cache.publish("payload", third, 3, generation=current_generation)

        self.assertIsNone(cache.get("payload", first))
        self.assertEqual(cache.get("payload", third), 3)

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
                "workbench.backend.tensorboard.events.event_accumulator.EventAccumulator",
                ReloadingAccumulator,
            ):
                loaded = tensorboard_reader.load_event_accumulator(run_dir)

        self.assertIs(loaded, created[0])
        self.assertEqual(created[0].path, str(run_dir))
        self.assertTrue(created[0].reloaded)
        self.assertEqual(
            created[0].size_guidance,
            tensorboard_reader.DEFAULT_TENSORBOARD_SIZE_GUIDANCE,
        )

    def test_load_event_accumulator_accepts_custom_size_guidance(self) -> None:
        assert tensorboard_reader is not None
        created: list[object] = []

        class ReloadingAccumulator:
            def __init__(self, path: str, size_guidance: dict[int, int]) -> None:
                self.size_guidance = size_guidance
                created.append(self)

            def Reload(self) -> None:
                pass

        custom_guidance = {1: 2}
        with patch(
            "workbench.backend.tensorboard.events.event_accumulator.EventAccumulator",
            ReloadingAccumulator,
        ):
            tensorboard_reader.load_event_accumulator(
                Path("run"),
                size_guidance=custom_guidance,
            )

        self.assertEqual(created[0].size_guidance, custom_guidance)

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
            "workbench.backend.tensorboard.events.event_accumulator.EventAccumulator",
            FailingAccumulator,
        ):
            self.assertIsNone(
                tensorboard_reader.load_event_accumulator(Path("load-failure"))
            )

        with patch(
            "workbench.backend.tensorboard.events.event_accumulator.EventAccumulator",
            ReloadFailingAccumulator,
        ):
            self.assertIsNone(
                tensorboard_reader.load_event_accumulator(Path("reload-failure"))
            )


if __name__ == "__main__":
    unittest.main()

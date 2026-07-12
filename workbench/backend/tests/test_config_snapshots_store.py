from __future__ import annotations

import tempfile
import unittest
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

from workbench.backend.config_snapshots import (
    ConfigSnapshotConflictError,
    ConfigSnapshotConflictReason,
    ConfigSnapshotRecord,
    ConfigSnapshotStore,
    FileSystemConfigSnapshotStore,
    InMemoryConfigSnapshotStore,
)


def make_record(
    snapshot_id: str = "snap-1",
    model: str = "experts/linear",
    preset: str = "base",
    name: str | None = None,
    overrides: dict[str, str] | None = None,
    created_at: str | None = None,
) -> ConfigSnapshotRecord:
    return ConfigSnapshotRecord(
        id=snapshot_id,
        model=model,
        preset=preset,
        name=name if name is not None else f"tuned-{snapshot_id}",
        overrides=(
            overrides
            if overrides is not None
            else {"learning_rate": f"value-{snapshot_id}"}
        ),
        **({"created_at": created_at} if created_at is not None else {}),
    )


def create_record(
    store: ConfigSnapshotStore,
    record: ConfigSnapshotRecord,
) -> ConfigSnapshotRecord:
    return store.create(record)


@contextmanager
def config_snapshot_stores() -> Iterator[list[tuple[str, ConfigSnapshotStore]]]:
    with tempfile.TemporaryDirectory() as tmp:
        yield [
            ("memory", InMemoryConfigSnapshotStore()),
            ("filesystem", FileSystemConfigSnapshotStore(Path(tmp))),
        ]


class ConfigSnapshotStoreContractTests(unittest.TestCase):
    def test_records_are_immutable_values_for_both_adapters(self) -> None:
        with config_snapshot_stores() as stores:
            for adapter, store in stores:
                with self.subTest(adapter=adapter):
                    source_overrides = {"learning_rate": "0.01"}
                    created = create_record(
                        store,
                        make_record(overrides=source_overrides),
                    )
                    source_overrides["learning_rate"] = "0.5"

                    loaded = store.get(created.id)
                    self.assertIsNotNone(loaded)
                    assert loaded is not None
                    self.assertEqual(loaded.overrides, {"learning_rate": "0.01"})
                    with self.assertRaises(FrozenInstanceError):
                        loaded.name = "mutated"  # type: ignore[misc]
                    with self.assertRaises(TypeError):
                        loaded.overrides["learning_rate"] = "0.5"  # type: ignore[index]

    def test_conflicts_and_failed_replacements_publish_nothing_for_both_adapters(
        self,
    ) -> None:
        with config_snapshot_stores() as stores:
            for adapter, store in stores:
                with self.subTest(adapter=adapter):
                    original = create_record(store, make_record())

                    with self.assertRaises(ConfigSnapshotConflictError) as name:
                        create_record(
                            store,
                            make_record(
                                snapshot_id="name-conflict",
                                name=original.name.upper(),
                            ),
                        )
                    self.assertEqual(
                        name.exception.reason,
                        ConfigSnapshotConflictReason.NAME,
                    )

                    with self.assertRaises(ConfigSnapshotConflictError) as values:
                        create_record(
                            store,
                            make_record(
                                snapshot_id="values-conflict",
                                overrides=dict(original.overrides),
                            ),
                        )
                    self.assertEqual(
                        values.exception.reason,
                        ConfigSnapshotConflictReason.RUNTIME_DEFAULTS,
                    )

                    replacement = replace(
                        original,
                        name="updated",
                        updated_at="2026-02-01",
                    )
                    self.assertEqual(
                        store.update(original, replacement),
                        replacement,
                    )
                    with self.assertRaises(ConfigSnapshotConflictError) as stale:
                        store.update(
                            original,
                            replace(original, name="stale", updated_at="2026-02-02"),
                        )
                    self.assertEqual(
                        stale.exception.reason,
                        ConfigSnapshotConflictReason.STALE,
                    )

                    with self.assertRaisesRegex(ValueError, "record identity"):
                        store.update(
                            replacement,
                            replace(replacement, model="other"),
                        )

                    loaded = store.get(original.id)
                    self.assertEqual(loaded, replacement)
                    self.assertEqual(
                        [snapshot.id for snapshot in store.list_all()],
                        [original.id],
                    )


class InMemoryConfigSnapshotStoreTests(unittest.TestCase):
    def test_saves_and_lists_scoped_by_model(self) -> None:
        store = InMemoryConfigSnapshotStore()
        create_record(store, make_record(snapshot_id="a", model="m1"))
        create_record(store, make_record(snapshot_id="b", model="m2"))

        self.assertEqual([s.id for s in store.list("m1")], ["a"])
        self.assertEqual([s.id for s in store.list("m2")], ["b"])

    def test_get_and_delete(self) -> None:
        store = InMemoryConfigSnapshotStore()
        create_record(store, make_record(snapshot_id="a"))

        self.assertIsNotNone(store.get("a"))
        self.assertTrue(store.delete("a"))
        self.assertFalse(store.delete("a"))
        self.assertIsNone(store.get("a"))

    def test_list_is_sorted_by_created_at(self) -> None:
        store = InMemoryConfigSnapshotStore()
        create_record(
            store,
            make_record(snapshot_id="b", model="m", created_at="2026-01-02"),
        )
        create_record(
            store,
            make_record(snapshot_id="a", model="m", created_at="2026-01-01"),
        )

        self.assertEqual([s.id for s in store.list("m")], ["a", "b"])

    def test_list_all_is_sorted_across_models(self) -> None:
        store = InMemoryConfigSnapshotStore()
        create_record(
            store,
            make_record(
                snapshot_id="b",
                model="m2",
                preset="z",
                created_at="2026-01-02",
            )
        )
        create_record(
            store,
            make_record(
                snapshot_id="a",
                model="m1",
                preset="z",
                created_at="2026-01-02",
            )
        )
        create_record(
            store,
            make_record(
                snapshot_id="c",
                model="m1",
                preset="a",
                created_at="2026-01-03",
            )
        )

        self.assertEqual([s.id for s in store.list_all()], ["c", "a", "b"])


class FileSystemConfigSnapshotStoreTests(unittest.TestCase):
    def test_persists_across_store_instances(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            first = FileSystemConfigSnapshotStore(Path(tmp))
            create_record(
                first,
                make_record(snapshot_id="a", model="m1", overrides={"lr": "0.02"})
            )

            second = FileSystemConfigSnapshotStore(Path(tmp))
            loaded = second.get("a")

            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(loaded.overrides, {"lr": "0.02"})
            self.assertEqual([s.id for s in second.list("m1")], ["a"])

    def test_writes_one_file_per_snapshot_under_model_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FileSystemConfigSnapshotStore(Path(tmp))
            create_record(store, make_record(snapshot_id="a", model="m1"))

            self.assertTrue((Path(tmp) / "m1" / "a.json").exists())

    def test_delete_removes_the_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FileSystemConfigSnapshotStore(Path(tmp))
            create_record(store, make_record(snapshot_id="a", model="m1"))

            self.assertTrue(store.delete("a"))
            self.assertFalse((Path(tmp) / "m1" / "a.json").exists())
            self.assertFalse(store.delete("a"))

    def test_nested_model_paths_can_be_found_for_delete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FileSystemConfigSnapshotStore(Path(tmp))
            create_record(
                store,
                make_record(snapshot_id="a", model="linears/linear"),
            )

            self.assertTrue((Path(tmp) / "linears" / "linear" / "a.json").exists())
            self.assertEqual(
                [snapshot.id for snapshot in store.list("linears/linear")],
                ["a"],
            )
            self.assertTrue(store.delete("a"))
            self.assertFalse((Path(tmp) / "linears" / "linear" / "a.json").exists())

    def test_list_all_finds_nested_model_paths_and_sorts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FileSystemConfigSnapshotStore(Path(tmp))
            create_record(
                store,
                make_record(
                    snapshot_id="b",
                    model="linears/linear_adaptive",
                    preset="wide",
                    created_at="2026-01-02",
                )
            )
            create_record(
                store,
                make_record(
                    snapshot_id="c",
                    model="linears/linear",
                    preset="fast",
                    created_at="2026-01-03",
                )
            )
            create_record(
                store,
                make_record(
                    snapshot_id="a",
                    model="linears/linear",
                    preset="baseline",
                    created_at="2026-01-04",
                )
            )

            self.assertEqual(
                [
                    (snapshot.model, snapshot.preset, snapshot.id)
                    for snapshot in store.list_all()
                ],
                [
                    ("linears/linear", "baseline", "a"),
                    ("linears/linear", "fast", "c"),
                    ("linears/linear_adaptive", "wide", "b"),
                ],
            )

    def test_list_unknown_model_is_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FileSystemConfigSnapshotStore(Path(tmp))

            self.assertEqual(store.list("nope"), [])

    def test_corrupt_files_are_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FileSystemConfigSnapshotStore(Path(tmp))
            model_dir = Path(tmp) / "m1"
            model_dir.mkdir(parents=True)
            (model_dir / "broken.json").write_text("{not json", encoding="utf-8")

            self.assertEqual(store.list("m1"), [])
            self.assertEqual(store.list_all(), [])

    def test_non_utf8_corrupt_files_are_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FileSystemConfigSnapshotStore(Path(tmp))
            model_dir = Path(tmp) / "m1"
            model_dir.mkdir(parents=True)
            (model_dir / "broken.json").write_bytes(bytes((255, 254)))

            self.assertEqual(store.list("m1"), [])
            self.assertEqual(store.list_all(), [])

    def test_list_all_ignores_noncanonical_snapshot_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = FileSystemConfigSnapshotStore(root)
            model_dir = root / "m1"
            wrong_model_dir = root / "wrong"
            model_dir.mkdir(parents=True)
            wrong_model_dir.mkdir(parents=True)
            (model_dir / "wrong-name.json").write_text(
                '{"id": "a", "model": "m1", "preset": "base", "name": "x", '
                '"overrides": {}, "created_at": "2026", "updated_at": "2026"}',
                encoding="utf-8",
            )
            (wrong_model_dir / "a.json").write_text(
                '{"id": "a", "model": "m1", "preset": "base", "name": "x", '
                '"overrides": {}, "created_at": "2026", "updated_at": "2026"}',
                encoding="utf-8",
            )

            self.assertEqual(store.list_all(), [])

    def test_rejects_unsafe_model_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = FileSystemConfigSnapshotStore(root / "snapshots")
            outside = root / "outside"

            cases = (
                "../outside",
                str(outside),
                "linears\\linear",
                "linears//linear",
                "",
            )
            for model in cases:
                with self.subTest(model=model):
                    with self.assertRaises(ValueError):
                        create_record(store, make_record(snapshot_id="a", model=model))
                    with self.assertRaises(ValueError):
                        store.list(model)
            self.assertFalse(outside.exists())

    def test_rejects_unsafe_snapshot_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = FileSystemConfigSnapshotStore(root / "snapshots")
            outside = root / "outside"
            outside.mkdir()

            cases = (
                "../snap",
                str(outside / "snap"),
                "snap/id",
                "snap\\id",
                "snap.json",
                "",
            )
            for snapshot_id in cases:
                with self.subTest(snapshot_id=snapshot_id):
                    with self.assertRaises(ValueError):
                        create_record(
                            store,
                            make_record(snapshot_id=snapshot_id, model="m1"),
                        )
                    with self.assertRaises(ValueError):
                        store.get(snapshot_id)
                    with self.assertRaises(ValueError):
                        store.delete(snapshot_id)
            self.assertEqual(list(outside.iterdir()), [])

    def test_rejects_symlink_escape_for_model_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshots_root = root / "snapshots"
            outside = root / "outside"
            snapshots_root.mkdir()
            outside.mkdir()
            link = snapshots_root / "linked"
            try:
                link.symlink_to(outside, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"symlinks unavailable: {exc}")

            store = FileSystemConfigSnapshotStore(snapshots_root)

            with self.assertRaises(ValueError):
                create_record(
                    store,
                    make_record(snapshot_id="a", model="linked/linear"),
                )
            with self.assertRaises(ValueError):
                store.list("linked/linear")
            self.assertEqual(list(outside.iterdir()), [])

    def test_ignores_symlink_snapshot_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshots_root = root / "snapshots"
            outside = root / "outside"
            model_dir = snapshots_root / "m1"
            model_dir.mkdir(parents=True)
            outside.mkdir()
            outside_snapshot = outside / "a.json"
            outside_snapshot.write_text(
                '{"id": "a", "model": "m1", "preset": "base", "name": "x", '
                '"overrides": {}, "created_at": "2026", "updated_at": "2026"}',
                encoding="utf-8",
            )
            link = model_dir / "a.json"
            try:
                link.symlink_to(outside_snapshot)
            except OSError as exc:
                self.skipTest(f"symlinks unavailable: {exc}")

            store = FileSystemConfigSnapshotStore(snapshots_root)

            self.assertIsNone(store.get("a"))
            self.assertEqual(store.list("m1"), [])
            self.assertEqual(store.list_all(), [])
            self.assertFalse(store.delete("a"))
            self.assertTrue(link.is_symlink())
            self.assertTrue(outside_snapshot.exists())


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from viewer.backend.config_snapshots import (
    ConfigSnapshotRecord,
    FileSystemConfigSnapshotStore,
    InMemoryConfigSnapshotStore,
)


def make_record(
    snapshot_id: str = "snap-1",
    model: str = "experts/experts_linear",
    preset: str = "base",
    name: str = "tuned",
    overrides: dict[str, str] | None = None,
    created_at: str | None = None,
) -> ConfigSnapshotRecord:
    record = ConfigSnapshotRecord(
        id=snapshot_id,
        model=model,
        preset=preset,
        name=name,
        overrides=overrides if overrides is not None else {"learning_rate": "0.01"},
    )
    if created_at is not None:
        record.created_at = created_at
    return record


class InMemoryConfigSnapshotStoreTests(unittest.TestCase):
    def test_saves_and_lists_scoped_by_model(self) -> None:
        store = InMemoryConfigSnapshotStore()
        store.save(make_record(snapshot_id="a", model="m1"))
        store.save(make_record(snapshot_id="b", model="m2"))

        self.assertEqual([s.id for s in store.list("m1")], ["a"])
        self.assertEqual([s.id for s in store.list("m2")], ["b"])

    def test_get_and_delete(self) -> None:
        store = InMemoryConfigSnapshotStore()
        store.save(make_record(snapshot_id="a"))

        self.assertIsNotNone(store.get("a"))
        self.assertTrue(store.delete("a"))
        self.assertFalse(store.delete("a"))
        self.assertIsNone(store.get("a"))

    def test_list_is_sorted_by_created_at(self) -> None:
        store = InMemoryConfigSnapshotStore()
        store.save(make_record(snapshot_id="b", model="m", created_at="2026-01-02"))
        store.save(make_record(snapshot_id="a", model="m", created_at="2026-01-01"))

        self.assertEqual([s.id for s in store.list("m")], ["a", "b"])


class FileSystemConfigSnapshotStoreTests(unittest.TestCase):
    def test_persists_across_store_instances(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            first = FileSystemConfigSnapshotStore(Path(tmp))
            first.save(
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
            store.save(make_record(snapshot_id="a", model="m1"))

            self.assertTrue((Path(tmp) / "m1" / "a.json").exists())

    def test_delete_removes_the_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FileSystemConfigSnapshotStore(Path(tmp))
            store.save(make_record(snapshot_id="a", model="m1"))

            self.assertTrue(store.delete("a"))
            self.assertFalse((Path(tmp) / "m1" / "a.json").exists())
            self.assertFalse(store.delete("a"))

    def test_nested_model_paths_can_be_found_for_delete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FileSystemConfigSnapshotStore(Path(tmp))
            store.save(make_record(snapshot_id="a", model="linears/linear"))

            self.assertTrue((Path(tmp) / "linears" / "linear" / "a.json").exists())
            self.assertEqual(
                [snapshot.id for snapshot in store.list("linears/linear")],
                ["a"],
            )
            self.assertTrue(store.delete("a"))
            self.assertFalse((Path(tmp) / "linears" / "linear" / "a.json").exists())

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
                        store.save(make_record(snapshot_id="a", model=model))
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
                        store.save(make_record(snapshot_id=snapshot_id, model="m1"))
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
                store.save(make_record(snapshot_id="a", model="linked/linear"))
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
            self.assertFalse(store.delete("a"))
            self.assertTrue(link.is_symlink())
            self.assertTrue(outside_snapshot.exists())


if __name__ == "__main__":
    unittest.main()

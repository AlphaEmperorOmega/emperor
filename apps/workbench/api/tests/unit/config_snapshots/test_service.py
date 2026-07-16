from __future__ import annotations

import re
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from pathlib import Path
from threading import Barrier
from typing import Any

from emperor_workbench.config_snapshots import (
    ConfigSnapshotFailure,
    ConfigSnapshotRecord,
    ConfigSnapshotService,
)
from emperor_workbench.config_snapshots._filesystem_store import (
    FileSystemConfigSnapshotStore,
)
from emperor_workbench.config_snapshots._memory_store import (
    InMemoryConfigSnapshotStore,
)
from emperor_workbench.config_snapshots._store import ConfigSnapshotStore
from emperor_workbench.model_packages import ModelPackageCatalog
from tests.support import API_ROOT
from tests.support.model_packages import project_adapter_client


def _service(store: ConfigSnapshotStore) -> ConfigSnapshotService:
    return ConfigSnapshotService(
        store,
        model_packages=ModelPackageCatalog(project_adapter_client()),
    )


class _SynchronizedListStore:
    """Force current shallow list/check/save mutations into the same race."""

    def __init__(self, store: ConfigSnapshotStore) -> None:
        self._store = store
        self._barrier: Barrier | None = None

    def arm(self) -> None:
        self._barrier = Barrier(2)

    def list(self, model: str):  # type: ignore[no-untyped-def]
        snapshots = self._store.list(model)
        barrier = self._barrier
        if barrier is not None:
            barrier.wait(timeout=10)
        return snapshots

    def __getattr__(self, name: str) -> Any:
        return getattr(self._store, name)


def _capture_mutation(mutation):  # type: ignore[no-untyped-def]
    try:
        return mutation()
    except ConfigSnapshotFailure as exc:
        return exc


class ConfigSnapshotServiceAdaptiveValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = InMemoryConfigSnapshotStore()
        self.service = _service(self.store)

    def test_public_interface_returns_frozen_semantic_records(self) -> None:
        import emperor_workbench.config_snapshots as config_snapshots

        self.assertEqual(
            config_snapshots.__all__,
            [
                "ConfigSnapshotDeletion",
                "ConfigSnapshotFailure",
                "ConfigSnapshotRecord",
                "ConfigSnapshotService",
            ],
        )
        snapshot = self.service.create_snapshot(
            model="linears/linear",
            preset="baseline",
            name="semantic record",
            overrides={"HIDDEN_DIM": "128"},
        )

        self.assertIsInstance(snapshot, ConfigSnapshotRecord)
        self.assertEqual(
            self.service.list_snapshots("linears/linear"),
            (snapshot,),
        )
        with self.assertRaises(FrozenInstanceError):
            snapshot.name = "changed"  # type: ignore[misc]
        with self.assertRaises(TypeError):
            snapshot.overrides["HIDDEN_DIM"] = "64"  # type: ignore[index]

    def test_create_rejects_adaptive_flag_without_matching_option(self) -> None:
        with self.assertRaisesRegex(
            ConfigSnapshotFailure,
            re.escape(
                "Invalid config snapshot overrides: models.linears.linear_adaptive: "
                "runtime key 'weight_option' must be set when "
                "'weight_option_flag' is True"
            ),
        ):
            self.service.create_snapshot(
                model="linears/linear_adaptive",
                preset="baseline",
                name="flag only",
                overrides={"WEIGHT_OPTION_FLAG": "true"},
            )

    def test_update_rejects_adaptive_flag_without_matching_option(self) -> None:
        snapshot = self.service.create_snapshot(
            model="linears/linear_adaptive",
            preset="baseline",
            name="wider stack",
            overrides={"HIDDEN_DIM": "384"},
        )

        with self.assertRaisesRegex(
            ConfigSnapshotFailure,
            re.escape(
                "Invalid config snapshot overrides: models.linears.linear_adaptive: "
                "runtime key 'weight_option' must be set when "
                "'weight_option_flag' is True"
            ),
        ):
            self.service.update_snapshot(
                snapshot.id,
                overrides={"WEIGHT_OPTION_FLAG": "true"},
            )

        stored = self.store.get(snapshot.id)
        self.assertIsNotNone(stored)
        assert stored is not None
        self.assertEqual(stored.overrides, {"HIDDEN_DIM": "384"})

    def test_create_accepts_adaptive_flag_with_matching_option(self) -> None:
        snapshot = self.service.create_snapshot(
            model="linears/linear_adaptive",
            preset="baseline",
            name="single weight",
            overrides={
                "WEIGHT_OPTION_FLAG": "true",
                "WEIGHT_OPTION": "SingleModelDynamicWeightConfig",
            },
        )

        self.assertEqual(
            dict(snapshot.overrides),
            {
                "WEIGHT_OPTION_FLAG": "true",
                "WEIGHT_OPTION": "SingleModelDynamicWeightConfig",
            },
        )
        self.assertEqual(
            self.service.semantic_revision(snapshot),
            "1d87dbcd56ffce6b611ef0f8ae22967aca6c86ece0e1746e5bdf8ddb2b948b13",
        )

    def test_real_schema_create_and_update_freeze_canonical_serialization(
        self,
    ) -> None:
        snapshot = self.service.create_snapshot(
            model="linears/linear",
            preset="baseline",
            name="canonical values",
            overrides={
                "hidden-dim": "128.0",
                "stack-bias-flag": "FALSE",
                "stack_dropout_probability": "0.250",
            },
        )

        self.assertEqual(
            dict(snapshot.overrides),
            {
                "HIDDEN_DIM": "128",
                "STACK_DROPOUT_PROBABILITY": "0.25",
                "STACK_BIAS_FLAG": "false",
            },
        )
        self.assertEqual(
            self.service.semantic_revision(snapshot),
            "1242bb1b4cc19b429c0393ff4fb0301d1e9052e3478a3251a4d0a3b7dde09cf7",
        )

        updated = self.service.update_snapshot(
            snapshot.id,
            overrides={
                "hidden_dim": "64.0",
                "memory_stack_hidden_dim": "48",
            },
        )
        self.assertEqual(
            dict(updated.overrides),
            {"HIDDEN_DIM": "64", "MEMORY_STACK_HIDDEN_DIM": "48"},
        )

    def test_real_schema_rejects_preset_locked_create_without_persisting(self) -> None:
        with self.assertRaisesRegex(
            ConfigSnapshotFailure,
            "preset-locked fields",
        ):
            self.service.create_snapshot(
                model="linears/linear",
                preset="gating",
                name="disable locked gate",
                overrides={"gate_flag": "false"},
            )

        self.assertEqual(self.store.list("linears/linear"), [])

    def test_failed_real_construction_does_not_persist_create(self) -> None:
        with self.assertRaisesRegex(
            ConfigSnapshotFailure,
            "runtime key 'weight_option' must be set",
        ):
            self.service.create_snapshot(
                model="linears/linear_adaptive",
                preset="baseline",
                name="invalid adaptive config",
                overrides={"weight_option_flag": "true"},
            )

        self.assertEqual(self.store.list("linears/linear_adaptive"), [])

    def test_mutations_do_not_depend_on_http_schema_serialization(self) -> None:
        capability_root = API_ROOT / "src" / "emperor_workbench" / "config_snapshots"
        transport_imports = [
            source.name
            for source in capability_root.glob("*.py")
            if "emperor_workbench.api" in source.read_text(encoding="utf-8")
        ]
        self.assertEqual(transport_imports, [])

        snapshot = self.service.create_snapshot(
            model="linears/linear",
            preset="baseline",
            name="semantic schema",
            overrides={"learning_rate": "0.01"},
        )
        updated = self.service.update_snapshot(
            snapshot.id,
            overrides={"learning_rate": "0.02"},
        )

        self.assertEqual(dict(updated.overrides), {"LEARNING_RATE": "0.02"})

    def test_explicit_resource_id_reconciles_an_idempotent_create(self) -> None:
        first = self.service.create_snapshot(
            model="linears/linear",
            preset="baseline",
            name="first",
            overrides={"learning_rate": "0.01"},
            snapshot_id="stable-resource-id",
        )
        replay = self.service.create_snapshot(
            model="linears/linear",
            preset="baseline",
            name="ignored replay payload",
            overrides={"learning_rate": "0.02"},
            snapshot_id="stable-resource-id",
        )

        self.assertIs(replay, first)
        self.assertEqual(self.service.list_all_snapshots(), (first,))


class ConfigSnapshotServiceAtomicMutationTests(unittest.TestCase):
    def _stores(
        self,
        filesystem_root: Path,
    ) -> list[tuple[str, ConfigSnapshotStore]]:
        return [
            ("memory", InMemoryConfigSnapshotStore()),
            ("filesystem", FileSystemConfigSnapshotStore(filesystem_root)),
        ]

    def test_failed_combined_update_preserves_the_complete_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for adapter, store in self._stores(Path(tmp)):
                with self.subTest(adapter=adapter):
                    service = _service(store)
                    snapshot = service.create_snapshot(
                        model="linears/linear",
                        preset="baseline",
                        name="original",
                        overrides={"learning_rate": "0.01"},
                    )

                    with self.assertRaisesRegex(
                        ConfigSnapshotFailure,
                        "non-default",
                    ):
                        service.update_snapshot(
                            snapshot.id,
                            name="must not leak",
                            overrides={"learning_rate": "0.001"},
                        )

                    stored = store.get(snapshot.id)
                    self.assertIsNotNone(stored)
                    assert stored is not None
                    self.assertEqual(stored.name, "original")
                    self.assertEqual(stored.overrides, {"LEARNING_RATE": "0.01"})

    def test_concurrent_creates_preserve_name_and_runtime_defaults_uniqueness(
        self,
    ) -> None:
        cases = (
            (
                "runtime-defaults",
                (
                    ("first", {"learning_rate": "0.01"}),
                    ("second", {"learning_rate": "0.01"}),
                ),
            ),
            (
                "name",
                (
                    ("Same Name", {"learning_rate": "0.01"}),
                    (" same name ", {"learning_rate": "0.02"}),
                ),
            ),
        )
        for case, candidates in cases:
            with tempfile.TemporaryDirectory() as tmp:
                for adapter, raw_store in self._stores(Path(tmp)):
                    with self.subTest(case=case, adapter=adapter):
                        store = _SynchronizedListStore(raw_store)
                        store.arm()
                        service = _service(store)  # type: ignore[arg-type]

                        def create(
                            candidate: tuple[str, dict[str, str]],
                            service: ConfigSnapshotService = service,
                        ):
                            name, overrides = candidate
                            return _capture_mutation(
                                lambda: service.create_snapshot(
                                    model="linears/linear",
                                    preset="baseline",
                                    name=name,
                                    overrides=overrides,
                                )
                            )

                        with ThreadPoolExecutor(max_workers=2) as executor:
                            outcomes = list(executor.map(create, candidates))

                        self.assertEqual(
                            sum(
                                isinstance(outcome, ConfigSnapshotRecord)
                                for outcome in outcomes
                            ),
                            1,
                        )
                        self.assertEqual(
                            sum(
                                isinstance(outcome, ConfigSnapshotFailure)
                                for outcome in outcomes
                            ),
                            1,
                        )
                        self.assertEqual(len(raw_store.list_all()), 1)

    def test_concurrent_updates_preserve_runtime_defaults_uniqueness(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for adapter, raw_store in self._stores(Path(tmp)):
                with self.subTest(adapter=adapter):
                    store = _SynchronizedListStore(raw_store)
                    service = _service(store)  # type: ignore[arg-type]
                    first = service.create_snapshot(
                        model="linears/linear",
                        preset="baseline",
                        name="first",
                        overrides={"learning_rate": "0.01"},
                    )
                    second = service.create_snapshot(
                        model="linears/linear",
                        preset="baseline",
                        name="second",
                        overrides={"learning_rate": "0.02"},
                    )
                    store.arm()

                    def update(
                        snapshot_id: str,
                        service: ConfigSnapshotService = service,
                    ):
                        return _capture_mutation(
                            lambda: service.update_snapshot(
                                snapshot_id,
                                overrides={"hidden_dim": "256"},
                            )
                        )

                    with ThreadPoolExecutor(max_workers=2) as executor:
                        outcomes = list(executor.map(update, (first.id, second.id)))

                    self.assertEqual(
                        sum(
                            isinstance(outcome, ConfigSnapshotRecord)
                            for outcome in outcomes
                        ),
                        1,
                    )
                    self.assertEqual(
                        sum(
                            isinstance(outcome, ConfigSnapshotFailure)
                            for outcome in outcomes
                        ),
                        1,
                    )


if __name__ == "__main__":
    unittest.main()

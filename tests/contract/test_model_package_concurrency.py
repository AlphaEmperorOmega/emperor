from __future__ import annotations

import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

from model_runtime.packages import ModelIdentity, ModelPackage


class _TrackedAdapter:
    def __init__(self, load) -> None:
        self.load = load

    def load_metadata(self):
        return self.load("metadata")

    def load_runtime_options_type(self):
        return self.load("runtime_options_type")

    def bind_runtime_defaults(self, values):
        return SimpleNamespace()

    def load_preset_type(self):
        return self.load("preset_type")

    def load_presets(self):
        return self.load("presets")

    def build_configuration(self, presets, preset, dataset, **kwargs):
        return None

    def build_model(self, configuration):
        return None

    def build_experiment(self, preset, **kwargs):
        return None


class _ValueAdapter(_TrackedAdapter):
    def __init__(self) -> None:
        super().__init__(str)


class ModelPackageConcurrencyContractTests(unittest.TestCase):
    def test_cold_package_exposes_only_immutable_canonical_identity(self) -> None:
        package = ModelPackage(ModelIdentity("test", "linear"), _ValueAdapter())

        self.assertEqual(package.catalog_key, "test/linear")
        self.assertEqual(
            package.to_identity_payload(),
            {"modelType": "test", "model": "linear"},
        )
        self.assertNotIn("_adapter", repr(package))
        with self.assertRaises(FrozenInstanceError):
            package.identity = ModelIdentity("test", "other")

    def test_lazy_initialization_is_serial_and_shared_across_concurrent_callers(
        self,
    ) -> None:
        start = threading.Barrier(4)
        counter_lock = threading.Lock()
        active_loads = 0
        maximum_active_loads = 0
        loads: dict[str, int] = {}

        def tracked_load(operation: str) -> object:
            nonlocal active_loads
            nonlocal maximum_active_loads
            with counter_lock:
                active_loads += 1
                maximum_active_loads = max(maximum_active_loads, active_loads)
                loads[operation] = loads.get(operation, 0) + 1
            try:
                time.sleep(0.05)
                return SimpleNamespace(operation=operation)
            finally:
                with counter_lock:
                    active_loads -= 1

        package = ModelPackage(
            ModelIdentity("test", "linear"),
            _TrackedAdapter(tracked_load),
        )

        def read_metadata() -> object:
            start.wait()
            return package.metadata

        def read_preset_type() -> object:
            start.wait()
            return package.preset_type

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(read_metadata),
                executor.submit(read_metadata),
                executor.submit(read_preset_type),
            ]
            start.wait()
            metadata_a, metadata_b, preset_type = [
                future.result(timeout=2) for future in futures
            ]

        self.assertEqual(maximum_active_loads, 1)
        self.assertEqual(loads, {"metadata": 1, "preset_type": 1})
        self.assertIs(metadata_a, metadata_b)
        self.assertIs(preset_type, package.preset_type)

    def test_cold_initialization_is_serial_across_sibling_packages(self) -> None:
        start = threading.Barrier(3)
        counter_lock = threading.Lock()
        active_loads = 0
        maximum_active_loads = 0

        def load_metadata(_operation: str) -> object:
            nonlocal active_loads
            nonlocal maximum_active_loads
            with counter_lock:
                active_loads += 1
                maximum_active_loads = max(maximum_active_loads, active_loads)
            try:
                time.sleep(0.05)
                return SimpleNamespace()
            finally:
                with counter_lock:
                    active_loads -= 1

        package_a = ModelPackage(
            ModelIdentity("test", "a"),
            _TrackedAdapter(load_metadata),
        )
        package_b = ModelPackage(
            ModelIdentity("test", "b"),
            _TrackedAdapter(load_metadata),
        )

        def read_metadata(package: ModelPackage) -> object:
            start.wait()
            return package.metadata

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(read_metadata, package_a),
                executor.submit(read_metadata, package_b),
            ]
            start.wait()
            for future in futures:
                future.result(timeout=2)

        self.assertEqual(maximum_active_loads, 1)


if __name__ == "__main__":
    unittest.main()

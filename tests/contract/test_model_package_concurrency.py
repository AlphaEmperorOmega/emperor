from __future__ import annotations

import copy
import pickle
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import patch

from model_runtime.packages import ModelPackage


class ModelPackageConcurrencyContractTests(unittest.TestCase):
    def test_cold_package_retains_dataclass_value_behavior(self) -> None:
        package = ModelPackage("test", "linear", "models.test.linear")

        self.assertEqual(
            asdict(package),
            {
                "model_type": "test",
                "model": "linear",
                "module_path": "models.test.linear",
            },
        )
        self.assertEqual(copy.deepcopy(package), package)
        self.assertEqual(pickle.loads(pickle.dumps(package)), package)

    def test_lazy_initialization_is_serial_and_shared_across_concurrent_callers(
        self,
    ) -> None:
        package = ModelPackage("test", "linear", "models.test.linear")
        start = threading.Barrier(4)
        counter_lock = threading.Lock()
        active_loads = 0
        maximum_active_loads = 0
        metadata_loads = 0
        preset_module_loads = 0

        def tracked_load(value: object, *, metadata: bool) -> object:
            nonlocal active_loads
            nonlocal maximum_active_loads
            nonlocal metadata_loads
            nonlocal preset_module_loads
            with counter_lock:
                active_loads += 1
                maximum_active_loads = max(maximum_active_loads, active_loads)
                if metadata:
                    metadata_loads += 1
                else:
                    preset_module_loads += 1
            try:
                time.sleep(0.05)
                return value
            finally:
                with counter_lock:
                    active_loads -= 1

        def load_metadata(*_args: object, **_kwargs: object) -> object:
            return tracked_load(SimpleNamespace(), metadata=True)

        def import_module(_module_path: str) -> object:
            return tracked_load(SimpleNamespace(), metadata=False)

        def read_metadata() -> object:
            start.wait()
            return package.metadata

        def read_presets_module() -> object:
            start.wait()
            return package.presets_module

        with (
            patch(
                "model_runtime.packages.definition.load_model_metadata_from_module_path",
                side_effect=load_metadata,
            ),
            patch(
                "model_runtime.packages.definition.importlib.import_module",
                side_effect=import_module,
            ),
            ThreadPoolExecutor(max_workers=3) as executor,
        ):
            futures = [
                executor.submit(read_metadata),
                executor.submit(read_metadata),
                executor.submit(read_presets_module),
            ]
            start.wait()
            metadata_a, metadata_b, presets_module = [
                future.result(timeout=2) for future in futures
            ]

        self.assertEqual(maximum_active_loads, 1)
        self.assertEqual(metadata_loads, 1)
        self.assertEqual(preset_module_loads, 1)
        self.assertIs(metadata_a, metadata_b)
        self.assertIs(presets_module, package.presets_module)

    def test_cold_initialization_is_serial_across_sibling_packages(self) -> None:
        package_a = ModelPackage("test", "a", "models.test.a")
        package_b = ModelPackage("test", "b", "models.test.b")
        start = threading.Barrier(3)
        counter_lock = threading.Lock()
        active_loads = 0
        maximum_active_loads = 0

        def load_metadata(*_args: object, **_kwargs: object) -> object:
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

        def read_metadata(package: ModelPackage) -> object:
            start.wait()
            return package.metadata

        with (
            patch(
                "model_runtime.packages.definition.load_model_metadata_from_module_path",
                side_effect=load_metadata,
            ),
            ThreadPoolExecutor(max_workers=2) as executor,
        ):
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

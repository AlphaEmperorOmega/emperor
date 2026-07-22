from __future__ import annotations

import unittest

from torch.nn import Module

from emperor.config import ModelConfig
from model_runtime.packages import ModelPackage
from models.catalog import (
    MODEL_CATALOG,
    discover_model_ids,
    discover_model_packages,
    model_package,
)


class TestModelPackageInterface(unittest.TestCase):
    def test_every_catalog_entry_uses_the_same_model_package_interface(self):
        packages = discover_model_packages()

        self.assertEqual(len(packages), len(MODEL_CATALOG))
        self.assertEqual(
            [package.catalog_key for package in packages],
            discover_model_ids(),
        )

        for package in packages:
            with self.subTest(model_package=package.catalog_key):
                package_module = (
                    f"models.{package.identity.model_type}.{package.identity.model}"
                )
                self.assertIsInstance(package, ModelPackage)
                self.assertIs(model_package(package.catalog_key), package)
                self.assertEqual(
                    package.identity.to_payload(),
                    {
                        "modelType": package.identity.model_type,
                        "model": package.identity.model,
                    },
                )
                self.assertEqual(
                    package.runtime_defaults.__name__,
                    f"{package_module}.config",
                )
                runtime = package.bind_runtime_defaults()
                self.assertIs(type(runtime), package.runtime_options_type)
                self.assertEqual(
                    type(runtime).__module__,
                    f"{package_module}.runtime_options",
                )
                self.assertIn(
                    package.default_experiment_task,
                    package.dataset_metadata,
                )
                self.assertEqual(
                    package.metadata.dataset_options.__name__,
                    f"{package_module}.dataset_options",
                )
                self.assertEqual(
                    package.metadata.monitor_options_source.__name__,
                    f"{package_module}.monitor_options",
                )
                self.assertEqual(
                    package.metadata.search_space.__name__,
                    f"{package_module}.search_space",
                )
                self.assertTrue(package.preset_type)
                self.assertEqual(
                    package.preset_type.__module__,
                    f"{package_module}.presets",
                )
                self.assertEqual(
                    type(package.presets).__module__,
                    f"{package_module}.presets",
                )
                selected_preset = next(iter(package.preset_type))
                self.assertIsInstance(package.preset_locks(selected_preset), dict)

                configurations = package.build_configurations()
                self.assertTrue(configurations)
                self.assertTrue(
                    all(
                        isinstance(configuration, ModelConfig)
                        for configuration in configurations
                    )
                )
                model = package.build_model(configurations[0])
                self.assertIsInstance(model, Module)


if __name__ == "__main__":
    unittest.main()

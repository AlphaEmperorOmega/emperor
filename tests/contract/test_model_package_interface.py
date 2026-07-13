from __future__ import annotations

import unittest

from emperor.config import ModelConfig
from models.catalog import (
    MODEL_CATALOG,
    discover_model_ids,
    discover_model_packages,
    model_package,
)
from torch.nn import Module

from model_runtime.packages import ModelPackage


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
                self.assertIsInstance(package, ModelPackage)
                self.assertIs(model_package(package.catalog_key), package)
                self.assertEqual(
                    package.identity.to_payload(),
                    {
                        "modelType": package.model_type,
                        "model": package.model,
                    },
                )
                self.assertEqual(
                    package.runtime_defaults.__name__,
                    f"{package.module_path}.config",
                )
                self.assertIn(
                    package.default_experiment_task,
                    package.dataset_metadata,
                )
                self.assertEqual(
                    package.metadata.dataset_options_module.__name__,
                    f"{package.module_path}.dataset_options",
                )
                self.assertEqual(
                    package.metadata.monitor_options_module.__name__,
                    f"{package.module_path}.monitor_options",
                )
                self.assertEqual(
                    package.metadata.search_space_module.__name__,
                    f"{package.module_path}.search_space",
                )
                self.assertTrue(package.preset_type)
                self.assertEqual(
                    package.preset_type.__module__,
                    f"{package.module_path}.presets",
                )
                self.assertEqual(
                    type(package.presets).__module__,
                    f"{package.module_path}.presets",
                )
                self.assertEqual(
                    package.experiment_type.__module__,
                    f"{package.module_path}.presets",
                )

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
                self.assertEqual(
                    type(model).__module__,
                    f"{package.module_path}.model",
                )


if __name__ == "__main__":
    unittest.main()

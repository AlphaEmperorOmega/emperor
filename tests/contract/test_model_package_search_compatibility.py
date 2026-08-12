from __future__ import annotations

import unittest

from models.catalog import model_package

_MLP_MIXER_PACKAGES = (
    "mlp_mixer/linear",
    "mlp_mixer/linear_adaptive",
    "mlp_mixer/expert_linear",
    "mlp_mixer/expert_linear_adaptive",
)


class ModelPackageSearchCompatibilityContractTests(unittest.TestCase):
    def test_advertised_patch_sizes_build_for_every_declared_dataset(self) -> None:
        failures: list[str] = []
        for catalog_key in _MLP_MIXER_PACKAGES:
            package = model_package(catalog_key)
            self.assertIsNotNone(package)
            assert package is not None
            patch_sizes = package.search_metadata["SEARCH_SPACE_IMAGE_PATCH_SIZE"]
            datasets = {
                dataset
                for task_datasets in package.dataset_metadata.values()
                for dataset in task_datasets
            }
            for dataset in datasets:
                for patch_size in patch_sizes:
                    try:
                        package.build_configuration(
                            dataset=dataset,
                            config_overrides={"image_patch_size": patch_size},
                        )
                    except (TypeError, ValueError) as error:
                        failures.append(
                            f"{catalog_key}:{dataset.__name__}:{patch_size}: {error}"
                        )

        self.assertEqual([], failures, "\n" + "\n".join(failures))

    def test_large_cifar_patch_remains_available_as_a_custom_override(self) -> None:
        for catalog_key in _MLP_MIXER_PACKAGES:
            package = model_package(catalog_key)
            self.assertIsNotNone(package)
            assert package is not None
            cifar10 = next(
                dataset
                for task_datasets in package.dataset_metadata.values()
                for dataset in task_datasets
                if dataset.__name__ == "Cifar10"
            )
            with self.subTest(model_package=catalog_key):
                configuration = package.build_configuration(
                    dataset=cifar10,
                    config_overrides={"image_patch_size": 16},
                )
                experiment_config = configuration.experiment_config
                self.assertIsNotNone(experiment_config)
                assert experiment_config is not None
                self.assertEqual(experiment_config.patch_config.patch_size, 16)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from model_runtime.packages import config_key_to_model_param
from models.catalog import discover_model_packages


def _compatible_preset(package, field: str, value: object):
    candidates = []
    for preset in package.preset_type:
        lock = package.preset_locks(preset).get(field)
        if lock is None or lock.value == value:
            candidates.append(preset)
    if not candidates:
        return package.default_preset
    return min(candidates, key=lambda preset: len(package.preset_locks(preset)))


class ModelPackageConfigReachabilityContractTests(unittest.TestCase):
    def test_every_advertised_default_builds_with_a_declared_compatible_dataset(
        self,
    ) -> None:
        """Prove key reachability without imposing cross-dataset compatibility."""

        failures: list[str] = []

        for package in discover_model_packages():
            runtime_defaults = package.runtime_defaults_spec
            default_dataset = package.resolve_dataset(None)
            compatible_datasets = [default_dataset]
            for datasets in package.dataset_metadata.values():
                for dataset in datasets:
                    if dataset not in compatible_datasets:
                        compatible_datasets.append(dataset)
            for key in runtime_defaults.supported_keys:
                field = config_key_to_model_param(key)
                value = runtime_defaults.current_value(key)
                preset = _compatible_preset(package, field, value)
                errors: list[TypeError | ValueError] = []
                for dataset in compatible_datasets:
                    try:
                        package.build_configuration(
                            preset=preset,
                            dataset=dataset,
                            config_overrides={field: value},
                        )
                    except (TypeError, ValueError) as error:
                        errors.append(error)
                    else:
                        break
                else:
                    error = errors[0]
                    failures.append(
                        f"{package.catalog_key}:{key}: {type(error).__name__}: {error}"
                    )

        self.assertEqual([], failures, "\n" + "\n".join(failures))


if __name__ == "__main__":
    unittest.main()

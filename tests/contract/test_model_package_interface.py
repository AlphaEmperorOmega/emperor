from __future__ import annotations

import hashlib
import inspect
import json
import pickle
import random
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from torch.nn import Module

from emperor.config import ModelConfig
from model_runtime.packages import ModelPackage
from model_runtime.runs import (
    ExperimentBase,
    PlanningBudget,
    RunRequest,
    SearchSpec,
    plan_runs,
)
from model_runtime.runs._handoff import require_run_experiment
from model_runtime.runs.artifacts import FilesystemRunArtifacts
from models.catalog import (
    MODEL_CATALOG,
    discover_model_ids,
    discover_model_packages,
    model_package,
)

_ORDERING_DIGEST_BY_PACKAGE = {
    "bert/linear": (
        "eaafbc9ef2d5cca6e0623d9fce9082bf40c7ecfde569894b36ba2adbd8a7fd87",
        345,
        5,
    ),
    "bert/linear_adaptive": (
        "f065d9d32e9a644f9520d7ceaa18b884df2f1e248fe8760ac0931a109779feee",
        559,
        111,
    ),
    "bert/expert_linear": (
        "517f843a7ab248d84705edb49ce3f754cca960989072269d0e69db7f81fdb418",
        464,
        6,
    ),
    "bert/expert_linear_adaptive": (
        "44905590d799663fdd9e833702b5b78b006199fb78bc740da57863be8ee1db97",
        688,
        35,
    ),
    "experts/linear": (
        "66406436190a8ad9fe96bf14cc474c1fe65628e5a7c169cec10237c2c2b8669e",
        268,
        6,
    ),
    "experts/linear_adaptive": (
        "7729b7c6bb6e1deb91c13a9a57780ef05c147663cf5054d850fbf541450fca12",
        534,
        35,
    ),
    "gpt/linear": (
        "6fda739e4d7014e6e9390358941af77e338dc4ddc440ae34d3727b3d1a885ba1",
        337,
        5,
    ),
    "gpt/linear_adaptive": (
        "44330b26310380af858309b58010bebe676ebb1b1fc4f7dbda27adcba81c4726",
        551,
        111,
    ),
    "gpt/expert_linear": (
        "a8ae473b01ac94e45ef25313643d216642f2231a5d2a3f1ab99e19129f8d7e86",
        456,
        6,
    ),
    "gpt/expert_linear_adaptive": (
        "32c3bf63fc7727ae7923b3ef1125e8c34a83c778ba4c98c57c0765487205a14c",
        682,
        35,
    ),
    "linears/linear": (
        "85ec35e6c9b9e47c4f17adb7fa2bf136a7053812a4a49049ec9142cb853e1957",
        149,
        6,
    ),
    "linears/linear_adaptive": (
        "b5cd8723934ccc8ad529ac75e731290220f6fb120baafe6d89d4cc4dcd4c8bbf",
        267,
        31,
    ),
    "mlp_mixer/linear": (
        "23616d91c8cdc313ad66e6d10223677d036936758db48c249418ce0bf65d6110",
        352,
        7,
    ),
    "mlp_mixer/linear_adaptive": (
        "3dc0762eefd35f3f0d66d5e0b51f772fea2796e7e90933a7cfb0b2059a48e569",
        430,
        8,
    ),
    "mlp_mixer/expert_linear": (
        "ad6aecbb7753d96e72eb441a2c0eb515b0dc20fc9d1a03aaa959303ed8c7a45c",
        469,
        9,
    ),
    "mlp_mixer/expert_linear_adaptive": (
        "3f36b31116bf6dc27bf1e8a28883e170954a66ac007a5ca878d9bf32fd2d833f",
        547,
        10,
    ),
    "neuron/linear": (
        "9a4cdb0703cb8d74228e2142c3b75ad208f5115721980517e2b19b5c723804f3",
        203,
        9,
    ),
    "neuron/linear_adaptive": (
        "56433e1f691d92af59dc724c794691cac7c0ec86e1bf1bdbd793c1e6b49dead6",
        321,
        34,
    ),
    "neuron/expert_linear": (
        "7bd00fd8d74b13cebdabbe539c47f4b6536288992049d9bf82d9cb6af3ad1b32",
        322,
        9,
    ),
    "neuron/expert_linear_adaptive": (
        "7ab80485ab5b94c984cca9896885b45e610520e7b0011d9ff71cf15762c5d561",
        588,
        38,
    ),
    "parametric/parametric_generator": (
        "4deb456afa016c4dca7b206dd7270f0b6ee4deb3d04b66cfad5dff55549876b0",
        71,
        7,
    ),
    "parametric/parametric_matrix": (
        "ad8d5b713107f42679ca1cc226aff8d3adeef425418ba949c5c10430311768c9",
        67,
        7,
    ),
    "parametric/parametric_vector": (
        "6a03511dbff5243b343b6ffc8c39e92e30436a4520037e6f0dd32dee2a922309",
        66,
        6,
    ),
    "transformer/linear": (
        "39be4bed03b0e82f6e43621222c2572988e38f18237a654b450655a3a8e2fc57",
        775,
        8,
    ),
    "transformer/linear_adaptive": (
        "5f7f0eea2196acfe549e510f917cb002dad91ef080180b1f9da0667a3a4b1a96",
        1911,
        16,
    ),
    "transformer/expert_linear": (
        "0fc525cd0d84845c31194f30cf42d69283b7241af19ae8e063924f448bd6c3a6",
        910,
        8,
    ),
    "transformer/expert_linear_adaptive": (
        "87d76d9ee2e3536f2b343a6fe946c5baa1314b1aa932283886f419600b24bed1",
        1950,
        26,
    ),
    "vit/linear": (
        "ebd6a2d092a4dbdc4a192653bdbd29f24662e24c07d7b1daf717313c12ac955f",
        338,
        6,
    ),
    "vit/linear_adaptive": (
        "fb617d740b67964458b7bae7aec3eb72cd3bfa97d44042bda060920e67531c24",
        552,
        111,
    ),
    "vit/expert_linear": (
        "43ef7dc72bc668700387883d9e5f3788535442bdf129cae1a33ded11b039b14e",
        457,
        6,
    ),
    "vit/expert_linear_adaptive": (
        "6afd48dbd6a4bf6a62bf7174916e3790a170f1973a7093441433c92e4d6102be",
        684,
        35,
    ),
}


class TestModelPackageInterface(unittest.TestCase):
    def test_every_catalog_package_preserves_runtime_parameter_and_axis_order(self):
        actual = {}
        for package in discover_model_packages():
            runtime_defaults = package.runtime_defaults_spec
            parameter_keys = list(runtime_defaults.ordered_configuration_keys())
            plan = plan_runs(
                package,
                RunRequest(
                    presets=(package.preset_name(package.default_preset),),
                    datasets=(package.resolve_dataset(None).__name__,),
                    search=SearchSpec(mode="random", random_samples=1),
                ),
                random_source=random.Random(0),
                budget=PlanningBudget.unlimited(),
            )
            search = plan.search
            assert search is not None
            axis_keys = [axis.key for axis in search.axes or ()]
            payload = json.dumps(
                {"parameters": parameter_keys, "axes": axis_keys},
                separators=(",", ":"),
            )
            actual[package.catalog_key] = (
                hashlib.sha256(payload.encode()).hexdigest(),
                len(parameter_keys),
                len(axis_keys),
            )

        self.assertEqual(actual, _ORDERING_DIGEST_BY_PACKAGE)

    def test_runtime_defaults_interpretation_is_authoritative_and_cached(self):
        package = model_package("linears/linear")
        assert package is not None

        first = package.runtime_defaults_spec
        second = package.runtime_defaults_spec
        supported_keys = set(first.supported_keys)

        self.assertIs(first, second)
        self.assertEqual(set(first.supported_keys), supported_keys)
        default_keys = tuple(key for key, _value in first.default_items())
        self.assertEqual(set(default_keys), supported_keys)
        self.assertEqual(len(default_keys), len(supported_keys))
        self.assertEqual(first.resolve_key("hidden-dim"), "HIDDEN_DIM")
        self.assertEqual(first.resolve_key("hidden_dim"), "HIDDEN_DIM")
        self.assertEqual(
            first.parse_overrides({"hidden-dim": "64"}),
            {"hidden_dim": 64},
        )
        self.assertEqual(
            first.serialize_overrides({"hidden-dim": "64"}),
            {"HIDDEN_DIM": 64},
        )
        self.assertEqual(
            tuple(first.search_values),
            tuple(package.search_metadata),
        )
        self.assertTrue(first.preset_locks("gating"))

        curated_package = model_package("gpt/expert_linear")
        assert curated_package is not None
        curated = curated_package.runtime_defaults_spec
        curated_supported_keys = set(curated.supported_keys)
        self.assertNotIn("HALTING_OUTPUT_DIM", curated_supported_keys)
        self.assertTrue(curated.skipped_schema_keys)
        self.assertTrue(curated.skipped_schema_keys <= curated_supported_keys)

    def test_model_metadata_sources_are_not_part_of_the_package_interface(self):
        package = model_package("linears/linear")
        assert package is not None

        self.assertFalse(hasattr(package, "runtime_defaults"))
        for source_name in (
            "runtime_defaults",
            "dataset_options",
            "monitor_options_source",
            "search_space",
        ):
            with self.subTest(source=source_name):
                self.assertFalse(hasattr(package.metadata, source_name))

    def test_every_catalog_package_builds_the_typed_run_experiment_port(self):
        with TemporaryDirectory() as temporary_directory:
            artifacts = FilesystemRunArtifacts(root=Path(temporary_directory))
            for package in discover_model_packages():
                with self.subTest(model_package=package.catalog_key):
                    experiment = package.build_experiment(
                        package.default_preset,
                        experiment_task=package.default_experiment_task,
                        run_artifacts=artifacts,
                    )
                    self.assertIs(
                        require_run_experiment(experiment, package.catalog_key),
                        experiment,
                    )
                    self.assertEqual(
                        experiment.num_epochs,
                        package.runtime_defaults_spec.current_value_or(
                            "NUM_EPOCHS",
                            10,
                        ),
                    )

    def test_every_catalog_package_preserves_its_public_experiment_identity(self):
        experiment_types: set[type[ExperimentBase]] = set()
        with TemporaryDirectory() as temporary_directory:
            artifacts = FilesystemRunArtifacts(root=Path(temporary_directory))
            for package in discover_model_packages():
                with self.subTest(model_package=package.catalog_key):
                    experiment = package.build_experiment(
                        package.default_preset,
                        experiment_task=package.default_experiment_task,
                        run_artifacts=artifacts,
                    )
                    experiment_type = type(experiment)
                    expected_module = (
                        f"models.{package.identity.model_type}."
                        f"{package.identity.model}.presets"
                    )

                    self.assertIsNot(experiment_type, ExperimentBase)
                    self.assertTrue(issubclass(experiment_type, ExperimentBase))
                    self.assertEqual(experiment_type.__name__, "Experiment")
                    self.assertEqual(experiment_type.__module__, expected_module)
                    self.assertEqual(
                        tuple(inspect.signature(experiment_type).parameters),
                        (
                            "experiment_preset",
                            "experiment_task",
                            "model_package",
                            "run_artifacts",
                        ),
                    )
                    self.assertIs(
                        pickle.loads(pickle.dumps(experiment_type)),
                        experiment_type,
                    )
                    experiment_types.add(experiment_type)

        self.assertEqual(len(experiment_types), len(MODEL_CATALOG))

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
                self.assertIs(package.runtime_defaults_spec.package, package)
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
                self.assertIs(package.metadata.identity, package.identity)
                self.assertIsInstance(package.monitor_metadata, list)
                self.assertIsInstance(package.search_metadata, dict)
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

                configuration = package.build_configuration()
                self.assertIsInstance(configuration, ModelConfig)
                model = package.build_model(configuration)
                self.assertIsInstance(model, Module)


if __name__ == "__main__":
    unittest.main()

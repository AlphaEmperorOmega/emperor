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
        "89c8f59c481e0d53b728dc8166ec28181968576ac0c6950d072c22f50ebac32f",
        373,
        9,
    ),
    "bert/linear_adaptive": (
        "da3c671e971812e57cb20fd55f9b451d7760a2bd04fe966e0529033a5fd903e6",
        834,
        124,
    ),
    "bert/expert_linear": (
        "dba8ac6613a87ad2cd9c4294d657968c59d9966089eab4261d34fc9ad956efbf",
        500,
        10,
    ),
    "bert/expert_linear_adaptive": (
        "9015739bebf335008ad7d31360221e7376f63396700359f110efef1942310440",
        915,
        40,
    ),
    "experts/linear": (
        "7a63485a775991bde435a97fe7ace7ded5023c31a437edeba1e69ef71032ee65",
        287,
        7,
    ),
    "experts/linear_adaptive": (
        "e78713ab982fd82f450e8a0e833fc8f25af991d96db52019d11f1d766cbc3cf5",
        880,
        37,
    ),
    "gpt/linear": (
        "0bc262594db8cc2ef56d01b34e50708bf72628459cd75c063a155cb358ce9802",
        364,
        8,
    ),
    "gpt/linear_adaptive": (
        "81b8421ea7c7f7f9bae9825ddbda3a758cd67fe438345a948964a01aeaa8a9bf",
        825,
        123,
    ),
    "gpt/expert_linear": (
        "045d8df1941df9c8f3454b89f7bbd6de6f9989d6e037e371d7d664b1f98f66d2",
        491,
        9,
    ),
    "gpt/expert_linear_adaptive": (
        "6559e3d67f67ac0385820d590d1c453575f0a688dfe8e3b4bfa7297f5d919a9e",
        908,
        39,
    ),
    "linears/linear": (
        "2093377908e48de0665bd7f4c63fb7b8ca03d3672d429fff85822baddea0d8fb",
        160,
        7,
    ),
    "linears/linear_adaptive": (
        "cf6bcbc299701d28e8d929876841fe48642fd07e9ddf11e62075dad530c09899",
        517,
        33,
    ),
    "mlp_mixer/linear": (
        "e18cf34ba92c58f0c8a4d6b86dfebea16f1faacd4f737432bf061fd30b038b60",
        379,
        9,
    ),
    "mlp_mixer/linear_adaptive": (
        "f71a8301d9910f035d1c6f6bdafbac850122780d7893f52cc8e318555fd3365c",
        550,
        10,
    ),
    "mlp_mixer/expert_linear": (
        "58a7842502c265f746fcf8e0b5661931e2a7242d0185f33398cb166f35390df8",
        504,
        11,
    ),
    "mlp_mixer/expert_linear_adaptive": (
        "aba748e4f80a6af0b9e75d064aeff5b094bd9277bcd50ce6fe0f40f514110c9a",
        675,
        12,
    ),
    "neuron/linear": (
        "6f98de2ee92999f4ef5907beacd91aae487a1e89ae350fdfeaa8c600f8b183f5",
        220,
        10,
    ),
    "neuron/linear_adaptive": (
        "583faece6a7290b937849ffd9a11ace0b5495954c461a2f40406290d5813f7c8",
        577,
        36,
    ),
    "neuron/expert_linear": (
        "ab457987d08c5f5a284c94eab7a0481916a60e8a1bdcb045ae5bf03a1de00af7",
        347,
        10,
    ),
    "neuron/expert_linear_adaptive": (
        "cbd39b8ba892d7900885c2088af942234c0ee61b1b0ba5820bbaf9f4ef452ac6",
        940,
        40,
    ),
    "parametric/parametric_generator": (
        "4b9e4fd713db817400c090d1fa07ced411cb574cf1582b288cb564e3c9ee7214",
        72,
        7,
    ),
    "parametric/parametric_matrix": (
        "fd56b87316fae8fa511c3cb3fbe4aedea7577382d81724472588ec37977ec7c8",
        66,
        7,
    ),
    "parametric/parametric_vector": (
        "4efd1de70bed3b66f5d45ab559b65adcaf3ddeb078c7ed84304a94b906c3fa97",
        67,
        6,
    ),
    "transformer/linear": (
        "31bfb26a220d0a5f6b90fa52b60b7710d3389d25561538464f0b6a1c351ab363",
        830,
        12,
    ),
    "transformer/linear_adaptive": (
        "b9d4d18c9815752eddbcbf3f2d57b7942789b9daaadba5f2666780955e07d4fd",
        3182,
        20,
    ),
    "transformer/expert_linear": (
        "24b9abef1ccd48dee9844a91f43822ea635a683c8527e71a17c78c58f688d31d",
        980,
        10,
    ),
    "transformer/expert_linear_adaptive": (
        "ca7b7c8e9a65306d7e138b63d8e0083d1e48fbacc2071743eabec9d20c26ae20",
        3556,
        30,
    ),
    "vit/linear": (
        "5dba382f1c7bf26fabe6bdde50b442049296161386ee87d8cd6f9e1af9bcb4a4",
        364,
        8,
    ),
    "vit/linear_adaptive": (
        "8e04dc711a7acc7524b594d78d971c010d851027346bff5aaec4f62c7eefdd52",
        825,
        122,
    ),
    "vit/expert_linear": (
        "e795ab20898c060695a8e676172d26a86de5ef0da29fffd25562374bf00438ec",
        491,
        8,
    ),
    "vit/expert_linear_adaptive": (
        "c14ecc4b8a2fa4967facee1005c743a63825457c7179ebff5b50d12f250cb940",
        909,
        38,
    ),
}


_ORDERING_WITH_ATTENTION_SETTINGS_BY_PACKAGE = {
    "bert/linear": (
        "d4b448038c53b9fc119bc6bbbe234cfcbb262f59ae5fa81e439b8e6adeee546d",
        413,
        9,
    ),
    "bert/linear_adaptive": (
        "098c54e9fed196ad82a25245e332f28fa10c09289718819434942c899998d299",
        930,
        124,
    ),
    "bert/expert_linear": (
        "2b7937a119344f29e6d8f4f0cbbca3c8de3f3922dfbd26d0d0fa35e718173b4e",
        554,
        10,
    ),
    "bert/expert_linear_adaptive": (
        "93144c5c7a53ec0a4b9367a295e816ae55d913f95da79cee8c6bd7dcb80578f2",
        1017,
        40,
    ),
    "experts/linear": (
        "7a63485a775991bde435a97fe7ace7ded5023c31a437edeba1e69ef71032ee65",
        287,
        7,
    ),
    "experts/linear_adaptive": (
        "e78713ab982fd82f450e8a0e833fc8f25af991d96db52019d11f1d766cbc3cf5",
        880,
        37,
    ),
    "gpt/linear": (
        "0bc262594db8cc2ef56d01b34e50708bf72628459cd75c063a155cb358ce9802",
        364,
        8,
    ),
    "gpt/linear_adaptive": (
        "81b8421ea7c7f7f9bae9825ddbda3a758cd67fe438345a948964a01aeaa8a9bf",
        825,
        123,
    ),
    "gpt/expert_linear": (
        "045d8df1941df9c8f3454b89f7bbd6de6f9989d6e037e371d7d664b1f98f66d2",
        491,
        9,
    ),
    "gpt/expert_linear_adaptive": (
        "6559e3d67f67ac0385820d590d1c453575f0a688dfe8e3b4bfa7297f5d919a9e",
        908,
        39,
    ),
    "linears/linear": (
        "c525e3226785d9b331406a013ec97b4fd09cc93a53b74e3b464d2bf9640618ff",
        176,
        7,
    ),
    "linears/linear_adaptive": (
        "9997be086f46c9e63555a52592f6c179b2a5591f58ad6cf33662d8925d64f295",
        573,
        33,
    ),
    "mlp_mixer/linear": (
        "e18cf34ba92c58f0c8a4d6b86dfebea16f1faacd4f737432bf061fd30b038b60",
        379,
        9,
    ),
    "mlp_mixer/linear_adaptive": (
        "f71a8301d9910f035d1c6f6bdafbac850122780d7893f52cc8e318555fd3365c",
        550,
        10,
    ),
    "mlp_mixer/expert_linear": (
        "58a7842502c265f746fcf8e0b5661931e2a7242d0185f33398cb166f35390df8",
        504,
        11,
    ),
    "mlp_mixer/expert_linear_adaptive": (
        "aba748e4f80a6af0b9e75d064aeff5b094bd9277bcd50ce6fe0f40f514110c9a",
        675,
        12,
    ),
    "neuron/linear": (
        "6f98de2ee92999f4ef5907beacd91aae487a1e89ae350fdfeaa8c600f8b183f5",
        220,
        10,
    ),
    "neuron/linear_adaptive": (
        "583faece6a7290b937849ffd9a11ace0b5495954c461a2f40406290d5813f7c8",
        577,
        36,
    ),
    "neuron/expert_linear": (
        "ab457987d08c5f5a284c94eab7a0481916a60e8a1bdcb045ae5bf03a1de00af7",
        347,
        10,
    ),
    "neuron/expert_linear_adaptive": (
        "cbd39b8ba892d7900885c2088af942234c0ee61b1b0ba5820bbaf9f4ef452ac6",
        940,
        40,
    ),
    "parametric/parametric_generator": (
        "4b9e4fd713db817400c090d1fa07ced411cb574cf1582b288cb564e3c9ee7214",
        72,
        7,
    ),
    "parametric/parametric_matrix": (
        "fd56b87316fae8fa511c3cb3fbe4aedea7577382d81724472588ec37977ec7c8",
        66,
        7,
    ),
    "parametric/parametric_vector": (
        "4efd1de70bed3b66f5d45ab559b65adcaf3ddeb078c7ed84304a94b906c3fa97",
        67,
        6,
    ),
    "transformer/linear": (
        "0657e605bfab291e6de5cb2c21f79df72596ce65fca6067db555543a57e7698e",
        920,
        12,
    ),
    "transformer/linear_adaptive": (
        "36cf44988b972744c002920dfdf4a06e3447d6a0c5291eac63b7c4bf882d2b32",
        3572,
        20,
    ),
    "transformer/expert_linear": (
        "878f5a2dd11f60fd0573a0c05398f5c5e10e2324c068300510f3477f88798269",
        1094,
        10,
    ),
    "transformer/expert_linear_adaptive": (
        "ebc99ed2b9ea536c67bc97a75d0170ca8a30eb97cb44baad9e7b771b229ba994",
        3990,
        30,
    ),
    "vit/linear": (
        "5dba382f1c7bf26fabe6bdde50b442049296161386ee87d8cd6f9e1af9bcb4a4",
        364,
        8,
    ),
    "vit/linear_adaptive": (
        "8e04dc711a7acc7524b594d78d971c010d851027346bff5aaec4f62c7eefdd52",
        825,
        122,
    ),
    "vit/expert_linear": (
        "e795ab20898c060695a8e676172d26a86de5ef0da29fffd25562374bf00438ec",
        491,
        8,
    ),
    "vit/expert_linear_adaptive": (
        "c14ecc4b8a2fa4967facee1005c743a63825457c7179ebff5b50d12f250cb940",
        909,
        38,
    ),
}


class TestModelPackageInterface(unittest.TestCase):
    def test_every_catalog_package_preserves_runtime_parameter_and_axis_order(self):
        actual = {}
        actual_with_attention_settings = {}
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
            existing_parameter_keys = [
                key
                for key in parameter_keys
                if not key.endswith(
                    ("RESIDUAL_BLOCK_SIZE", "RESIDUAL_RMS_NORM_EPSILON")
                )
            ]
            for keys, signatures in (
                (existing_parameter_keys, actual),
                (parameter_keys, actual_with_attention_settings),
            ):
                payload = json.dumps(
                    {"parameters": keys, "axes": axis_keys},
                    separators=(",", ":"),
                )
                signatures[package.catalog_key] = (
                    hashlib.sha256(payload.encode()).hexdigest(),
                    len(keys),
                    len(axis_keys),
                )

        self.assertEqual(actual, _ORDERING_DIGEST_BY_PACKAGE)
        self.assertEqual(
            actual_with_attention_settings,
            _ORDERING_WITH_ATTENTION_SETTINGS_BY_PACKAGE,
        )

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

    def test_checkpoint_reconstruction_is_an_explicit_optional_capability(self):
        tensor_shapes = {
            "input_model.model.weight_params": (784, 12),
            "main_model.layers.0.model.weight_params": (12, 12),
            "output_model.model.weight_params": (12, 10),
        }
        expected = {
            "input_dim": 784,
            "output_dim": 10,
            "hidden_dim": 12,
            "stack_num_layers": 1,
        }

        for package in discover_model_packages():
            with self.subTest(model_package=package.catalog_key):
                overrides = package.checkpoint_config_overrides(tensor_shapes)
                if package.catalog_key == "linears/linear":
                    self.assertEqual(overrides, expected)
                else:
                    self.assertEqual(overrides, {})


if __name__ == "__main__":
    unittest.main()

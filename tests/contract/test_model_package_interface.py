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
        "42533e356822a2005f13f1ace941a7687f86d4119d63ed1339f8c5b1db35ee58",
        347,
        5,
    ),
    "bert/linear_adaptive": (
        "fbcf37a1d8fe89211b8c7691c1b69ec1fcab3fdba9c38c989e5e18b64a2e5106",
        561,
        111,
    ),
    "bert/expert_linear": (
        "e6b07b3ba83928844aac0864b9aa53ff7e04368041677989cdd1ba913eb9cc9c",
        466,
        6,
    ),
    "bert/expert_linear_adaptive": (
        "062aad406c09b2aeaf3fd1433f88186a7ce8c4c51cfbf6e01b4be2b117e0645b",
        690,
        35,
    ),
    "experts/linear": (
        "50891ccbda11a1fc8fa9517581d051c62d6778b9c01ae145385bb3599253406d",
        270,
        6,
    ),
    "experts/linear_adaptive": (
        "020228f44768d5ba4c6d86724966b5e873ed9cca330fd82e425d146b16a6cc72",
        536,
        35,
    ),
    "gpt/linear": (
        "304cb166575b74c9e1c8548f0ff2758071abffa56cb748c94c99103da0626cf0",
        339,
        5,
    ),
    "gpt/linear_adaptive": (
        "8c1c3fca9903fb3383c79f69f6ee637a9989e2142ff454e7364f388e99b5b843",
        553,
        111,
    ),
    "gpt/expert_linear": (
        "1402c879088b7e8391f6310b596147d98d2bae81b829cf9dcc463dde87c655cf",
        458,
        6,
    ),
    "gpt/expert_linear_adaptive": (
        "612fadee604839e00e197d5f6f3a691401938e7b49cac5d5e0230f685a682551",
        684,
        35,
    ),
    "linears/linear": (
        "403f9c1fafbb82d9fd3119628c326f2ca2a70f30d342eb607bfa938501be14df",
        151,
        6,
    ),
    "linears/linear_adaptive": (
        "411981b76971f6907e4151081a5326ce270842184531670fc4be27648971d3e3",
        269,
        31,
    ),
    "mlp_mixer/linear": (
        "1d9ed72881fdb8cd8a3d3e1ad4176bb717089c8bfc0f40b8bd6d840faf4d95c9",
        354,
        7,
    ),
    "mlp_mixer/linear_adaptive": (
        "360af9d91cf18a6dacb1b0ed8562baecba739b93710c0ef5f0a8332ea309cd95",
        432,
        8,
    ),
    "mlp_mixer/expert_linear": (
        "69b4371e50b64e5eefde2003e08e4fe6d14223d131f02c1e3f08f1cdedfa0f9e",
        471,
        9,
    ),
    "mlp_mixer/expert_linear_adaptive": (
        "d23bdd088081d752aa9a2370d4abb35a3e64584de94b933ea3422a1f6a332fd3",
        549,
        10,
    ),
    "neuron/linear": (
        "90b83de14da8f8b55c6f99be127103184ffc310aaf602e9544a5bb01c7bd0e36",
        205,
        9,
    ),
    "neuron/linear_adaptive": (
        "46df20c7bf8a071d9b56d90b6f36a91f71925a674ddffaf0dd7051f32e376631",
        323,
        34,
    ),
    "neuron/expert_linear": (
        "d390f7b57e3991d071ffd2e3b44fc5de14bb651bc21b0512fea7c759c5f1d545",
        324,
        9,
    ),
    "neuron/expert_linear_adaptive": (
        "ece6a4f13d73a3f44aff04bb9658a701e6bd078720cb3fb8ef8f40b22f27b905",
        590,
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
        "0292a6f85f220201afd984003cdffd36b1a2b32046da34a85da94b5c1dae5adc",
        776,
        8,
    ),
    "transformer/linear_adaptive": (
        "377c62bbebb154dc8ac12e270a6dd410cba375e3024842b1c7456c8f80be6abf",
        1913,
        16,
    ),
    "transformer/expert_linear": (
        "1966c3f74898037d460bdf06ddda8811c4e735bb7d8e409464d6e9e63ef24698",
        912,
        8,
    ),
    "transformer/expert_linear_adaptive": (
        "b17fcce08588cdef31fc6dcf4ba84ce22298a3429381eb4abe5f020e2f61201e",
        1952,
        26,
    ),
    "vit/linear": (
        "3cf90b10495a729c6b1a8565d453f7db5e05f4e2f04910c116130443c268f2f5",
        340,
        6,
    ),
    "vit/linear_adaptive": (
        "0720941a0920cc75442c6a664daf25202823e2be0796ee60059e746993da39d8",
        554,
        111,
    ),
    "vit/expert_linear": (
        "a7e6b891844214f1fa300477e3fda799fe0c8c81da879c3bdd2394e1ee9dc8d4",
        459,
        6,
    ),
    "vit/expert_linear_adaptive": (
        "584b11d0c7ff4e1bd8bcc6d38b9a65461bc70601423ed933f6c2afa0c274e296",
        686,
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

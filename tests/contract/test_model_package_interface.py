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
        "7c9569bbd1a6e22bdeddd5a5165a61e4c11f3a502c6d0a036ddc4477376f932c",
        347,
        5,
    ),
    "bert/linear_adaptive": (
        "70c1c9199bfec35d21411c9e3771240a96c6c1cbd5fb443c409f60d33d40ae61",
        780,
        111,
    ),
    "bert/expert_linear": (
        "9893c950d3333e7d69185643552d7a5f2c8e847020001a1f902a8c321d93b01c",
        466,
        6,
    ),
    "bert/expert_linear_adaptive": (
        "6b2f4e51601758bf7e71561705bea5e3b94804b91f2ccca81ea2e9d45527215c",
        856,
        35,
    ),
    "experts/linear": (
        "4c0b454f615e9053c21b21b138062bce9006c54be3ee1ab9976e512cd6b0762b",
        270,
        6,
    ),
    "experts/linear_adaptive": (
        "601678a78315af663f05c50c3acde5bea87de923013703fcc4841c2e4a929c86",
        828,
        35,
    ),
    "gpt/linear": (
        "d1916412b1846028e50da0831de745a8c451ebf04ee2266d85515c58ef6970b4",
        339,
        5,
    ),
    "gpt/linear_adaptive": (
        "93e39605adcdbae3033800aa6c7ead2b7eb708ead658c0260fc1a0519d1b1ce8",
        772,
        111,
    ),
    "gpt/expert_linear": (
        "8522ce5bec2146f350737afe84970f3f62ddf3a5456ad8a1a75ffdbfa23be744",
        458,
        6,
    ),
    "gpt/expert_linear_adaptive": (
        "a6e9eca648226a9e8b92aa09938f010b3219b0055f70607fbb0892ccc79ad1ef",
        850,
        35,
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
        "4b71d52b0c9f819d25ec63dcd57a1e06fa255a300b890202f181fedad706ea40",
        354,
        7,
    ),
    "mlp_mixer/linear_adaptive": (
        "1e559b7d472baae29174b3abee1e4dc8455ca3365ee0832301f26e0d2c10310c",
        515,
        8,
    ),
    "mlp_mixer/expert_linear": (
        "c89a4ac19d84b1283d84e5a1c92bc19137a31a5ac4347d438262ac8680eaef60",
        471,
        9,
    ),
    "mlp_mixer/expert_linear_adaptive": (
        "426ff7e0d4f426d8211bb47e143732f308e5b3d28797641b0626072bb0919220",
        632,
        10,
    ),
    "neuron/linear": (
        "358837b4c893c2d8ecbaa2de7afb0613ebe77b508c39ae03c6d0bf2deb909055",
        209,
        9,
    ),
    "neuron/linear_adaptive": (
        "f5c37f4682101a6d9ab81bc08190ade857549088680fb3080fd4bf744a8dad29",
        546,
        34,
    ),
    "neuron/expert_linear": (
        "ef6c04083da7d5451e7377066c6410977bdb2059ff901bfaacb4133dc685e30b",
        328,
        9,
    ),
    "neuron/expert_linear_adaptive": (
        "4c628684fc5415d14da3cd94244760f7052ee3d6517576efcf2d1f5b55c6f484",
        886,
        38,
    ),
    "parametric/parametric_generator": (
        "59b7820c17594cf276831e1b49ac0c0421f253addd4d4667a07e1676d9996cfa",
        71,
        7,
    ),
    "parametric/parametric_matrix": (
        "4b9ee33f6a15c925ac2f9989ce5dcc4e5bc504eb00a9f3550d00fb89b33bbd0a",
        65,
        7,
    ),
    "parametric/parametric_vector": (
        "cb2a5fe26e7f62fc9660c49368502c44557342beff8e4cd0767c1363e3c50731",
        66,
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
        "75fc826dce8aa52e8ab4a36cee87db972a515ef89dd663997c0841b5a88cb251",
        340,
        6,
    ),
    "vit/linear_adaptive": (
        "cb3be1944b8a6de5d81c328cd74219f2e3a7d5295a075ba128efca6477085bcd",
        773,
        111,
    ),
    "vit/expert_linear": (
        "09e34aa8d95204ad8912ca700f2bd5d5c3a140dd14aa30a987e0a91e681d3691",
        459,
        6,
    ),
    "vit/expert_linear_adaptive": (
        "6ab5d5e65ed345a822d0a10ff7391ff35e83a328e59710056b9da4f47628b840",
        852,
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

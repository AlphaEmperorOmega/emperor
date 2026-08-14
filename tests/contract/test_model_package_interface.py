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
        "f6484eab47c0a975a6ddfed7d718206d335ae041f3020afde957daffbcb5cd3c",
        346,
        5,
    ),
    "bert/linear_adaptive": (
        "2834ff8f867db8f529f49efeb9faa06e6daf7e72bfd11315c99a7788e3be26e5",
        560,
        111,
    ),
    "bert/expert_linear": (
        "72e11d06aa06ca7b8881d22140c5fb59e56566dab3b23c3735c74b6268f19a18",
        465,
        6,
    ),
    "bert/expert_linear_adaptive": (
        "12a2b36e6b37ecd66f1da98c972d1f3ee7b0035b170076344aed1615de204848",
        689,
        35,
    ),
    "experts/linear": (
        "d83bb2851484811461f5edbb8dbf52f6bc8413aadacd0d86fd099b8a2766d61e",
        269,
        6,
    ),
    "experts/linear_adaptive": (
        "7418705a906626a345b18d29401858abca110c7ef26c87eb1fb02d38e044a550",
        535,
        35,
    ),
    "gpt/linear": (
        "ff29671d728c8feeaec5e5eac8430bcdf80035c503c968a2575c25e1c271ee87",
        338,
        5,
    ),
    "gpt/linear_adaptive": (
        "bd79adf0d31d127c63ca61ff60942dae1c925e9ce9b220d2cfefbd7074de9e11",
        552,
        111,
    ),
    "gpt/expert_linear": (
        "c0b2bd78f02ca911da2a976137fb6c139664ccb7cff41a50b06c350ba6d95575",
        457,
        6,
    ),
    "gpt/expert_linear_adaptive": (
        "599b049af932889077bde17eb416ebc1816bc10775ec3b5a03b1d2a884fd3715",
        683,
        35,
    ),
    "linears/linear": (
        "1079e6b030c1e92897a27528599cceda0bfd8f7a2e628f67a19375bf9d91112a",
        150,
        6,
    ),
    "linears/linear_adaptive": (
        "26f8d46103c7c756f35c520552a1a11c917b0a1a91bb8ad0a04778958a966333",
        268,
        31,
    ),
    "mlp_mixer/linear": (
        "4487a0f52ed1c6ab44054206e641bcb4c37535aee0ee1d528cf2ee118cbce3c8",
        353,
        7,
    ),
    "mlp_mixer/linear_adaptive": (
        "034b6368333ad11f2b2b65019963f18e48e75f4f60d0d6c6c0f363daffb315ac",
        431,
        8,
    ),
    "mlp_mixer/expert_linear": (
        "ba4f8ebdc63190bbebb611ca50e79f0ba980756a28329b7d338a6324f3d6fa55",
        470,
        9,
    ),
    "mlp_mixer/expert_linear_adaptive": (
        "f4de748fd3ff7019945d71ee2c338e3b9861484c56670d7b6c6bb78dd9387558",
        548,
        10,
    ),
    "neuron/linear": (
        "e4d4316eff689083b5671be3e7f64090c8f0dee68256f62766794b258c994c74",
        204,
        9,
    ),
    "neuron/linear_adaptive": (
        "5217b2750dd943ba4174370e16807c8db28cf5b073cca53335a6beefe68f6e97",
        322,
        34,
    ),
    "neuron/expert_linear": (
        "f824b22d88258957f86d5b32d1f8156e61be24b16311389d1ca94dd90881edc2",
        323,
        9,
    ),
    "neuron/expert_linear_adaptive": (
        "63a8054b569f9e54af248a1841216be72a83a45a446bd98f65bbaa896e75880a",
        589,
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
        "c331bd7c38ac5489bd70084f4b911d161a80215198cd5acb0616b9dc889bb1c2",
        1912,
        16,
    ),
    "transformer/expert_linear": (
        "3b7a8b2824a911141c622e3bf7b6de1510654425fe30ae611fc5cf28fd26c076",
        911,
        8,
    ),
    "transformer/expert_linear_adaptive": (
        "cfbb141e78e120470ed50ff29229f5acdaf480d96c70813f1a653175446add8b",
        1951,
        26,
    ),
    "vit/linear": (
        "aac2dd0b31731a4585b8251fc28653d8e08367d9298aaf6aa94eaf35b720f2fa",
        339,
        6,
    ),
    "vit/linear_adaptive": (
        "ea6a8accd19481c846f6c60da98ca37d6acdcbfdd985495ea4163206473fecdd",
        553,
        111,
    ),
    "vit/expert_linear": (
        "fa1629ea3348f68abcdc551774389bb69d727e9d39a13c571ef9ae4313a43ba2",
        458,
        6,
    ),
    "vit/expert_linear_adaptive": (
        "367f2796adcf499a0316c874816a794f1bf03092b736e80364973c060f75d228",
        685,
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

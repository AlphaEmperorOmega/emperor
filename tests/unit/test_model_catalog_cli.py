import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from model_runtime.packages.identity import is_safe_model_segment, split_model_id
from models.catalog import (
    discover_model_identities_for_type,
    discover_model_ids,
    discover_model_types,
    is_safe_model_id,
    model_package,
    model_type_exists,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


class TestModelCatalogDiscovery(unittest.TestCase):
    def test_model_types_are_sorted_and_unique(self):
        model_types = discover_model_types()

        self.assertEqual(
            model_types,
            [
                "bert",
                "experts",
                "gpt",
                "linears",
                "neuron",
                "parametric",
                "transformer",
                "vit",
            ],
        )
        self.assertEqual(model_types, sorted(model_types))
        self.assertEqual(len(model_types), len(set(model_types)))

    def test_linears_model_type_returns_only_linear_identities(self):
        identities = discover_model_identities_for_type("linears")

        self.assertEqual(
            [(identity.model_type, identity.model) for identity in identities],
            [
                ("linears", "linear"),
                ("linears", "linear_adaptive"),
            ],
        )

    def test_unknown_or_unsafe_model_type_returns_no_identities(self):
        for model_type in ["missing", "../linears"]:
            with self.subTest(model_type=model_type):
                self.assertFalse(model_type_exists(model_type))
                self.assertEqual(
                    discover_model_identities_for_type(model_type),
                    [],
                )

    def test_identity_parsing_rejects_non_string_boundary_values(self):
        for value in (None, 1, True, object()):
            with self.subTest(value=value):
                self.assertFalse(is_safe_model_segment(value))
                self.assertFalse(is_safe_model_id(value))
                self.assertIsNone(split_model_id(value))

    def test_retired_and_flat_identities_are_unsupported(self):
        for identity in (
            "linear",
            "linear_adaptive",
            "expert_linear",
            "vit_linear",
            "neuron_linear",
            "transformer_encoder/vit_linear",
            "models.linears.linear",
        ):
            with self.subTest(identity=identity):
                self.assertIsNone(model_package(identity))


class TestModelCatalogCli(unittest.TestCase):
    def run_catalog(self, *args):
        return subprocess.run(
            [sys.executable, "-m", "models.catalog", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_list_types_prints_available_model_types(self):
        completed = self.run_catalog("--list-types")

        self.assertEqual(completed.returncode, 0)
        self.assertEqual(completed.stderr, "")
        self.assertEqual(
            completed.stdout.splitlines(),
            [
                "bert",
                "experts",
                "gpt",
                "linears",
                "neuron",
                "parametric",
                "transformer",
                "vit",
            ],
        )

    def test_list_prints_full_copyable_catalog(self):
        completed = self.run_catalog("--list")

        self.assertEqual(completed.returncode, 0)
        self.assertEqual(completed.stderr, "")
        self.assertEqual(completed.stdout.splitlines(), discover_model_ids())

    def test_list_for_model_type_prints_canonical_identities(self):
        completed = self.run_catalog("--list", "--model-type", "linears")

        self.assertEqual(completed.returncode, 0)
        self.assertEqual(completed.stderr, "")
        self.assertEqual(
            completed.stdout.splitlines(),
            [
                "linears/linear",
                "linears/linear_adaptive",
            ],
        )

    def test_list_for_bert_model_type_prints_canonical_identities(self):
        completed = self.run_catalog("--list", "--model-type", "bert")

        self.assertEqual(completed.returncode, 0)
        self.assertEqual(completed.stderr, "")
        self.assertEqual(
            completed.stdout.splitlines(),
            [
                "bert/linear",
                "bert/linear_adaptive",
                "bert/expert_linear",
                "bert/expert_linear_adaptive",
            ],
        )

    def test_list_for_vit_model_type_prints_canonical_identities(self):
        completed = self.run_catalog("--list", "--model-type", "vit")

        self.assertEqual(completed.returncode, 0)
        self.assertEqual(completed.stderr, "")
        self.assertEqual(
            completed.stdout.splitlines(),
            [
                "vit/linear",
                "vit/linear_adaptive",
                "vit/expert_linear",
                "vit/expert_linear_adaptive",
            ],
        )

    def test_list_for_gpt_model_type_prints_canonical_identities(self):
        completed = self.run_catalog("--list", "--model-type", "gpt")

        self.assertEqual(completed.returncode, 0)
        self.assertEqual(completed.stderr, "")
        self.assertEqual(
            completed.stdout.splitlines(),
            [
                "gpt/linear",
                "gpt/linear_adaptive",
                "gpt/expert_linear",
                "gpt/expert_linear_adaptive",
            ],
        )

    def test_list_for_transformer_model_type_prints_canonical_identities(self):
        completed = self.run_catalog("--list", "--model-type", "transformer")

        self.assertEqual(completed.returncode, 0)
        self.assertEqual(completed.stderr, "")
        self.assertEqual(
            completed.stdout.splitlines(),
            [
                "transformer/linear",
                "transformer/linear_adaptive",
                "transformer/expert_linear",
                "transformer/expert_linear_adaptive",
            ],
        )

    def test_list_for_neuron_model_type_prints_canonical_identities(self):
        completed = self.run_catalog("--list", "--model-type", "neuron")

        self.assertEqual(completed.returncode, 0)
        self.assertEqual(completed.stderr, "")
        self.assertEqual(
            completed.stdout.splitlines(),
            [
                "neuron/linear",
                "neuron/linear_adaptive",
                "neuron/expert_linear",
                "neuron/expert_linear_adaptive",
            ],
        )

    def test_list_for_unknown_model_type_fails(self):
        completed = self.run_catalog("--list", "--model-type", "missing")

        self.assertNotEqual(completed.returncode, 0)
        self.assertEqual(completed.stdout, "")
        self.assertIn("Unknown model type: --model-type missing", completed.stderr)


class TestConfigOverrideCli(unittest.TestCase):
    def run_config_overrides(self, *args):
        return subprocess.run(
            [sys.executable, "-m", "models.config_overrides", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_list_monitors_prints_raw_monitor_names(self):
        completed = self.run_config_overrides("linears/linear", "--monitors")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stderr, "")
        self.assertEqual(
            completed.stdout.splitlines(),
            [
                "linear",
                "recurrent-layer",
                "layer-controller",
                "halting",
                "memory",
            ],
        )


class TestProjectCatalogCli(unittest.TestCase):
    def run_experiment(self, *args):
        with tempfile.TemporaryDirectory() as matplotlib_config_dir:
            environment = dict(os.environ)
            environment["MPLCONFIGDIR"] = matplotlib_config_dir
            return subprocess.run(
                [sys.executable, "-m", "models.project_cli", *args],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
                env=environment,
            )

    def test_no_argument_help_describes_checkpoint_continuation(self):
        completed = self.run_experiment()

        self.assertEqual(completed.returncode, 0)
        self.assertEqual(completed.stderr, "")
        self.assertIn("--resume-checkpoint <path>", completed.stdout)
        self.assertIn("--datasets mnist --resume-checkpoint", completed.stdout)
        self.assertIn("--print-model-shapes", completed.stdout)
        self.assertIn("--print-model-tensor-shapes", completed.stdout)

    def test_list_model_types_prints_copyable_model_type_flags(self):
        completed = self.run_experiment("--list-model-types")

        self.assertEqual(completed.returncode, 0)
        self.assertEqual(completed.stderr, "")
        self.assertEqual(
            completed.stdout.splitlines(),
            [
                (
                    "Usage: mise run experiment -- --model-type <type> "
                    "--model <name> [options]"
                ),
                "",
                "Available model types:",
                "  --model-type bert",
                "  --model-type experts",
                "  --model-type gpt",
                "  --model-type linears",
                "  --model-type neuron",
                "  --model-type parametric",
                "  --model-type transformer",
                "  --model-type vit",
            ],
        )

    def test_list_models_keeps_full_catalog_behavior(self):
        completed = self.run_experiment("--list-models")

        self.assertEqual(completed.returncode, 0)
        self.assertEqual(completed.stderr, "")
        self.assertIn("Available models:", completed.stdout)
        self.assertIn("  --model-type bert --model linear", completed.stdout)
        self.assertIn(
            "  --model-type bert --model linear_adaptive",
            completed.stdout,
        )
        self.assertIn("  --model-type gpt --model linear", completed.stdout)
        self.assertIn(
            "  --model-type gpt --model linear_adaptive",
            completed.stdout,
        )
        self.assertIn(
            "  --model-type gpt --model expert_linear",
            completed.stdout,
        )
        self.assertIn(
            "  --model-type gpt --model expert_linear_adaptive",
            completed.stdout,
        )
        self.assertIn("  --model-type linears --model linear", completed.stdout)
        self.assertIn(
            "  --model-type linears --model linear_adaptive",
            completed.stdout,
        )
        self.assertIn("  --model-type vit --model linear", completed.stdout)
        self.assertIn(
            "  --model-type vit --model linear_adaptive",
            completed.stdout,
        )
        self.assertIn(
            "  --model-type vit --model expert_linear",
            completed.stdout,
        )
        self.assertIn(
            "  --model-type vit --model expert_linear_adaptive",
            completed.stdout,
        )
        self.assertIn(
            "  --model-type transformer --model linear",
            completed.stdout,
        )
        self.assertIn(
            "  --model-type transformer --model linear_adaptive",
            completed.stdout,
        )
        self.assertIn(
            "  --model-type transformer --model expert_linear",
            completed.stdout,
        )
        self.assertIn(
            "  --model-type transformer --model expert_linear_adaptive",
            completed.stdout,
        )
        self.assertIn("  --model-type neuron --model linear", completed.stdout)
        self.assertIn(
            "  --model-type neuron --model linear_adaptive",
            completed.stdout,
        )
        self.assertIn(
            "  --model-type neuron --model expert_linear",
            completed.stdout,
        )
        self.assertIn(
            "  --model-type neuron --model expert_linear_adaptive",
            completed.stdout,
        )

    def test_list_models_for_model_type_prints_copyable_model_flags(self):
        completed = self.run_experiment("--model-type", "linears", "--list-models")

        self.assertEqual(completed.returncode, 0)
        self.assertEqual(completed.stderr, "")
        self.assertEqual(
            completed.stdout.splitlines(),
            [
                (
                    "Usage: mise run experiment -- --model-type linears "
                    "--model <name> [options]"
                ),
                "",
                "Available models for --model-type linears:",
                "  --model linear",
                "  --model linear_adaptive",
            ],
        )

    def test_list_models_for_bert_model_type_prints_backend_flags(self):
        completed = self.run_experiment("--model-type", "bert", "--list-models")

        self.assertEqual(completed.returncode, 0)
        self.assertEqual(completed.stderr, "")
        self.assertEqual(
            completed.stdout.splitlines(),
            [
                (
                    "Usage: mise run experiment -- --model-type bert "
                    "--model <name> [options]"
                ),
                "",
                "Available models for --model-type bert:",
                "  --model linear",
                "  --model linear_adaptive",
                "  --model expert_linear",
                "  --model expert_linear_adaptive",
            ],
        )

    def test_list_models_for_vit_model_type_prints_backend_flags(self):
        completed = self.run_experiment("--model-type", "vit", "--list-models")

        self.assertEqual(completed.returncode, 0)
        self.assertEqual(completed.stderr, "")
        self.assertEqual(
            completed.stdout.splitlines(),
            [
                (
                    "Usage: mise run experiment -- --model-type vit "
                    "--model <name> [options]"
                ),
                "",
                "Available models for --model-type vit:",
                "  --model linear",
                "  --model linear_adaptive",
                "  --model expert_linear",
                "  --model expert_linear_adaptive",
            ],
        )

    def test_list_models_for_gpt_model_type_prints_backend_flags(self):
        completed = self.run_experiment("--model-type", "gpt", "--list-models")

        self.assertEqual(completed.returncode, 0)
        self.assertEqual(completed.stderr, "")
        self.assertEqual(
            completed.stdout.splitlines(),
            [
                (
                    "Usage: mise run experiment -- --model-type gpt "
                    "--model <name> [options]"
                ),
                "",
                "Available models for --model-type gpt:",
                "  --model linear",
                "  --model linear_adaptive",
                "  --model expert_linear",
                "  --model expert_linear_adaptive",
            ],
        )

    def test_list_gpt_datasets_uses_causal_language_model_names(self):
        completed = self.run_experiment(
            "--model-type",
            "gpt",
            "--model",
            "linear",
            "--list-datasets",
        )

        self.assertEqual(completed.returncode, 0)
        self.assertEqual(completed.stderr, "")
        self.assertEqual(
            completed.stdout.splitlines(),
            [
                (
                    "Usage: mise run experiment -- --model-type gpt "
                    "--model linear [options]"
                ),
                "",
                "Available datasets for --model-type gpt --model linear:",
                "  --datasets wiki-text2",
                "  --datasets penn-treebank",
            ],
        )

    def test_list_transformer_datasets_prints_both_copyable_directions(self):
        completed = self.run_experiment(
            "--model-type",
            "transformer",
            "--model",
            "linear",
            "--list-datasets",
        )

        self.assertEqual(completed.returncode, 0)
        self.assertEqual(completed.stderr, "")
        self.assertEqual(
            completed.stdout.splitlines(),
            [
                (
                    "Usage: mise run experiment -- --model-type transformer "
                    "--model linear [options]"
                ),
                "",
                ("Available datasets for --model-type transformer --model linear:"),
                "  --datasets multi30k-de-en",
                "  --datasets multi30k-en-de",
            ],
        )

    def test_list_models_for_neuron_model_type_prints_backend_flags(self):
        completed = self.run_experiment("--model-type", "neuron", "--list-models")

        self.assertEqual(completed.returncode, 0)
        self.assertEqual(completed.stderr, "")
        self.assertEqual(
            completed.stdout.splitlines(),
            [
                (
                    "Usage: mise run experiment -- --model-type neuron "
                    "--model <name> [options]"
                ),
                "",
                "Available models for --model-type neuron:",
                "  --model linear",
                "  --model linear_adaptive",
                "  --model expert_linear",
                "  --model expert_linear_adaptive",
            ],
        )

    def test_list_datasets_for_model_prints_copyable_lowercase_dataset_flags(self):
        completed = self.run_experiment(
            "--model-type",
            "linears",
            "--model",
            "linear",
            "--list-datasets",
        )

        self.assertEqual(completed.returncode, 0)
        self.assertEqual(completed.stderr, "")
        self.assertEqual(
            completed.stdout.splitlines(),
            [
                (
                    "Usage: mise run experiment -- --model-type linears "
                    "--model linear [options]"
                ),
                "",
                "Available datasets for --model-type linears --model linear:",
                "  --datasets mnist",
                "  --datasets fashion-mnist",
                "  --datasets cifar10",
                "  --datasets cifar100",
            ],
        )

    def test_list_monitors_for_model_prints_copyable_monitor_flags(self):
        completed = self.run_experiment(
            "--model-type",
            "linears",
            "--model",
            "linear",
            "--list-monitors",
        )

        self.assertEqual(completed.returncode, 0)
        self.assertEqual(completed.stderr, "")
        self.assertEqual(
            completed.stdout.splitlines(),
            [
                (
                    "Usage: mise run experiment -- --model-type linears "
                    "--model linear [options]"
                ),
                "",
                "Available monitors for --model-type linears --model linear:",
                "  --monitors linear",
                "  --monitors recurrent-layer",
                "  --monitors layer-controller",
                "  --monitors halting",
                "  --monitors memory",
            ],
        )

    def test_print_model_rejects_monitors_before_inspection(self):
        completed = self.run_experiment(
            "--model-type",
            "linears",
            "--model",
            "linear",
            "--preset",
            "baseline",
            "--print-model",
            "--monitors",
            "linear",
        )

        self.assertEqual(completed.returncode, 1)
        self.assertEqual(completed.stderr, "")
        self.assertEqual(
            completed.stdout.splitlines(),
            [
                (
                    "Error: --monitors applies to training runs and cannot be used "
                    "with --print-model."
                ),
                "",
                (
                    "Run 'mise run experiment -- --model-type linears --model linear "
                    "--preset <preset> --print-model' without --monitors."
                ),
            ],
        )

    def test_print_model_inspects_selected_model_package(self):
        completed = self.run_experiment(
            "--model-type",
            "linears",
            "--model",
            "linear",
            "--preset",
            "baseline",
            "--print-model",
            "--format",
            "json",
        )

        self.assertEqual(completed.returncode, 0)
        self.assertEqual(completed.stderr, "")
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["modelType"], "linears")
        self.assertEqual(payload["model"], "linear")
        self.assertEqual(payload["preset"], "baseline")
        self.assertGreater(payload["parameterCount"], 0)
        self.assertTrue(payload["nodes"])

    def test_print_model_shapes_annotates_the_existing_tree(self):
        completed = self.run_experiment(
            "--model-type",
            "linears",
            "--model",
            "linear",
            "--preset",
            "baseline",
            "--datasets",
            "mnist",
            "--print-model-shapes",
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stderr, "")
        self.assertIn(
            "shape sample: dataset=Mnist, task=image-classification, "
            "batch=1, mode=eval/no_grad",
            completed.stdout,
        )
        self.assertIn(
            "model: Model {in: X=float32[1,1,28,28] -> out: output=float32[1,10]}",
            completed.stdout,
        )
        self.assertIn("input_model: Layer", completed.stdout)
        self.assertIn(
            "in: state.hidden=float32[1,784] -> out: output.hidden=float32[1,32]",
            completed.stdout,
        )
        self.assertIn("loss_fn: CrossEntropyLoss {not called}", completed.stdout)

    def test_print_model_tensor_shapes_adds_executed_method_locals(self):
        completed = self.run_experiment(
            "--model-type",
            "linears",
            "--model",
            "linear",
            "--preset",
            "baseline",
            "--datasets",
            "mnist",
            "--print-model-tensor-shapes",
            "--config",
            "--stack-num-layers",
            "1",
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stderr, "")
        self.assertIn("tensor variables (executed Python):", completed.stdout)
        self.assertIn(
            "model :: Model.forward (models/linears/linear/model.py:",
            completed.stdout,
        )
        self.assertIn("X=float32[1,1,28,28]", completed.stdout)
        self.assertIn("X=float32[1,784]", completed.stdout)
        self.assertIn("LinearLayer.forward", completed.stdout)
        self.assertNotIn(
            "emperor/development/001_emperor/src/",
            completed.stdout,
        )

    def test_list_models_for_unknown_model_type_fails_with_guidance(self):
        completed = self.run_experiment("--model-type", "missing", "--list-models")

        self.assertEqual(completed.returncode, 1)
        self.assertEqual(completed.stderr, "")
        self.assertEqual(
            completed.stdout.splitlines(),
            [
                "Error: unknown model type '--model-type missing'.",
                "",
                (
                    "Run 'mise run experiment -- --list-model-types' "
                    "to see available model types."
                ),
            ],
        )


if __name__ == "__main__":
    unittest.main()

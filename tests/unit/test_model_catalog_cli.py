import shlex
import subprocess
import sys
import unittest
from pathlib import Path

from emperor.model_packages.identity import is_safe_model_segment, split_model_id
from models.catalog import (
    catalog_entry,
    discover_model_identities_for_type,
    discover_model_types,
    is_safe_model_id,
    model_type_exists,
    public_id_for_flat_name,
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

    def test_legacy_vit_id_and_flat_name_are_unsupported(self):
        self.assertIsNone(catalog_entry("transformer_encoder/" + "vit" + "_linear"))
        self.assertIsNone(public_id_for_flat_name("vit" + "_linear"))
        self.assertEqual(public_id_for_flat_name("linear"), "linears/linear")
        self.assertEqual(
            public_id_for_flat_name("linear_adaptive"),
            "linears/linear_adaptive",
        )
        self.assertEqual(
            public_id_for_flat_name("expert_linear"),
            "bert/expert_linear",
        )
        self.assertEqual(
            public_id_for_flat_name("expert_linear_adaptive"),
            "bert/expert_linear_adaptive",
        )
        removed_neuron_flat_name = "neuron" + "_linear"
        self.assertIsNone(public_id_for_flat_name(removed_neuron_flat_name))


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
        self.assertIn("--model-type linears --model linear", completed.stdout)
        self.assertIn(
            "--model-type linears --model linear_adaptive",
            completed.stdout,
        )
        self.assertIn(
            "--model-type bert --model linear",
            completed.stdout,
        )
        self.assertIn(
            "--model-type bert --model linear_adaptive",
            completed.stdout,
        )
        self.assertIn(
            "--model-type bert --model expert_linear",
            completed.stdout,
        )
        self.assertIn(
            "--model-type bert --model expert_linear_adaptive",
            completed.stdout,
        )
        self.assertIn("--model-type gpt --model linear", completed.stdout)
        self.assertIn(
            "--model-type gpt --model linear_adaptive",
            completed.stdout,
        )
        self.assertIn(
            "--model-type gpt --model expert_linear",
            completed.stdout,
        )
        self.assertIn(
            "--model-type gpt --model expert_linear_adaptive",
            completed.stdout,
        )
        self.assertIn(
            "--model-type vit --model linear",
            completed.stdout,
        )
        self.assertIn(
            "--model-type vit --model linear_adaptive",
            completed.stdout,
        )
        self.assertIn(
            "--model-type vit --model expert_linear",
            completed.stdout,
        )
        self.assertIn(
            "--model-type vit --model expert_linear_adaptive",
            completed.stdout,
        )
        self.assertIn(
            "--model-type transformer --model linear",
            completed.stdout,
        )
        self.assertIn(
            "--model-type transformer --model linear_adaptive",
            completed.stdout,
        )
        self.assertIn(
            "--model-type transformer --model expert_linear",
            completed.stdout,
        )
        self.assertIn(
            "--model-type transformer --model expert_linear_adaptive",
            completed.stdout,
        )
        self.assertIn(
            "--model-type neuron --model linear",
            completed.stdout,
        )
        self.assertIn(
            "--model-type neuron --model linear_adaptive",
            completed.stdout,
        )
        self.assertIn(
            "--model-type neuron --model expert_linear",
            completed.stdout,
        )
        self.assertIn(
            "--model-type neuron --model expert_linear_adaptive",
            completed.stdout,
        )

    def test_list_for_model_type_prints_copyable_model_flags(self):
        completed = self.run_catalog("--list", "--model-type", "linears")

        self.assertEqual(completed.returncode, 0)
        self.assertEqual(completed.stderr, "")
        self.assertEqual(
            completed.stdout.splitlines(),
            [
                "--model linear",
                "--model linear_adaptive",
            ],
        )

    def test_list_for_bert_model_type_prints_canonical_backend_flags(self):
        completed = self.run_catalog("--list", "--model-type", "bert")

        self.assertEqual(completed.returncode, 0)
        self.assertEqual(completed.stderr, "")
        self.assertEqual(
            completed.stdout.splitlines(),
            [
                "--model linear",
                "--model linear_adaptive",
                "--model expert_linear",
                "--model expert_linear_adaptive",
            ],
        )

    def test_list_for_vit_model_type_prints_canonical_backend_flags(self):
        completed = self.run_catalog("--list", "--model-type", "vit")

        self.assertEqual(completed.returncode, 0)
        self.assertEqual(completed.stderr, "")
        self.assertEqual(
            completed.stdout.splitlines(),
            [
                "--model linear",
                "--model linear_adaptive",
                "--model expert_linear",
                "--model expert_linear_adaptive",
            ],
        )

    def test_list_for_gpt_model_type_prints_canonical_backend_flags(self):
        completed = self.run_catalog("--list", "--model-type", "gpt")

        self.assertEqual(completed.returncode, 0)
        self.assertEqual(completed.stderr, "")
        self.assertEqual(
            completed.stdout.splitlines(),
            [
                "--model linear",
                "--model linear_adaptive",
                "--model expert_linear",
                "--model expert_linear_adaptive",
            ],
        )

    def test_list_for_transformer_model_type_prints_canonical_backend_flags(self):
        completed = self.run_catalog("--list", "--model-type", "transformer")

        self.assertEqual(completed.returncode, 0)
        self.assertEqual(completed.stderr, "")
        self.assertEqual(
            completed.stdout.splitlines(),
            [
                "--model linear",
                "--model linear_adaptive",
                "--model expert_linear",
                "--model expert_linear_adaptive",
            ],
        )

    def test_list_for_neuron_model_type_prints_canonical_backend_flags(self):
        completed = self.run_catalog("--list", "--model-type", "neuron")

        self.assertEqual(completed.returncode, 0)
        self.assertEqual(completed.stderr, "")
        self.assertEqual(
            completed.stdout.splitlines(),
            [
                "--model linear",
                "--model linear_adaptive",
                "--model expert_linear",
                "--model expert_linear_adaptive",
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


class TestExperimentShellCatalogCli(unittest.TestCase):
    def run_experiment(self, *args):
        command = "source experiment.sh"
        if args:
            command = f"{command} {shlex.join(args)}"
        return subprocess.run(
            ["bash", "-lc", command],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

    def run_model_command_with_python_stub(self, *args):
        command = "\n".join(
            [
                "source experiment.sh >/dev/null",
                "model_module() { printf '%s\\n' models.fake; }",
                "python3() { printf '%s\\n' \"$@\"; }",
                f"run_model_command linears linear {shlex.join(args)}",
            ]
        )
        return subprocess.run(
            ["bash", "-lc", command],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_list_model_types_prints_copyable_model_type_flags(self):
        completed = self.run_experiment("--list-model-types")

        self.assertEqual(completed.returncode, 0)
        self.assertEqual(completed.stderr, "")
        self.assertEqual(
            completed.stdout.splitlines(),
            [
                (
                    "Usage: source experiment.sh --model-type <type> "
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
                    "Usage: source experiment.sh --model-type linears "
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
                    "Usage: source experiment.sh --model-type bert "
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
                    "Usage: source experiment.sh --model-type vit "
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
                    "Usage: source experiment.sh --model-type gpt "
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
                    "Usage: source experiment.sh --model-type gpt "
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
                    "Usage: source experiment.sh --model-type transformer "
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
                    "Usage: source experiment.sh --model-type neuron "
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
                    "Usage: source experiment.sh --model-type linears "
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
                    "Usage: source experiment.sh --model-type linears "
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
                    "Run 'source experiment.sh --model-type linears --model linear "
                    "--preset <preset> --print-model' without --monitors."
                ),
            ],
        )

    def test_training_command_forwards_monitor_names_to_model_module(self):
        completed = self.run_model_command_with_python_stub(
            "--preset",
            "baseline",
            "--monitors",
            "linear",
            "halting",
        )

        self.assertEqual(completed.returncode, 0)
        self.assertEqual(completed.stderr, "")
        self.assertEqual(
            completed.stdout.splitlines(),
            [
                "-m",
                "models.fake",
                "--preset",
                "baseline",
                "--monitors",
                "linear",
                "halting",
            ],
        )

    def test_training_command_does_not_swallow_config_flags_after_monitors(self):
        completed = self.run_model_command_with_python_stub(
            "--preset",
            "baseline",
            "--monitors",
            "linear",
            "--config",
            "--num-epochs",
            "1",
        )

        self.assertEqual(completed.returncode, 0)
        self.assertEqual(completed.stderr, "")
        self.assertEqual(
            completed.stdout.splitlines(),
            [
                "-m",
                "models.fake",
                "--preset",
                "baseline",
                "--monitors",
                "linear",
                "--config",
                "--num-epochs",
                "1",
            ],
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
                    "Run 'source experiment.sh --list-model-types' "
                    "to see available model types."
                ),
            ],
        )


if __name__ == "__main__":
    unittest.main()

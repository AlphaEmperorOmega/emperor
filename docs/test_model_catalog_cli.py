import shlex
import subprocess
import sys
import unittest
from pathlib import Path

from models.catalog import (
    discover_model_identities_for_type,
    discover_model_types,
    model_type_exists,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class TestModelCatalogDiscovery(unittest.TestCase):
    def test_model_types_are_sorted_and_unique(self):
        model_types = discover_model_types()

        self.assertEqual(
            model_types,
            [
                "experts",
                "linears",
                "neuron",
                "parametric",
                "transformer_encoder",
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


class TestModelCatalogCli(unittest.TestCase):
    def run_catalog(self, *args):
        return subprocess.run(
            [sys.executable, "-m", "models.catalog", *args],
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
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
                "experts",
                "linears",
                "neuron",
                "parametric",
                "transformer_encoder",
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
            "--model-type transformer_encoder --model vit_linear",
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
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

    def test_list_monitors_prints_raw_monitor_names(self):
        completed = self.run_config_overrides("linears/linear", "--monitors")

        self.assertEqual(completed.returncode, 0)
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
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
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
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
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
                "  --model-type experts",
                "  --model-type linears",
                "  --model-type neuron",
                "  --model-type parametric",
                "  --model-type transformer_encoder",
            ],
        )

    def test_list_models_keeps_full_catalog_behavior(self):
        completed = self.run_experiment("--list-models")

        self.assertEqual(completed.returncode, 0)
        self.assertEqual(completed.stderr, "")
        self.assertIn("Available models:", completed.stdout)
        self.assertIn("  --model-type linears --model linear", completed.stdout)
        self.assertIn(
            "  --model-type linears --model linear_adaptive",
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

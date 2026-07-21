import importlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = REPO_ROOT / "src" / "emperor" / "experiments"

ROOT_EXPORTS = (
    "ExperimentTask",
    "experiment_task_label",
    "experiment_task_name",
    "resolve_experiment_task",
)

TASK_EXPORTS = {
    "emperor.experiments.bert_pretraining": "BertPretrainingExperiment",
    "emperor.experiments.classifier": "ClassifierExperiment",
    "emperor.experiments.language_model": "LanguageModelExperiment",
    "emperor.experiments.masked_language_model": "MaskedLanguageModelExperiment",
    "emperor.experiments.sequence_classifier": "SequenceClassifierExperiment",
    "emperor.experiments.translation": "TranslationExperiment",
}

TASK_PRIVATE_EXPORTS = {
    "emperor.experiments.bert_pretraining": (
        "BertPretrainingBatch",
        "BertPretrainingMetricsLogger",
        "BertPretrainingStepOutput",
    ),
    "emperor.experiments.classifier": ("ClassifierMetricsLogger",),
    "emperor.experiments.language_model": (
        "LanguageModelBatch",
        "LanguageModelMetricsLogger",
        "LanguageModelStepOutput",
    ),
    "emperor.experiments.masked_language_model": (
        "MaskedLanguageModelBatch",
        "MaskedLanguageModelMetricsLogger",
        "MaskedLanguageModelStepOutput",
    ),
    "emperor.experiments.sequence_classifier": ("SequenceClassifierMetricsLogger",),
    "emperor.experiments.translation": (
        "TranslationBatch",
        "TranslationStepOutput",
    ),
}

EXPECTED_SOURCE_TREE = (
    "__init__.py",
    "_tasks.py",
    "bert_pretraining/__init__.py",
    "bert_pretraining/_experiment.py",
    "bert_pretraining/_metrics.py",
    "bert_pretraining/_records.py",
    "classifier/__init__.py",
    "classifier/_experiment.py",
    "classifier/_metrics.py",
    "language_model/__init__.py",
    "language_model/_experiment.py",
    "language_model/_metrics.py",
    "language_model/_records.py",
    "masked_language_model/__init__.py",
    "masked_language_model/_experiment.py",
    "masked_language_model/_metrics.py",
    "sequence_classifier/__init__.py",
    "sequence_classifier/_experiment.py",
    "sequence_classifier/_metrics.py",
    "translation/__init__.py",
    "translation/_experiment.py",
    "translation/_metrics.py",
    "translation/_records.py",
)

FORMER_FLAT_FILES = (
    "bert_pretraining.py",
    "classifier.py",
    "language_model.py",
    "masked_language_model.py",
    "sequence_classifier.py",
    "tasks.py",
    "translation.py",
)


class ExperimentInterfaceTests(unittest.TestCase):
    def run_fresh_python(self, source: str) -> dict:
        completed = subprocess.run(
            [sys.executable, "-P", "-c", source],
            cwd=REPO_ROOT,
            capture_output=True,
            env={
                **os.environ,
                "MPLCONFIGDIR": str(Path(tempfile.gettempdir()) / "matplotlib"),
            },
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stderr, "")
        return json.loads(completed.stdout)

    def test_experiment_source_tree_is_exact(self) -> None:
        actual = tuple(
            path.relative_to(EXPERIMENTS_ROOT).as_posix()
            for path in sorted(EXPERIMENTS_ROOT.rglob("*.py"))
        )

        self.assertEqual(actual, EXPECTED_SOURCE_TREE)
        self.assertEqual(len(actual), 23)

    def test_former_flat_implementation_files_are_absent(self) -> None:
        for relative_path in FORMER_FLAT_FILES:
            with self.subTest(path=relative_path):
                self.assertFalse((EXPERIMENTS_ROOT / relative_path).exists())

    def test_root_and_task_exports_are_exact(self) -> None:
        root = importlib.import_module("emperor.experiments")
        self.assertEqual(tuple(root.__all__), ROOT_EXPORTS)

        for module_name, export_name in TASK_EXPORTS.items():
            with self.subTest(module=module_name):
                module = importlib.import_module(module_name)
                self.assertEqual(tuple(module.__all__), (export_name,))

    def test_non_interface_task_details_are_not_exported(self) -> None:
        root = importlib.import_module("emperor.experiments")
        for export_name in TASK_EXPORTS.values():
            with self.subTest(module=root.__name__, name=export_name):
                with self.assertRaises(AttributeError):
                    getattr(root, export_name)

        for module_name, private_names in TASK_PRIVATE_EXPORTS.items():
            module = importlib.import_module(module_name)
            for private_name in private_names:
                with self.subTest(module=module_name, name=private_name):
                    with self.assertRaises(AttributeError):
                        getattr(module, private_name)

    def test_task_facades_are_lazy_lightweight_and_resolve_exact_owners(self) -> None:
        for module_name, export_name in TASK_EXPORTS.items():
            owner_name = f"{module_name}._experiment"
            with self.subTest(module=module_name):
                result = self.run_fresh_python(
                    f"""
import importlib
import json
import sys

owner_name = {owner_name!r}
owner_loaded_before = owner_name in sys.modules
module = importlib.import_module({module_name!r})
owner_loaded_after_facade = owner_name in sys.modules
heavy_after_facade = {{
    name: name in sys.modules
    for name in ("lightning", "torch", "torchmetrics")
}}
export = getattr(module, {export_name!r})
try:
    getattr(module, "__emperor_undeclared_task_export__")
except AttributeError:
    rejects_undeclared = True
else:
    rejects_undeclared = False
print(json.dumps({{
    "all": module.__all__,
    "export_owner": export.__module__,
    "heavy_after_facade": heavy_after_facade,
    "owner_loaded_after_access": owner_name in sys.modules,
    "owner_loaded_after_facade": owner_loaded_after_facade,
    "owner_loaded_before": owner_loaded_before,
    "rejects_undeclared": rejects_undeclared,
}}))
"""
                )

                self.assertEqual(result["all"], [export_name])
                self.assertEqual(result["export_owner"], owner_name)
                self.assertEqual(
                    result["heavy_after_facade"],
                    {
                        "lightning": False,
                        "torch": False,
                        "torchmetrics": False,
                    },
                )
                self.assertFalse(result["owner_loaded_before"])
                self.assertFalse(result["owner_loaded_after_facade"])
                self.assertTrue(result["owner_loaded_after_access"])
                self.assertTrue(result["rejects_undeclared"])


if __name__ == "__main__":
    unittest.main()

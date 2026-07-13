import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


class TestFoundationalImports(unittest.TestCase):
    def run_fresh_python(self, source: str) -> dict:
        completed = subprocess.run(
            [sys.executable, "-c", source],
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

    def test_config_import_defers_plotting_and_notebook_dependencies(self):
        result = self.run_fresh_python(
            """
import json
import sys

import emperor.config

print(json.dumps({
    "ipython_display": "IPython.display" in sys.modules,
    "matplotlib_pyplot": "matplotlib.pyplot" in sys.modules,
}))
"""
        )

        self.assertEqual(
            result,
            {
                "ipython_display": False,
                "matplotlib_pyplot": False,
            },
        )

    def test_foundational_model_validation_import_defers_runtime_dependencies(
        self,
    ):
        result = self.run_fresh_python(
            """
import json
import sys

import emperor.base.validator

print(json.dumps({
    "ipython_display": "IPython.display" in sys.modules,
    "lightning": "lightning" in sys.modules,
    "matplotlib_pyplot": "matplotlib.pyplot" in sys.modules,
}))
"""
        )

        self.assertEqual(
            result,
            {
                "ipython_display": False,
                "lightning": False,
                "matplotlib_pyplot": False,
            },
        )

    def test_experiment_task_import_defers_concrete_experiment_modules(self):
        result = self.run_fresh_python(
            """
import json
import sys

from emperor.experiments.tasks import ExperimentTask

print(json.dumps({
    "language_model": "emperor.experiments.language_model" in sys.modules,
    "task": ExperimentTask.TEXT_TRANSLATION.name,
    "torch": "torch" in sys.modules,
    "torchmetrics": "torchmetrics" in sys.modules,
    "translation": "emperor.experiments.translation" in sys.modules,
}))
"""
        )

        self.assertEqual(
            result,
            {
                "language_model": False,
                "task": "TEXT_TRANSLATION",
                "torch": False,
                "torchmetrics": False,
                "translation": False,
            },
        )

    def test_package_level_experiment_exports_remain_compatible(self):
        result = self.run_fresh_python(
            """
import json

import emperor.experiments as experiments
from emperor.experiments import (
    ExperimentTask,
    LanguageModelExperiment,
    LanguageModelStepOutput,
    TranslationExperiment,
    TranslationStepOutput,
)

print(json.dumps({
    "all": experiments.__all__,
    "experiment_task": ExperimentTask.__module__,
    "language_model": LanguageModelExperiment.__module__,
    "language_model_output": LanguageModelStepOutput.__module__,
    "translation": TranslationExperiment.__module__,
    "translation_output": TranslationStepOutput.__module__,
}))
"""
        )

        self.assertEqual(
            result,
            {
                "all": [
                    "ExperimentTask",
                    "LanguageModelExperiment",
                    "LanguageModelStepOutput",
                    "TranslationExperiment",
                    "TranslationStepOutput",
                ],
                "experiment_task": "emperor.experiments.tasks",
                "language_model": "emperor.experiments.language_model",
                "language_model_output": "emperor.experiments.language_model",
                "translation": "emperor.experiments.translation",
                "translation_output": "emperor.experiments.translation",
            },
        )

    def test_legacy_base_utils_imports_preserve_canonical_identity(self):
        result = self.run_fresh_python(
            """
import json

from emperor.base.config import ConfigBase as CanonicalConfigBase
from emperor.base.config import optional_field as canonical_optional_field
from emperor.base.data import DataModule as CanonicalDataModule
from emperor.base.module import Module as CanonicalModule
from emperor.base.module import ParameterBank as CanonicalParameterBank
from emperor.base.visualization import ProgressBoard as CanonicalProgressBoard
from emperor.base.visualization import show_images as canonical_show_images
from emperor.base.utils import (
    ConfigBase as LegacyConfigBase,
    DataModule as LegacyDataModule,
    Module as LegacyModule,
    ParameterBank as LegacyParameterBank,
    ProgressBoard as LegacyProgressBoard,
    optional_field as legacy_optional_field,
    show_images as legacy_show_images,
)

print(json.dumps({
    "config_base": CanonicalConfigBase is LegacyConfigBase,
    "data_module": CanonicalDataModule is LegacyDataModule,
    "module": CanonicalModule is LegacyModule,
    "optional_field": canonical_optional_field is legacy_optional_field,
    "parameter_bank": CanonicalParameterBank is LegacyParameterBank,
    "progress_board": CanonicalProgressBoard is LegacyProgressBoard,
    "show_images": canonical_show_images is legacy_show_images,
}))
"""
        )

        self.assertEqual(
            result,
            {
                "config_base": True,
                "data_module": True,
                "module": True,
                "optional_field": True,
                "parameter_bank": True,
                "progress_board": True,
                "show_images": True,
            },
        )

    def test_visualization_owner_defers_plotting_and_notebook_dependencies(self):
        result = self.run_fresh_python(
            """
import json
import sys

import emperor.base.visualization

print(json.dumps({
    "ipython_display": "IPython.display" in sys.modules,
    "matplotlib_pyplot": "matplotlib.pyplot" in sys.modules,
}))
"""
        )

        self.assertEqual(
            result,
            {
                "ipython_display": False,
                "matplotlib_pyplot": False,
            },
        )

    def test_legacy_base_options_import_preserves_canonical_class_identity(self):
        result = self.run_fresh_python(
            """
import json

from emperor.base.option import BaseOptions as CanonicalBaseOptions
from emperor.base.options import BaseOptions as LegacyBaseOptions

print(json.dumps({
    "same_class": CanonicalBaseOptions is LegacyBaseOptions,
}))
"""
        )

        self.assertEqual(result, {"same_class": True})

    def test_monitor_metadata_import_defers_lightning(self):
        result = self.run_fresh_python(
            """
import json
import sys

from emperor.experiments.monitors import MonitorOption, MonitorSettings

print(json.dumps({
    "lightning": "lightning" in sys.modules,
    "option": MonitorOption.__name__,
    "settings": MonitorSettings.__name__,
}))
"""
        )

        self.assertEqual(
            result,
            {
                "lightning": False,
                "option": "MonitorOption",
                "settings": "MonitorSettings",
            },
        )


if __name__ == "__main__":
    unittest.main()

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

import emperor._validation

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

from emperor.experiments import ExperimentTask

print(json.dumps({
    "bert_pretraining": (
        "emperor.experiments.bert_pretraining._experiment" in sys.modules
    ),
    "classifier": "emperor.experiments.classifier._experiment" in sys.modules,
    "language_model": (
        "emperor.experiments.language_model._experiment" in sys.modules
    ),
    "masked_language_model": (
        "emperor.experiments.masked_language_model._experiment" in sys.modules
    ),
    "sequence_classifier": (
        "emperor.experiments.sequence_classifier._experiment" in sys.modules
    ),
    "task": ExperimentTask.TEXT_TRANSLATION.name,
    "torch": "torch" in sys.modules,
    "torchmetrics": "torchmetrics" in sys.modules,
    "translation": "emperor.experiments.translation._experiment" in sys.modules,
}))
"""
        )

        self.assertEqual(
            result,
            {
                "bert_pretraining": False,
                "classifier": False,
                "language_model": False,
                "masked_language_model": False,
                "sequence_classifier": False,
                "task": "TEXT_TRANSLATION",
                "torch": False,
                "torchmetrics": False,
                "translation": False,
            },
        )

    def test_experiment_task_interface_is_exact_and_lightweight(self):
        result = self.run_fresh_python(
            """
import json

import emperor.experiments as experiments
from emperor.experiments import (
    ExperimentTask,
    experiment_task_label,
    experiment_task_name,
    resolve_experiment_task,
)

print(json.dumps({
    "all": experiments.__all__,
    "experiment_task": ExperimentTask.__module__,
    "experiment_task_label": experiment_task_label.__module__,
    "experiment_task_name": experiment_task_name.__module__,
    "resolve_experiment_task": resolve_experiment_task.__module__,
}))
"""
        )

        self.assertEqual(
            result,
            {
                "all": [
                    "ExperimentTask",
                    "experiment_task_label",
                    "experiment_task_name",
                    "resolve_experiment_task",
                ],
                "experiment_task": "emperor.experiments._tasks",
                "experiment_task_label": "emperor.experiments._tasks",
                "experiment_task_name": "emperor.experiments._tasks",
                "resolve_experiment_task": "emperor.experiments._tasks",
            },
        )

    def test_nn_interface_is_curated_and_defers_plotting_dependencies(self):
        result = self.run_fresh_python(
            """
import json
import sys

import emperor.nn as nn

print(json.dumps({
    "all": nn.__all__,
    "ipython_display": "IPython.display" in sys.modules,
    "matplotlib_pyplot": "matplotlib.pyplot" in sys.modules,
    "module_owner": nn.Module.__module__,
    "parameter_bank_exported": hasattr(nn, "ParameterBank"),
}))
"""
        )

        self.assertEqual(
            result,
            {
                "all": ["Module"],
                "ipython_display": False,
                "matplotlib_pyplot": False,
                "module_owner": "emperor.nn._module",
                "parameter_bank_exported": False,
            },
        )

    def test_layers_interface_is_curated_and_lazy(self):
        result = self.run_fresh_python(
            """
import json
import sys

import emperor.layers as layers

print(json.dumps({
    "all": layers.__all__,
    "layer_gate_exported": hasattr(layers, "LayerGate"),
    "lightning": "lightning" in sys.modules,
    "matplotlib_pyplot": "matplotlib.pyplot" in sys.modules,
    "torch": "torch" in sys.modules,
    "validator_exported": hasattr(layers, "LayerValidator"),
}))
"""
        )

        self.assertEqual(
            result,
            {
                "all": [
                    "ActivationOptions",
                    "GateConfig",
                    "LastLayerBiasOptions",
                    "LayerConfig",
                    "LayerGateOptions",
                    "LayerNormPositionOptions",
                    "LayerStackConfig",
                    "RecurrentLayerConfig",
                    "ResidualConfig",
                    "ResidualConnectionOptions",
                    "LayerState",
                    "ResidualConnection",
                    "Layer",
                    "LayerStack",
                    "RecurrentLayer",
                    "LayerControllerMonitorCallback",
                    "RecurrentLayerMonitorCallback",
                ],
                "layer_gate_exported": False,
                "lightning": False,
                "matplotlib_pyplot": False,
                "torch": False,
                "validator_exported": False,
            },
        )

    def test_monitoring_interface_is_curated_and_metadata_stays_lightweight(self):
        result = self.run_fresh_python(
            """
import json
import sys

import emperor.monitoring as monitoring
from emperor.monitoring import MonitorOption, MonitorSettings

print(json.dumps({
    "all": monitoring.__all__,
    "lightning": "lightning" in sys.modules,
    "option_owner": MonitorOption.__module__,
    "private_metadata_exported": hasattr(monitoring, "MonitorKind"),
    "settings_owner": MonitorSettings.__module__,
    "torch": "torch" in sys.modules,
}))
"""
        )

        self.assertEqual(
            result,
            {
                "all": [
                    "MonitorOption",
                    "MonitorSettings",
                    "MonitorEmissionPolicy",
                    "MonitorTensorHistory",
                ],
                "lightning": False,
                "option_owner": "emperor.monitoring._metadata",
                "private_metadata_exported": False,
                "settings_owner": "emperor.monitoring._metadata",
                "torch": False,
            },
        )

    def test_lazy_foundation_interfaces_reject_unknown_attributes_exactly(self):
        import emperor.monitoring as monitoring
        import emperor.nn as nn

        with self.assertRaises(AttributeError) as nn_error:
            _ = nn.MissingNeuralFoundation
        self.assertEqual(
            str(nn_error.exception),
            "module 'emperor.nn' has no attribute 'MissingNeuralFoundation'",
        )

        with self.assertRaises(AttributeError) as monitoring_error:
            _ = monitoring.MissingMonitorFoundation
        self.assertEqual(
            str(monitoring_error.exception),
            "module 'emperor.monitoring' has no attribute 'MissingMonitorFoundation'",
        )


if __name__ == "__main__":
    unittest.main()

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

EXPECTED_EXPORTS = (
    "MixtureOfExpertsConfig",
    "MixtureOfExpertsLayerConfig",
    "MixtureOfExpertsModelConfig",
    "DroppedTokenOptions",
    "ExpertWeightingPositionOptions",
    "RoutingInitializationMode",
    "MixtureOfExpertsLayerState",
)

EXPECTED_OWNERS = {
    "MixtureOfExpertsConfig": "emperor.experts._config",
    "MixtureOfExpertsLayerConfig": "emperor.experts._config",
    "MixtureOfExpertsModelConfig": "emperor.experts._config",
    "DroppedTokenOptions": "emperor.experts._options",
    "ExpertWeightingPositionOptions": "emperor.experts._options",
    "RoutingInitializationMode": "emperor.experts._options",
    "MixtureOfExpertsLayerState": "emperor.experts._state",
}


class TestExpertsPublicInterface(unittest.TestCase):
    def test_exact_config_driven_exports_use_an_ordinary_interface(self):
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                """
import json
import sys

import emperor.experts as experts

expected_eager_modules = (
    "emperor.experts._config",
    "emperor.experts._options",
    "emperor.experts._state",
)
heavy_modules = (
    "emperor.experts._model",
    "emperor.experts._validation.mixture",
    "emperor.experts._validation.model",
    "emperor.experts._routing.capacity",
    "emperor.experts._routing.weighting",
    "emperor.experts._layers.mixture",
    "emperor.experts._layers.layer",
    "emperor.experts._layers.map",
    "emperor.experts._layers.reduce",
)
runtime_loaded = {
    "lightning": "lightning" in sys.modules,
}
owners = {name: getattr(experts, name).__module__ for name in experts.__all__}

print(json.dumps({
    "all": experts.__all__,
    "expected_eager_modules": {
        name: name in sys.modules for name in expected_eager_modules
    },
    "heavy_modules": {name: name in sys.modules for name in heavy_modules},
    "owners": owners,
    "private_exports": {
        name: hasattr(experts, name)
        for name in (
            "ExpertCapacityHandler",
            "ExpertInputData",
            "ExpertWeightingHandler",
            "MixtureOfExperts",
            "MixtureOfExpertsLayer",
            "MixtureOfExpertsMap",
            "MixtureOfExpertsModel",
            "MixtureOfExpertsModelValidator",
            "MixtureOfExpertsReduce",
            "MixtureOfExpertsValidator",
        )
    },
    "runtime_loaded": runtime_loaded,
    "shortcut_attributes": {
        "__getattr__": hasattr(experts, "__getattr__"),
        "_LAZY_EXPORTS": hasattr(experts, "_LAZY_EXPORTS"),
    },
}))
""",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            env={
                **os.environ,
                "MPLCONFIGDIR": str(
                    Path(tempfile.gettempdir()) / "matplotlib-experts-interface"
                ),
            },
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(completed.stdout)

        self.assertEqual(tuple(result["all"]), EXPECTED_EXPORTS)
        self.assertEqual(result["owners"], EXPECTED_OWNERS)
        self.assertEqual(
            result["expected_eager_modules"],
            dict.fromkeys(result["expected_eager_modules"], True),
        )
        self.assertEqual(
            result["heavy_modules"],
            dict.fromkeys(result["heavy_modules"], False),
        )
        self.assertEqual(
            result["private_exports"],
            dict.fromkeys(result["private_exports"], False),
        )
        self.assertEqual(
            result["runtime_loaded"],
            {"lightning": False},
        )
        self.assertEqual(
            result["shortcut_attributes"],
            {"__getattr__": False, "_LAZY_EXPORTS": False},
        )

    def test_removed_concrete_implementation_imports_fail(self):
        for implementation_name in (
            "MixtureOfExperts",
            "MixtureOfExpertsMap",
            "MixtureOfExpertsModel",
            "MixtureOfExpertsReduce",
        ):
            with self.subTest(implementation=implementation_name):
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        f"from emperor.experts import {implementation_name}",
                    ],
                    cwd=REPO_ROOT,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn("ImportError", completed.stderr)

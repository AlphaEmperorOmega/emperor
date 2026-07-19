from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"


class TestSourceModelShapeInspectionCli(unittest.TestCase):
    def test_print_model_shapes_annotates_the_model_tree(self) -> None:
        environment = os.environ.copy()
        with tempfile.TemporaryDirectory() as matplotlib_config_dir:
            environment["MPLCONFIGDIR"] = matplotlib_config_dir
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "models.project_cli",
                    "--model-type",
                    "linears",
                    "--model",
                    "linear",
                    "--preset",
                    "baseline",
                    "--datasets",
                    "mnist",
                    "--print-model-shapes",
                ],
                cwd=SOURCE_ROOT,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stderr, "")
        self.assertIn(
            "shape sample: dataset=Mnist, task=image-classification, "
            "batch=1, mode=eval/no_grad",
            completed.stdout,
        )
        self.assertIn(
            "model: Model {in: X=float32[1,1,28,28] -> "
            "out: output=float32[1,10]}",
            completed.stdout,
        )
        self.assertIn("loss_fn: CrossEntropyLoss {not called}", completed.stdout)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from models.inspection_cli import _print_tree

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"


class TestSourceModelShapeInspectionCli(unittest.TestCase):
    def test_text_tree_expands_shared_dag_nodes_only_once(self) -> None:
        layer_count = 17
        nodes = [{"id": "root", "path": "model", "typeName": "Root", "details": {}}]
        edges: list[dict[str, str]] = []
        previous_ids = ["root"]
        for layer_index in range(layer_count):
            layer_ids = [f"layer_{layer_index}_{side}" for side in ("left", "right")]
            nodes.extend(
                {
                    "id": node_id,
                    "path": node_id,
                    "typeName": "Shared",
                    "details": {},
                }
                for node_id in layer_ids
            )
            edges.extend(
                {"source": parent_id, "target": child_id}
                for parent_id in previous_ids
                for child_id in layer_ids
            )
            previous_ids = layer_ids

        with patch("builtins.print") as output:
            _print_tree({"nodes": nodes, "edges": edges})

        rendered_lines = [str(call.args[0]) for call in output.call_args_list]
        self.assertEqual(len(rendered_lines), len(edges) + 1)
        self.assertTrue(any("[reference]" in line for line in rendered_lines))

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
            "model: Model {in: X=float32[1,1,28,28] -> out: output=float32[1,10]}",
            completed.stdout,
        )
        self.assertIn("loss_fn: CrossEntropyLoss {not called}", completed.stdout)


if __name__ == "__main__":
    unittest.main()

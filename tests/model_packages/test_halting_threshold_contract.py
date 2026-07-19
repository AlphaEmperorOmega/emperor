import ast
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MODEL_SOURCE_ROOT = REPOSITORY_ROOT / "src" / "models"
DEFAULT_HALTING_THRESHOLD = 0.999
DEFAULT_THRESHOLD_NAMES = {
    "HALTING_THRESHOLD",
    "ATTN_HALTING_THRESHOLD",
    "ATTN_RECURRENT_HALTING_THRESHOLD",
    "FF_HALTING_THRESHOLD",
    "FF_RECURRENT_HALTING_THRESHOLD",
}


class TestModelHaltingThresholdContract(unittest.TestCase):
    def test_literal_model_halting_thresholds_match_strategy_default(self) -> None:
        literal_thresholds: list[tuple[Path, int, float]] = []
        for path in sorted(MODEL_SOURCE_ROOT.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                    continue
                target = self._assignment_target(node)
                value = node.value
                if (
                    target is None
                    or target not in DEFAULT_THRESHOLD_NAMES
                    or not isinstance(value, ast.Constant)
                    or not isinstance(value.value, float)
                ):
                    continue
                literal_thresholds.append((path, node.lineno, value.value))

        self.assertTrue(literal_thresholds)
        mismatches = [
            f"{path.relative_to(REPOSITORY_ROOT)}:{line_number}={threshold}"
            for path, line_number, threshold in literal_thresholds
            if threshold != DEFAULT_HALTING_THRESHOLD
        ]
        self.assertEqual(mismatches, [])

    @staticmethod
    def _assignment_target(node: ast.Assign | ast.AnnAssign) -> str | None:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            return node.target.id
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            return node.targets[0].id
        return None


if __name__ == "__main__":
    unittest.main()

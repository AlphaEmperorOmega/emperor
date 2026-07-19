import ast
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MODEL_SOURCE_ROOT = REPOSITORY_ROOT / "src" / "models"
ATTENTION_CONFIG_FACTORIES = {
    "SelfAttentionConfig",
    "MixtureOfAttentionHeadsConfig",
}


class TestModelAttentionLayoutContract(unittest.TestCase):
    def test_every_model_attention_factory_declares_batch_first_true(self) -> None:
        attention_calls: list[tuple[Path, int, ast.Call]] = []
        for path in sorted(MODEL_SOURCE_ROOT.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                function_name = self._function_name(node.func)
                if function_name in ATTENTION_CONFIG_FACTORIES:
                    attention_calls.append((path, node.lineno, node))

        self.assertTrue(attention_calls)
        missing_or_invalid: list[str] = []
        for path, line_number, call in attention_calls:
            keyword = next(
                (item for item in call.keywords if item.arg == "batch_first_flag"),
                None,
            )
            if (
                keyword is None
                or not isinstance(keyword.value, ast.Constant)
                or keyword.value.value is not True
            ):
                relative_path = path.relative_to(REPOSITORY_ROOT)
                missing_or_invalid.append(f"{relative_path}:{line_number}")

        self.assertEqual(
            missing_or_invalid,
            [],
            "Model attention factories must explicitly set batch_first_flag=True: "
            + ", ".join(missing_or_invalid),
        )

    @staticmethod
    def _function_name(function: ast.expr) -> str | None:
        if isinstance(function, ast.Name):
            return function.id
        if isinstance(function, ast.Attribute):
            return function.attr
        return None


if __name__ == "__main__":
    unittest.main()

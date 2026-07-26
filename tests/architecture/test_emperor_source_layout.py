import ast
import re
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
EMPEROR_SOURCE = SOURCE_ROOT / "emperor"


def module_name(path: Path) -> str:
    relative_path = path.relative_to(SOURCE_ROOT)
    module_parts = list(relative_path.parts)
    if module_parts[-1] == "__init__.py":
        module_parts = module_parts[:-1]
    else:
        module_parts[-1] = path.stem
    return ".".join(module_parts)


def emperor_modules() -> set[str]:
    return {module_name(path) for path in EMPEROR_SOURCE.rglob("*.py")}


def parsed_source_files():
    for path in SOURCE_ROOT.rglob("*.py"):
        yield path, ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


class EmperorSourceLayoutTests(unittest.TestCase):
    def test_emperor_source_contains_no_symlink_bridge(self):
        symlinks = [
            path.relative_to(REPOSITORY_ROOT).as_posix()
            for path in EMPEROR_SOURCE.rglob("*")
            if path.is_symlink()
        ]

        self.assertEqual(symlinks, [])

    def test_production_imports_reference_physical_emperor_modules(self):
        available_modules = emperor_modules()
        missing_modules = []

        for path, syntax_tree in parsed_source_files():
            for node in ast.walk(syntax_tree):
                referenced_modules = []
                if isinstance(node, ast.Import):
                    referenced_modules.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    referenced_modules.append(node.module)

                for referenced_module in referenced_modules:
                    if not (
                        referenced_module == "emperor"
                        or referenced_module.startswith("emperor.")
                    ):
                        continue
                    if referenced_module not in available_modules:
                        missing_modules.append(
                            (
                                path.relative_to(REPOSITORY_ROOT).as_posix(),
                                node.lineno,
                                referenced_module,
                            )
                        )

        self.assertEqual(missing_modules, [])

    def test_emperor_runtime_contains_no_import_redirection_hook(self):
        forbidden_tokens = (
            "sys.path",
            "sys.modules",
            "sys.meta_path",
            "spec_from_file_location",
            "SourceFileLoader",
            "MetaPathFinder",
            "PathFinder",
            "__path__",
        )
        matches = []

        for path in EMPEROR_SOURCE.rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            for token in forbidden_tokens:
                if token in source:
                    matches.append(
                        (path.relative_to(REPOSITORY_ROOT).as_posix(), token)
                    )

        self.assertEqual(matches, [])

    def test_source_strings_do_not_preserve_retired_emperor_paths(self):
        available_modules = emperor_modules()
        retired_references = []
        module_reference = re.compile(
            r"^emperor(?:\.[A-Za-z_][A-Za-z0-9_]*)*"
        )
        legacy_path_reference = re.compile(r"(?<!src/)emperor/(?:$|[A-Za-z_])")

        for path, syntax_tree in parsed_source_files():
            for node in ast.walk(syntax_tree):
                if not isinstance(node, ast.Constant) or not isinstance(
                    node.value, str
                ):
                    continue

                value = node.value
                if legacy_path_reference.search(value):
                    retired_references.append(
                        (
                            path.relative_to(REPOSITORY_ROOT).as_posix(),
                            node.lineno,
                            value,
                        )
                    )
                    continue

                match = module_reference.match(value)
                if match is None:
                    continue
                module_parts = match.group(0).split(".")
                current_module_parts = [module_parts[0]]
                for part in module_parts[1:]:
                    if part[0].isupper():
                        break
                    current_module_parts.append(part)
                referenced_module = ".".join(current_module_parts)
                if referenced_module not in available_modules:
                    retired_references.append(
                        (
                            path.relative_to(REPOSITORY_ROOT).as_posix(),
                            node.lineno,
                            value,
                        )
                    )

        self.assertEqual(retired_references, [])

    def test_residual_composition_has_no_retired_dispatch_or_config_shape(self):
        retired_names = {
            "AttentionResidualOption",
            "PAIRWISE_RESIDUAL_TYPES",
            "RESIDUAL_OPTION_TYPES",
            "ResidualConnection",
            "ResidualConnectionOptions",
            "_PairwiseResidualParameters",
        }
        retired_residual_config_fields = {
            "attention_config",
            "model_config",
            "option",
        }
        violations = []

        for path, syntax_tree in parsed_source_files():
            relative_path = path.relative_to(REPOSITORY_ROOT).as_posix()
            for node in ast.walk(syntax_tree):
                if isinstance(node, ast.Name) and node.id in retired_names:
                    violations.append((relative_path, node.lineno, node.id))
                    continue
                if isinstance(node, ast.Attribute):
                    if node.attr in retired_names or node.attr == "attention_residual":
                        violations.append((relative_path, node.lineno, node.attr))
                    continue
                if not isinstance(node, ast.Call):
                    continue
                function_name = (
                    node.func.id
                    if isinstance(node.func, ast.Name)
                    else node.func.attr
                    if isinstance(node.func, ast.Attribute)
                    else None
                )
                if function_name != "ResidualConfig":
                    continue
                for keyword in node.keywords:
                    if keyword.arg in retired_residual_config_fields:
                        violations.append(
                            (
                                relative_path,
                                node.lineno,
                                f"ResidualConfig({keyword.arg}=...)",
                            )
                        )

        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()

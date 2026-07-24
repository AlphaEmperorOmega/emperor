from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_OWNED_DIRECTORY_ROOTS = (
    Path("src"),
    Path("tests"),
    Path("tools"),
    Path("docs"),
    Path("apps/workbench/api/src"),
    Path("apps/workbench/api/tests"),
    Path("apps/workbench/web/src"),
    Path("apps/workbench/web/tests"),
)
_OWNED_ROOT_FILES = (
    Path("CONTEXT.md"),
    Path("MANIFEST.in"),
    Path("README.md"),
    Path("env.ps1"),
    Path("mise.toml"),
    Path("pyproject.toml"),
    Path("apps/workbench/README.md"),
    Path("apps/workbench/api/pyproject.toml"),
    Path("apps/workbench/web/eslint.config.mjs"),
    Path("apps/workbench/web/next.config.ts"),
    Path("apps/workbench/web/package.json"),
    Path("apps/workbench/web/tsconfig.json"),
)
_TEXT_SUFFIXES = {
    ".in",
    ".js",
    ".json",
    ".md",
    ".mjs",
    ".ps1",
    ".py",
    ".pyi",
    ".sh",
    ".toml",
    ".ts",
    ".tsx",
}
_EXCLUDED_PARTS = {
    ".git",
    ".mypy_cache",
    ".next",
    ".pytest_cache",
    ".ruff_cache",
    ".scratch",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
}


def _owned_text_files() -> tuple[Path, ...]:
    files = {
        PROJECT_ROOT / path
        for path in _OWNED_ROOT_FILES
        if (PROJECT_ROOT / path).is_file()
    }
    for relative_root in _OWNED_DIRECTORY_ROOTS:
        root = PROJECT_ROOT / relative_root
        if not root.exists():
            continue
        files.update(
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.suffix in _TEXT_SUFFIXES
            and not (_EXCLUDED_PARTS & set(path.relative_to(PROJECT_ROOT).parts))
        )
    return tuple(sorted(files))


def _relative(path: Path) -> Path:
    return path.relative_to(PROJECT_ROOT)


def _implementation_files() -> tuple[Path, ...]:
    roots = (
        PROJECT_ROOT / "src",
        PROJECT_ROOT / "apps/workbench/api/src",
        PROJECT_ROOT / "apps/workbench/web/src",
    )
    return tuple(
        path
        for path in _owned_text_files()
        if any(path.is_relative_to(root) for root in roots)
        and ".test." not in path.name
    )


class CanonicalCutoverScannerTests(unittest.TestCase):
    def test_retired_artifact_paths_are_absent(self) -> None:
        retired_paths = (
            Path("env" + ".sh"),
            Path("run" + "_test.sh"),
            Path("download" + "_logs.sh"),
            Path("experiment" + ".sh"),
            Path("src/models") / ("model" + "_metadata.py"),
            Path("src/models") / ("dataset" + "_naming.py"),
            Path("src/models") / ("experiment" + "_mode.py"),
            Path("src/models") / ("log" + "_migration.py"),
            Path("src/emperor/neuron") / ("_optimizer" + "_checkpoint.py"),
        )

        self.assertEqual(
            [str(path) for path in retired_paths if (PROJECT_ROOT / path).exists()],
            [],
        )

    def test_owned_text_does_not_advertise_retired_interfaces(self) -> None:
        forbidden_fragments = (
            "logs" + "-archive",
            "Model" + " Visualizer",
            "env" + ".sh",
            "run" + "_test.sh",
            "download" + "_logs.sh",
            "experiment" + ".sh",
        )
        findings: list[str] = []
        for path in _owned_text_files():
            source = path.read_text(encoding="utf-8")
            for fragment in forbidden_fragments:
                if fragment in source:
                    findings.append(f"{_relative(path)}: {fragment}")

        self.assertEqual(findings, [])

    def test_implementation_text_uses_canonical_routes_and_monitor_paths(self) -> None:
        retired_route = "/models/" + "linear/"
        retired_monitor_path = re.compile(r"main_model\.(?:linears\.)?\d")
        retired_symbols = (
            "Model" + "CatalogEntry",
            "FLAT" + "_TO_PUBLIC_ID",
            "checkpoint" + "_metadata_module",
            "public" + "_id_for_flat_name",
            "model_package" + "_from_module_path",
            "model_package" + "_for_module",
            "maximum" + "_dimension",
            "legacy" + "_args",
            "legacy" + "_options",
        )
        findings: list[str] = []
        for path in _implementation_files():
            source = path.read_text(encoding="utf-8")
            if retired_route in source:
                findings.append(f"{_relative(path)}: flat model route")
            if retired_monitor_path.search(source):
                findings.append(f"{_relative(path)}: monitor path alias")
            for symbol in retired_symbols:
                if re.search(rf"\b{re.escape(symbol)}\b", source):
                    findings.append(f"{_relative(path)}: {symbol}")

        self.assertEqual(findings, [])

    def test_python_imports_do_not_target_retired_facades(self) -> None:
        retired_modules = {
            "models." + "config_ast_listing",
            "models." + "config_value_parser",
            "models." + "dataset_naming",
            "models." + "experiment_mode",
            "models." + "log_migration",
            "models." + "model_metadata",
            "models." + "parser",
        }
        findings: list[str] = []
        for path in _owned_text_files():
            if path.suffix != ".py":
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                imported: tuple[str, ...] = ()
                if isinstance(node, ast.Import):
                    imported = tuple(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module is not None:
                    imported = (node.module,)
                for module in imported:
                    if any(
                        module == retired or module.startswith(retired + ".")
                        for retired in retired_modules
                    ):
                        findings.append(f"{_relative(path)}:{node.lineno}: {module}")

        self.assertEqual(findings, [])


if __name__ == "__main__":
    unittest.main()

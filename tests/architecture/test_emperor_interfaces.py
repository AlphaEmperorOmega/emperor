from __future__ import annotations

import ast
import re
import tomllib
import unittest
import zipfile
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EMPEROR_ROOT = PROJECT_ROOT / "src" / "emperor"
MANIFEST_PATH = Path(__file__).with_name("emperor_interfaces.toml")


@dataclass(frozen=True, order=True)
class ImportReference:
    consumer: str
    module: str
    names: tuple[str, ...]


def _load_manifest() -> dict[str, object]:
    with MANIFEST_PATH.open("rb") as manifest_file:
        return tomllib.load(manifest_file)


def _module_path(module_name: str) -> Path | None:
    relative = Path(*module_name.split(".")[1:])
    package_path = EMPEROR_ROOT / relative / "__init__.py"
    if package_path.is_file():
        return package_path
    module_path = (EMPEROR_ROOT / relative).with_suffix(".py")
    if module_path.is_file():
        return module_path
    return None


def _literal_all(source_path: Path) -> tuple[str, ...] | None:
    tree = ast.parse(source_path.read_text(encoding="utf-8"), source_path.as_posix())
    assignments = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = (node.target,)
        else:
            continue
        if any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in targets
        ):
            assignments.append(node)

    if len(assignments) != 1 or assignments[0].value is None:
        return None
    try:
        literal = ast.literal_eval(assignments[0].value)
    except (TypeError, ValueError):
        return None
    if not isinstance(literal, (list, tuple)) or not all(
        isinstance(name, str) for name in literal
    ):
        return None
    return tuple(literal)


def _dynamic_imports(tree: ast.AST) -> tuple[str, ...]:
    modules: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        function = node.func
        is_dynamic_import = (
            isinstance(function, ast.Name)
            and function.id in {"__import__", "import_module"}
        ) or (isinstance(function, ast.Attribute) and function.attr == "import_module")
        first_argument = node.args[0]
        if (
            is_dynamic_import
            and isinstance(first_argument, ast.Constant)
            and isinstance(first_argument.value, str)
        ):
            modules.add(first_argument.value)
    return tuple(sorted(modules))


def _imports_under(root: Path) -> tuple[ImportReference, ...]:
    references: set[ImportReference] = set()
    if not root.exists():
        return ()
    source_paths = (root,) if root.is_file() else root.rglob("*.py")
    for source_path in sorted(source_paths):
        if source_path.suffix != ".py":
            continue
        consumer = source_path.relative_to(PROJECT_ROOT).as_posix()
        tree = ast.parse(
            source_path.read_text(encoding="utf-8"), source_path.as_posix()
        )
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    references.add(ImportReference(consumer, alias.name, ()))
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                references.add(
                    ImportReference(
                        consumer,
                        node.module,
                        tuple(alias.name for alias in node.names),
                    )
                )
        for module_name in _dynamic_imports(tree):
            references.add(ImportReference(consumer, module_name, ()))
    return tuple(sorted(references))


def _source_module(source_path: Path) -> str:
    relative = source_path.relative_to(PROJECT_ROOT / "src").with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _emperor_owner(module_name: str) -> str:
    parts = module_name.split(".")
    return ".".join(parts[:2]) if len(parts) > 1 else module_name


def _is_type_checking_guard(node: ast.If) -> bool:
    return (isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING") or (
        isinstance(node.test, ast.Attribute)
        and isinstance(node.test.value, ast.Name)
        and node.test.value.id == "typing"
        and node.test.attr == "TYPE_CHECKING"
    )


class _RuntimeImportVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.modules: set[str] = set()

    def visit_If(self, node: ast.If) -> None:
        if _is_type_checking_guard(node):
            for statement in node.orelse:
                self.visit(statement)
            return
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        self.modules.update(alias.name for alias in node.names)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module is not None and node.level == 0:
            self.modules.add(node.module)

    def visit_Call(self, node: ast.Call) -> None:
        self.modules.update(_dynamic_imports(node))
        self.generic_visit(node)


def _strongly_connected_components(
    graph: dict[str, set[str]],
) -> tuple[tuple[str, ...], ...]:
    index = 0
    indices: dict[str, int] = {}
    low_links: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    components: list[tuple[str, ...]] = []

    def visit(node: str) -> None:
        nonlocal index
        indices[node] = index
        low_links[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)

        for dependency in sorted(graph.get(node, ())):
            if dependency not in indices:
                visit(dependency)
                low_links[node] = min(low_links[node], low_links[dependency])
            elif dependency in on_stack:
                low_links[node] = min(low_links[node], indices[dependency])

        if low_links[node] != indices[node]:
            return
        component: list[str] = []
        while True:
            member = stack.pop()
            on_stack.remove(member)
            component.append(member)
            if member == node:
                break
        if len(component) > 1:
            components.append(tuple(sorted(component)))

    for node in sorted(graph):
        if node not in indices:
            visit(node)
    return tuple(sorted(components))


def _runtime_dependency_cycles() -> tuple[tuple[str, ...], ...]:
    graph: dict[str, set[str]] = {}
    for source_path in sorted(EMPEROR_ROOT.rglob("*.py")):
        source_owner = _emperor_owner(_source_module(source_path))
        graph.setdefault(source_owner, set())
        tree = ast.parse(
            source_path.read_text(encoding="utf-8"), source_path.as_posix()
        )
        visitor = _RuntimeImportVisitor()
        visitor.visit(tree)
        for imported_module in visitor.modules:
            if not imported_module.startswith("emperor."):
                continue
            imported_owner = _emperor_owner(imported_module)
            if imported_owner != source_owner:
                graph[source_owner].add(imported_owner)
                graph.setdefault(imported_owner, set())
    return _strongly_connected_components(graph)


def _retired_module_matches(module: str, retired_modules: tuple[str, ...]) -> bool:
    return any(
        module == retired or module.startswith(f"{retired}.")
        for retired in retired_modules
    )


def _documentation_paths(root: Path) -> tuple[Path, ...]:
    if not root.exists():
        return ()
    if root.is_file():
        return (root,)
    return tuple(
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower() in {".md", ".rst", ".txt"}
    )


class EmperorInterfaceArchitectureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = _load_manifest()
        cls.interfaces = {
            interface["module"]: interface for interface in cls.manifest["interfaces"]
        }
        cls.retired_modules = tuple(cls.manifest["retired_modules"])
        cls.retired_symbols = {
            (entry["module"], entry["name"])
            for entry in cls.manifest["retired_symbol_imports"]
        }

    def test_manifest_distinguishes_all_interface_kinds(self) -> None:
        self.assertEqual(self.manifest["schema_version"], 2)
        self.assertTrue(self.manifest["migration_complete"])
        self.assertEqual(
            len(self.interfaces),
            len(self.manifest["interfaces"]),
            "Interface Modules must be unique",
        )
        self.assertEqual(
            {contract["kind"] for contract in self.interfaces.values()},
            {"configuration", "runtime", "namespace"},
        )

        for module_name, contract in self.interfaces.items():
            with self.subTest(module=module_name):
                exports = tuple(contract["exports"])
                owners = tuple(contract["owners"])
                self.assertEqual(len(exports), len(set(exports)))
                self.assertEqual(len(exports), len(owners))
                if contract["kind"] == "namespace":
                    self.assertEqual(exports, ())
                else:
                    self.assertTrue(exports)
                self.assertIsNotNone(_module_path(module_name))

        self.assertEqual(
            len(self.retired_modules),
            len(set(self.retired_modules)),
        )
        self.assertFalse(set(self.interfaces) & set(self.retired_modules))

    def test_interfaces_have_exact_literal_exports(self) -> None:
        for module_name, contract in self.interfaces.items():
            with self.subTest(module=module_name):
                source_path = _module_path(module_name)
                assert source_path is not None
                self.assertEqual(_literal_all(source_path), tuple(contract["exports"]))

    def test_root_package_has_no_eager_feature_exports(self) -> None:
        root_path = EMPEROR_ROOT / "__init__.py"
        tree = ast.parse(root_path.read_text(encoding="utf-8"), root_path.as_posix())
        eager_imports = [
            node
            for node in tree.body
            if isinstance(node, (ast.Import, ast.ImportFrom))
            and not (isinstance(node, ast.ImportFrom) and node.module == "__future__")
        ]
        self.assertEqual(eager_imports, [])
        self.assertIsNone(_literal_all(root_path))

    def test_every_initializer_rejects_facade_mechanics(self) -> None:
        violations: list[tuple[str, str]] = []
        for source_path in sorted(EMPEROR_ROOT.rglob("__init__.py")):
            relative_path = source_path.relative_to(PROJECT_ROOT).as_posix()
            tree = ast.parse(
                source_path.read_text(encoding="utf-8"),
                source_path.as_posix(),
            )
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and node.id in {
                    "_LAZY_EXPORTS",
                    "TYPE_CHECKING",
                }:
                    violations.append((relative_path, node.id))
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
                    node.name == "__getattr__"
                ):
                    violations.append((relative_path, "__getattr__"))
                elif isinstance(node, ast.Import):
                    if any(alias.name == "importlib" for alias in node.names):
                        violations.append((relative_path, "importlib"))
                elif isinstance(node, ast.ImportFrom):
                    if node.module == "importlib" or any(
                        alias.name == "TYPE_CHECKING" for alias in node.names
                    ):
                        violations.append((relative_path, node.module or "relative"))
                elif isinstance(node, ast.Call):
                    function = node.func
                    if (
                        isinstance(function, ast.Name)
                        and function.id in {"__import__", "import_module"}
                    ) or (
                        isinstance(function, ast.Attribute)
                        and function.attr == "import_module"
                    ):
                        violations.append((relative_path, "dynamic import"))

        self.assertEqual(violations, [])

    def test_retired_owner_modules_are_physically_absent(self) -> None:
        remaining = [
            module_name
            for module_name in self.retired_modules
            if _module_path(module_name) is not None
        ]
        self.assertEqual(remaining, [])

    def test_external_consumers_use_only_declared_interfaces(self) -> None:
        invalid_references: list[tuple[str, str, str]] = []
        for scope in self.manifest["external_interface_scopes"]:
            for reference in _imports_under(PROJECT_ROOT / scope):
                if reference.module != "emperor" and not reference.module.startswith(
                    "emperor."
                ):
                    continue
                if _retired_module_matches(reference.module, self.retired_modules):
                    invalid_references.append(
                        (reference.consumer, reference.module, "retired Module")
                    )
                    continue
                for name in reference.names:
                    if (reference.module, name) in self.retired_symbols:
                        invalid_references.append(
                            (
                                reference.consumer,
                                f"{reference.module}.{name}",
                                "retired symbol",
                            )
                        )

                if reference.module == "emperor" and not reference.names:
                    continue
                contract = self.interfaces.get(reference.module)
                if contract is None:
                    invalid_references.append(
                        (reference.consumer, reference.module, "undeclared Module")
                    )
                    continue
                declared_exports = set(contract["exports"])
                for name in reference.names:
                    if name == "*" or name not in declared_exports:
                        invalid_references.append(
                            (
                                reference.consumer,
                                f"{reference.module}.{name}",
                                "undeclared export",
                            )
                        )

        self.assertEqual(invalid_references, [])

    def test_documentation_contains_no_retired_imports(self) -> None:
        violations: list[tuple[str, str]] = []
        root_exports = {
            entry["module"]: set(entry["names"])
            for entry in self.manifest["retired_root_exports"]
        }
        for scope in self.manifest["documentation_scopes"]:
            for path in _documentation_paths(PROJECT_ROOT / scope):
                text = path.read_text(encoding="utf-8")
                relative_path = path.relative_to(PROJECT_ROOT).as_posix()
                for retired_module in self.retired_modules:
                    if retired_module in text:
                        violations.append((relative_path, retired_module))
                for module_name, retired_names in root_exports.items():
                    direct_reference = re.compile(
                        rf"\b{re.escape(module_name)}\.({'|'.join(map(re.escape, retired_names))})\b"
                    )
                    if direct_reference.search(text):
                        violations.append((relative_path, module_name))
                    import_pattern = re.compile(
                        rf"from\s+{re.escape(module_name)}\s+import\s+(\([^)]*\)|[^\n]+)",
                        re.DOTALL,
                    )
                    for match in import_pattern.finditer(text):
                        imported_names = set(
                            re.findall(r"\b[A-Za-z_]\w*\b", match.group(1))
                        )
                        if imported_names & retired_names:
                            violations.append((relative_path, module_name))

        self.assertEqual(violations, [])

    def test_internal_cross_owner_private_imports_match_exact_debt(self) -> None:
        actual: set[tuple[str, str]] = set()
        for source_path in sorted(EMPEROR_ROOT.rglob("*.py")):
            consumer = _source_module(source_path)
            consumer_owner = _emperor_owner(consumer)
            tree = ast.parse(
                source_path.read_text(encoding="utf-8"),
                source_path.as_posix(),
            )
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom) or node.module is None:
                    continue
                if not node.module.startswith("emperor."):
                    continue
                if node.module == "emperor._validation":
                    continue
                if _emperor_owner(node.module) == consumer_owner:
                    continue
                for alias in node.names:
                    full_name = f"{node.module}.{alias.name}"
                    if any(
                        segment.startswith("_") for segment in full_name.split(".")[1:]
                    ):
                        actual.add((consumer, full_name))

        expected = {
            (entry["consumer"], entry["imported"])
            for entry in self.manifest["legacy_cross_owner_private_imports"]
        }
        self.assertEqual(actual, expected)

    def test_runtime_dependency_cycles_match_exact_debt(self) -> None:
        expected = tuple(
            tuple(cycle.split(","))
            for cycle in self.manifest["legacy_dependency_cycles"]
        )
        self.assertEqual(_runtime_dependency_cycles(), expected)

    def test_local_checkpoints_do_not_reference_retired_owners(self) -> None:
        retired_patterns = [
            re.compile(re.escape(module.encode()) + rb"(?![A-Za-z0-9_.])")
            for module in self.retired_modules
        ]
        violations: list[tuple[str, str]] = []
        checkpoint_paths = [
            path
            for path in PROJECT_ROOT.rglob("*.ckpt")
            if not {".git", "node_modules"} & set(path.parts)
        ]
        for checkpoint_path in checkpoint_paths:
            payloads: list[bytes] = []
            if zipfile.is_zipfile(checkpoint_path):
                with zipfile.ZipFile(checkpoint_path) as archive:
                    payloads.extend(
                        archive.read(name)
                        for name in archive.namelist()
                        if name.endswith("data.pkl")
                    )
            else:
                payloads.append(checkpoint_path.read_bytes())
            for payload in payloads:
                for module_name, pattern in zip(
                    self.retired_modules, retired_patterns, strict=True
                ):
                    if pattern.search(payload):
                        violations.append(
                            (
                                checkpoint_path.relative_to(PROJECT_ROOT).as_posix(),
                                module_name,
                            )
                        )
                for module_name, symbol_name in self.retired_symbols:
                    module_position = payload.find(module_name.encode())
                    symbol_position = payload.find(
                        symbol_name.encode(),
                        max(module_position, 0),
                    )
                    if (
                        module_position >= 0
                        and symbol_position >= 0
                        and symbol_position - module_position < 256
                    ):
                        violations.append(
                            (
                                checkpoint_path.relative_to(PROJECT_ROOT).as_posix(),
                                f"{module_name}.{symbol_name}",
                            )
                        )

        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()

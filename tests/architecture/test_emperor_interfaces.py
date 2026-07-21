from __future__ import annotations

import ast
import hashlib
import tomllib
import unittest
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
    assignments = [
        node
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and (
            (
                isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "__all__"
                    for target in node.targets
                )
            )
            or (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id == "__all__"
            )
        )
    ]
    if len(assignments) != 1:
        return None
    value = assignments[0].value
    if value is None:
        return None
    try:
        literal = ast.literal_eval(value)
    except (ValueError, TypeError):
        return None
    if not isinstance(literal, (list, tuple)) or not all(
        isinstance(name, str) for name in literal
    ):
        return None
    return tuple(literal)


def _dynamic_imports(tree: ast.AST) -> list[str]:
    modules: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        function = node.func
        is_import = (
            isinstance(function, ast.Name)
            and function.id in {"__import__", "import_module"}
        ) or (isinstance(function, ast.Attribute) and function.attr == "import_module")
        first_argument = node.args[0]
        if (
            is_import
            and isinstance(first_argument, ast.Constant)
            and isinstance(first_argument.value, str)
            and first_argument.value.startswith("emperor")
        ):
            modules.append(first_argument.value)
    return modules


def _imports_under(root: Path) -> list[ImportReference]:
    references: set[ImportReference] = set()
    if not root.exists():
        return []
    for source_path in sorted(root.rglob("*.py")):
        consumer = source_path.relative_to(PROJECT_ROOT).as_posix()
        tree = ast.parse(
            source_path.read_text(encoding="utf-8"),
            source_path.as_posix(),
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
    return sorted(references)


def _edge_digest(edges: set[tuple[str, str]]) -> str:
    payload = "".join(
        f"{consumer}\0{module_name}\n" for consumer, module_name in sorted(edges)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _emperor_owner(module_name: str) -> str:
    parts = module_name.split(".")
    return ".".join(parts[:2]) if len(parts) > 1 else module_name


def _source_module(source_path: Path) -> str:
    relative = source_path.relative_to(PROJECT_ROOT / "src").with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _interface_owner(
    module_name: str,
    public_interfaces: tuple[str, ...],
) -> str:
    matching_interfaces = [
        interface
        for interface in public_interfaces
        if module_name == interface or module_name.startswith(f"{interface}.")
    ]
    if matching_interfaces:
        return max(matching_interfaces, key=len)
    return _emperor_owner(module_name)


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
        function = node.func
        is_import = (
            isinstance(function, ast.Name)
            and function.id in {"__import__", "import_module"}
        ) or (isinstance(function, ast.Attribute) and function.attr == "import_module")
        if (
            is_import
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            self.modules.add(node.args[0].value)
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


def _runtime_dependency_cycles(
    public_interfaces: tuple[str, ...],
) -> tuple[tuple[str, ...], ...]:
    graph: dict[str, set[str]] = {}
    for source_path in sorted(EMPEROR_ROOT.rglob("*.py")):
        source_module = _source_module(source_path)
        source_owner = _interface_owner(source_module, public_interfaces)
        graph.setdefault(source_owner, set())
        tree = ast.parse(
            source_path.read_text(encoding="utf-8"),
            source_path.as_posix(),
        )
        visitor = _RuntimeImportVisitor()
        visitor.visit(tree)
        for imported_module in visitor.modules:
            if not imported_module.startswith("emperor."):
                continue
            imported_owner = _interface_owner(imported_module, public_interfaces)
            if imported_owner != source_owner:
                graph[source_owner].add(imported_owner)
                graph.setdefault(imported_owner, set())
    return _strongly_connected_components(graph)


class EmperorInterfaceArchitectureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = _load_manifest()
        cls.public_interfaces = tuple(cls.manifest["public_interfaces"])
        cls.pending_interfaces = tuple(cls.manifest["pending_interfaces"])
        cls.protected_interfaces = cls.manifest["protected_interfaces"]

    def test_manifest_is_exact_and_internally_consistent(self) -> None:
        self.assertEqual(self.manifest["schema_version"], 1)
        self.assertEqual(len(self.public_interfaces), len(set(self.public_interfaces)))
        self.assertEqual(
            len(self.pending_interfaces),
            len(set(self.pending_interfaces)),
        )
        self.assertTrue(set(self.pending_interfaces) <= set(self.public_interfaces))
        self.assertEqual(
            set(self.protected_interfaces),
            set(self.public_interfaces) - set(self.pending_interfaces),
        )

        if self.manifest["migration_complete"]:
            self.assertEqual(self.pending_interfaces, ())
            self.assertEqual(self.manifest["allowed_legacy_modules"], [])
            self.assertEqual(self.manifest["legacy_import_count"], 0)
            self.assertEqual(self.manifest["legacy_cross_owner_private_imports"], [])
            self.assertEqual(self.manifest["legacy_dependency_cycles"], [])

    def test_protected_interfaces_have_exact_literal_exports(self) -> None:
        for module_name, contract in self.protected_interfaces.items():
            with self.subTest(module=module_name):
                source_path = _module_path(module_name)
                self.assertIsNotNone(source_path)
                assert source_path is not None
                actual_exports = _literal_all(source_path)
                expected_exports = tuple(contract["exports"])
                self.assertEqual(actual_exports, expected_exports)
                self.assertEqual(len(expected_exports), len(set(expected_exports)))

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

    def test_obsolete_layout_paths_cannot_return(self) -> None:
        forbidden_grab_bag_directories = sorted(
            path.relative_to(PROJECT_ROOT).as_posix()
            for path in EMPEROR_ROOT.rglob("*")
            if path.is_dir() and path.name in {"base", "core"}
        )
        forbidden_flat_paths = [
            path.relative_to(PROJECT_ROOT).as_posix()
            for path in (
                EMPEROR_ROOT / "transformer" / "feed_forward",
                EMPEROR_ROOT / "transformer" / "feed_forward.py",
                EMPEROR_ROOT / "experiments" / "tasks",
                EMPEROR_ROOT / "experiments" / "tasks.py",
            )
            if path.exists()
        ]

        self.assertEqual(
            forbidden_grab_bag_directories + forbidden_flat_paths,
            [],
        )

    def test_external_consumers_match_the_exact_legacy_import_ledger(self) -> None:
        allowed_legacy = set(self.manifest["allowed_legacy_modules"])
        actual_legacy_modules: set[str] = set()
        global_edges: set[tuple[str, str]] = set()
        unexpected_imports: list[tuple[str, str]] = []
        private_imports: list[tuple[str, str]] = []
        invalid_interface_symbols: list[tuple[str, str, str]] = []

        scopes = self.manifest["legacy_import_scopes"]
        for scope in scopes:
            root = PROJECT_ROOT / scope["root"]
            scope_edges: set[tuple[str, str]] = set()
            for reference in _imports_under(root):
                if reference.module != "emperor" and not reference.module.startswith(
                    "emperor."
                ):
                    continue

                module_segments = reference.module.split(".")[1:]
                private_names = [
                    name for name in reference.names if name.startswith("_")
                ]
                if any(segment.startswith("_") for segment in module_segments):
                    private_imports.append((reference.consumer, reference.module))
                if private_names:
                    private_imports.extend(
                        (reference.consumer, f"{reference.module}.{name}")
                        for name in private_names
                    )

                if reference.module in self.protected_interfaces:
                    exports = set(
                        self.protected_interfaces[reference.module]["exports"]
                    )
                    for name in reference.names:
                        if name == "*" or name not in exports:
                            invalid_interface_symbols.append(
                                (reference.consumer, reference.module, name)
                            )
                    continue
                if reference.module in self.pending_interfaces:
                    continue
                if reference.module == "emperor" and not reference.names:
                    continue

                edge = (reference.consumer, reference.module)
                scope_edges.add(edge)
                actual_legacy_modules.add(reference.module)
                if reference.module not in allowed_legacy:
                    unexpected_imports.append(edge)

            global_edges.update(scope_edges)
            self.assertEqual(
                len(scope_edges),
                scope["count"],
                msg=f"legacy import count changed for {scope['root']}",
            )
            self.assertEqual(
                _edge_digest(scope_edges),
                scope["sha256"],
                msg=f"legacy import ledger changed for {scope['root']}",
            )

        self.assertEqual(private_imports, [])
        self.assertEqual(invalid_interface_symbols, [])
        self.assertEqual(unexpected_imports, [])
        self.assertEqual(actual_legacy_modules, allowed_legacy)
        self.assertEqual(len(global_edges), self.manifest["legacy_import_count"])
        self.assertEqual(
            _edge_digest(global_edges),
            self.manifest["legacy_import_sha256"],
        )

    def test_internal_cross_owner_private_imports_match_exact_ledger(self) -> None:
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
                imported_owner = _emperor_owner(node.module)
                if imported_owner == consumer_owner:
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

    def test_runtime_dependency_cycles_match_exact_ledger(self) -> None:
        actual = _runtime_dependency_cycles(self.public_interfaces)
        expected = tuple(
            tuple(cycle.split(","))
            for cycle in self.manifest["legacy_dependency_cycles"]
        )

        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()

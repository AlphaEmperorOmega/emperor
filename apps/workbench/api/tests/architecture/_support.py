from __future__ import annotations

import ast
import hashlib
import json
import os
import stat
import subprocess
import sys
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ARCHITECTURE_ROOT = Path(__file__).resolve().parent
API_ROOT = ARCHITECTURE_ROOT.parents[1]
REPO_ROOT = ARCHITECTURE_ROOT.parents[4]
PACKAGE_ROOT = API_ROOT / "src" / "emperor_workbench"
PACKAGE_NAME = "emperor_workbench"
MANIFEST_PATH = ARCHITECTURE_ROOT / "emperor_workbench_interfaces.toml"
IMPORT_PURITY_RESULT_PREFIX = "__EMPEROR_IMPORT_PURITY__="

_IMPORT_PURITY_PROBE = r"""
import asyncio
import concurrent.futures
import _thread
import importlib
import json
import multiprocessing.process
import os
from pathlib import Path
import sys
import threading

monitor_root = Path(os.environ["EMPEROR_IMPORT_PURITY_ROOT"]).resolve()
source_roots = [Path(argument).resolve() for argument in sys.argv[1:-1]]
module_name = sys.argv[-1]
result_prefix = os.environ["EMPEROR_IMPORT_PURITY_RESULT_PREFIX"]
effects = set()
for source_root in reversed(source_roots):
    sys.path.insert(0, str(source_root))


def side_effect_path(value):
    if isinstance(value, int):
        return None
    try:
        raw_path = os.fspath(value)
    except TypeError:
        return None
    if isinstance(raw_path, bytes):
        raw_path = os.fsdecode(raw_path)
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    try:
        candidate = candidate.resolve(strict=False)
        relative = candidate.relative_to(monitor_root)
    except OSError:
        return "<unresolved>"
    except ValueError:
        return "<outside-isolated-root>"
    return relative.as_posix() or "."


def record_filesystem_paths(event, arguments):
    for argument in arguments:
        path_label = side_effect_path(argument)
        if path_label is not None:
            effects.add(f"filesystem:{event}:{path_label}")


write_flags = (
    os.O_WRONLY
    | os.O_RDWR
    | os.O_APPEND
    | os.O_CREAT
    | os.O_TRUNC
)
filesystem_events = {
    "os.chmod",
    "os.chown",
    "os.link",
    "os.mkdir",
    "os.remove",
    "os.rename",
    "os.rmdir",
    "os.symlink",
    "os.truncate",
    "os.utime",
    "shutil.copyfile",
    "shutil.copymode",
    "shutil.copystat",
    "sqlite3.connect",
}
process_events = {
    "os.fork",
    "os.forkpty",
    "os.posix_spawn",
    "os.posix_spawnp",
    "os.spawn",
    "os.system",
    "pty.spawn",
    "subprocess.Popen",
}


def audit_import(event, arguments):
    if event == "open":
        path, mode, flags = arguments
        writes = (
            isinstance(mode, str)
            and any(marker in mode for marker in ("w", "a", "x", "+"))
        ) or (
            mode is None
            and isinstance(flags, int)
            and bool(flags & write_flags)
        )
        if writes:
            record_filesystem_paths(event, (path,))
        return
    if event in filesystem_events:
        record_filesystem_paths(event, arguments)
        return
    if event in process_events:
        effects.add(f"process:{event}")
        raise RuntimeError(f"Import attempted process side effect: {event}")


sys.addaudithook(audit_import)


def deny_thread_start(thread, *args, **kwargs):
    effects.add(f"thread:start:{thread.name}")
    raise RuntimeError("Import attempted to start a thread")


def deny_low_level_thread(*args, **kwargs):
    effects.add("thread:_thread.start_new_thread")
    raise RuntimeError("Import attempted to start a low-level thread")


def deny_thread_pool(*args, **kwargs):
    effects.add("executor:ThreadPoolExecutor")
    raise RuntimeError("Import attempted to allocate a thread pool")


def deny_process_pool(*args, **kwargs):
    effects.add("executor:ProcessPoolExecutor")
    raise RuntimeError("Import attempted to allocate a process pool")


def deny_async_task(*args, **kwargs):
    effects.add("task:asyncio.create_task")
    raise RuntimeError("Import attempted to create an asyncio task")


threading.Thread.start = deny_thread_start
_thread.start_new_thread = deny_low_level_thread
concurrent.futures.ThreadPoolExecutor.__init__ = deny_thread_pool
concurrent.futures.ProcessPoolExecutor.__init__ = deny_process_pool
multiprocessing.process.BaseProcess.start = deny_process_pool
asyncio.create_task = deny_async_task
threads_before = {id(thread) for thread in threading.enumerate()}

try:
    importlib.import_module(module_name)
except BaseException as error:
    effects.add(f"import-error:{type(error).__name__}:{error}")

for thread in threading.enumerate():
    if id(thread) not in threads_before:
        effects.add(f"thread:alive:{thread.name}")

print(result_prefix + json.dumps(sorted(effects)))
"""


@dataclass(frozen=True, order=True)
class DependencyViolation:
    source: str
    imported: str
    source_owner: str
    imported_owner: str

    def as_dict(self) -> dict[str, str]:
        return {
            "source": self.source,
            "imported": self.imported,
            "source_owner": self.source_owner,
            "imported_owner": self.imported_owner,
        }


@dataclass(frozen=True, order=True)
class PrivateImport:
    source: str
    imported: str
    source_owner: str
    imported_owner: str

    def as_dict(self) -> dict[str, str]:
        return {
            "source": self.source,
            "imported": self.imported,
            "source_owner": self.source_owner,
            "imported_owner": self.imported_owner,
        }


@dataclass(frozen=True, order=True)
class FrameworkImport:
    source: str
    imported: str

    def as_dict(self) -> dict[str, str]:
        return {"source": self.source, "imported": self.imported}


@dataclass(frozen=True, order=True)
class AppStateViolation:
    module: str
    marker: str

    def as_dict(self) -> dict[str, str]:
        return {"module": self.module, "marker": self.marker}


def load_manifest() -> dict[str, Any]:
    with MANIFEST_PATH.open("rb") as manifest_file:
        return tomllib.load(manifest_file)


def source_paths() -> tuple[Path, ...]:
    return tuple(sorted(PACKAGE_ROOT.rglob("*.py")))


def _tree_fingerprint(root: Path) -> tuple[tuple[str, str, int, int, str], ...]:
    entries: list[tuple[str, str, int, int, str]] = []
    for path in sorted(root.rglob("*")):
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            kind = "symlink"
            content = os.readlink(path)
        elif stat.S_ISDIR(metadata.st_mode):
            kind = "directory"
            content = ""
        elif stat.S_ISREG(metadata.st_mode):
            kind = "file"
            content = hashlib.sha256(path.read_bytes()).hexdigest()
        else:
            kind = "other"
            content = ""
        entries.append(
            (
                path.relative_to(root).as_posix(),
                kind,
                stat.S_IMODE(metadata.st_mode),
                metadata.st_mtime_ns,
                content,
            )
        )
    return tuple(entries)


def module_import_side_effects(
    module_name: str,
    *,
    python_paths: tuple[Path, ...] = (),
) -> tuple[str, ...]:
    with tempfile.TemporaryDirectory(prefix="emperor-import-purity-") as temporary:
        monitor_root = Path(temporary)
        default_root = monitor_root / "defaults"
        runtime_root = monitor_root / "runtime"
        temporary_root = monitor_root / "tmp"
        for path in (default_root, runtime_root, temporary_root):
            path.mkdir()

        if os.name != "nt":
            default_root.chmod(0o555)
        before = _tree_fingerprint(monitor_root)
        environment = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("EMPEROR_", "WORKBENCH_API_"))
            and key not in {"PYTHONPATH", "PYTHONHOME"}
        }
        environment.update(
            {
                "APPDATA": str(default_root / "appdata"),
                "EMPEROR_IMPORT_PURITY_RESULT_PREFIX": IMPORT_PURITY_RESULT_PREFIX,
                "EMPEROR_IMPORT_PURITY_ROOT": str(monitor_root),
                "HOME": str(default_root / "home"),
                "LOCALAPPDATA": str(default_root / "local-appdata"),
                "MPLCONFIGDIR": str(default_root / "matplotlib"),
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONNOUSERSITE": "1",
                "TEMP": str(temporary_root),
                "TMP": str(temporary_root),
                "TMPDIR": str(temporary_root),
                "TORCH_HOME": str(default_root / "torch"),
                "USERPROFILE": str(default_root / "home"),
                "WORKBENCH_API_LOGS_ROOT": str(runtime_root / "logs"),
                "WORKBENCH_API_SNAPSHOTS_ROOT": str(runtime_root / "snapshots"),
                "WORKBENCH_API_STATE_ROOT": str(runtime_root / "state"),
                "XDG_CACHE_HOME": str(default_root / "xdg-cache"),
                "XDG_CONFIG_HOME": str(default_root / "xdg-config"),
                "XDG_DATA_HOME": str(default_root / "xdg-data"),
                "XDG_STATE_HOME": str(default_root / "xdg-state"),
            }
        )
        source_roots = (API_ROOT / "src", *python_paths)
        command = [
            sys.executable,
            "-I",
            "-B",
            "-c",
            _IMPORT_PURITY_PROBE,
            *(str(path) for path in source_roots),
            module_name,
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=monitor_root,
                env=environment,
                capture_output=True,
                check=False,
                text=True,
                timeout=30,
            )
            after = _tree_fingerprint(monitor_root)
        finally:
            if os.name != "nt":
                default_root.chmod(0o700)

    effects: set[str] = set()
    result_line = next(
        (
            line
            for line in reversed(completed.stdout.splitlines())
            if line.startswith(IMPORT_PURITY_RESULT_PREFIX)
        ),
        None,
    )
    if result_line is None:
        effects.add("probe:missing-result")
    else:
        payload = result_line.removeprefix(IMPORT_PURITY_RESULT_PREFIX)
        try:
            result = json.loads(payload)
        except json.JSONDecodeError as error:
            effects.add(f"probe:invalid-result:{error}")
        else:
            if not isinstance(result, list) or not all(
                isinstance(item, str) for item in result
            ):
                effects.add("probe:invalid-result-shape")
            else:
                effects.update(result)
    if completed.returncode != 0:
        effects.add(f"probe:return-code:{completed.returncode}")
    if before != after:
        effects.add("filesystem:tree-changed")
    return tuple(sorted(effects))


def source_module(source_path: Path) -> str:
    relative = source_path.relative_to(PACKAGE_ROOT).with_suffix("")
    parts = [PACKAGE_NAME, *relative.parts]
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def module_path(module_name: str) -> Path | None:
    if module_name == PACKAGE_NAME:
        return PACKAGE_ROOT / "__init__.py"
    relative = Path(*module_name.split(".")[1:])
    package_path = PACKAGE_ROOT / relative / "__init__.py"
    if package_path.is_file():
        return package_path
    source_path = (PACKAGE_ROOT / relative).with_suffix(".py")
    if source_path.is_file():
        return source_path
    return None


def parse_source(source_path: Path) -> ast.Module:
    return ast.parse(
        source_path.read_text(encoding="utf-8"),
        filename=source_path.as_posix(),
    )


def literal_all(source_path: Path) -> tuple[str, ...] | None:
    assignments: list[ast.Assign | ast.AnnAssign] = []
    for node in parse_source(source_path).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        ):
            assignments.append(node)
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "__all__"
        ):
            assignments.append(node)
    if len(assignments) != 1 or assignments[0].value is None:
        return None
    try:
        value = ast.literal_eval(assignments[0].value)
    except (TypeError, ValueError):
        return None
    if not isinstance(value, (list, tuple)) or not all(
        isinstance(item, str) for item in value
    ):
        return None
    return tuple(value)


def top_level_functions(source_path: Path) -> frozenset[str]:
    return frozenset(
        node.name
        for node in parse_source(source_path).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    )


def owner_prefixes(manifest: dict[str, Any]) -> dict[str, tuple[str, ...]]:
    return {
        str(owner["name"]): tuple(str(prefix) for prefix in owner["prefixes"])
        for owner in manifest["owners"]
    }


def owner_for(module_name: str, manifest: dict[str, Any]) -> str | None:
    matches = [
        (len(prefix), owner_name)
        for owner_name, prefixes in owner_prefixes(manifest).items()
        for prefix in prefixes
        if module_name == prefix or module_name.startswith(f"{prefix}.")
    ]
    if not matches:
        return None
    return max(matches)[1]


def public_modules() -> tuple[str, ...]:
    modules: list[str] = []
    for source_path in source_paths():
        relative = source_path.relative_to(PACKAGE_ROOT)
        parts = list(relative.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        if not parts or any(part.startswith("_") for part in parts):
            continue
        modules.append(".".join((PACKAGE_NAME, *parts)))
    return tuple(sorted(modules))


def legacy_public_modules(manifest: dict[str, Any]) -> tuple[str, ...]:
    target_interfaces = {
        *manifest["public_interfaces"],
        *manifest["process_interfaces"],
    }
    return tuple(
        module_name
        for module_name in public_modules()
        if module_name not in target_interfaces
    )


def _is_type_checking_guard(node: ast.If) -> bool:
    return (isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING") or (
        isinstance(node.test, ast.Attribute)
        and isinstance(node.test.value, ast.Name)
        and node.test.value.id == "typing"
        and node.test.attr == "TYPE_CHECKING"
    )


def _resolved_from_module(
    source_path: Path,
    source_name: str,
    node: ast.ImportFrom,
) -> str | None:
    if node.level == 0:
        return node.module
    package_name = (
        source_name
        if source_path.name == "__init__.py"
        else source_name.rpartition(".")[0]
    )
    package_parts = package_name.split(".")
    remove_count = node.level - 1
    if remove_count > len(package_parts):
        return None
    base_parts = package_parts[: len(package_parts) - remove_count]
    if node.module:
        base_parts.extend(node.module.split("."))
    return ".".join(base_parts)


class RuntimeImportVisitor(ast.NodeVisitor):
    def __init__(self, source_path: Path, source_name: str) -> None:
        self.source_path = source_path
        self.source_name = source_name
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
        resolved = _resolved_from_module(self.source_path, self.source_name, node)
        if resolved is not None:
            self.modules.add(resolved)

    def visit_Call(self, node: ast.Call) -> None:
        function = node.func
        is_dynamic_import = (
            isinstance(function, ast.Name)
            and function.id in {"__import__", "import_module"}
        ) or (isinstance(function, ast.Attribute) and function.attr == "import_module")
        if (
            is_dynamic_import
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            self.modules.add(node.args[0].value)
        self.generic_visit(node)


def runtime_imports(source_path: Path) -> frozenset[str]:
    source_name = source_module(source_path)
    visitor = RuntimeImportVisitor(source_path, source_name)
    visitor.visit(parse_source(source_path))
    return frozenset(visitor.modules)


def static_imports(source_path: Path) -> frozenset[str]:
    source_name = source_module(source_path)
    modules: set[str] = set()
    tree = parse_source(source_path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            resolved = _resolved_from_module(source_path, source_name, node)
            if resolved is not None:
                modules.add(resolved)
        elif isinstance(node, ast.Call):
            function = node.func
            is_dynamic_import = (
                isinstance(function, ast.Name)
                and function.id in {"__import__", "import_module"}
            ) or (
                isinstance(function, ast.Attribute) and function.attr == "import_module"
            )
            if (
                is_dynamic_import
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                modules.add(node.args[0].value)
    return frozenset(modules)


def dependency_graph(manifest: dict[str, Any]) -> dict[str, set[str]]:
    owner_names = set(owner_prefixes(manifest))
    graph = {owner_name: set() for owner_name in owner_names}
    for source_path in source_paths():
        source_name = source_module(source_path)
        source_owner = owner_for(source_name, manifest)
        if source_owner is None:
            continue
        for imported_module in runtime_imports(source_path):
            if not imported_module.startswith(f"{PACKAGE_NAME}."):
                continue
            imported_owner = owner_for(imported_module, manifest)
            if imported_owner is not None and imported_owner != source_owner:
                graph[source_owner].add(imported_owner)
    return graph


def dependency_violations(
    manifest: dict[str, Any],
) -> tuple[DependencyViolation, ...]:
    allowed = {
        str(owner): {str(dependency) for dependency in dependencies}
        for owner, dependencies in manifest["allowed_dependencies"].items()
    }
    violations: set[DependencyViolation] = set()
    for source_path in source_paths():
        source_name = source_module(source_path)
        source_owner = owner_for(source_name, manifest)
        if source_owner is None:
            continue
        for imported_module in runtime_imports(source_path):
            if not imported_module.startswith(f"{PACKAGE_NAME}."):
                continue
            imported_owner = owner_for(imported_module, manifest)
            if (
                imported_owner is not None
                and imported_owner != source_owner
                and imported_owner not in allowed[source_owner]
            ):
                violations.add(
                    DependencyViolation(
                        source_name,
                        imported_module,
                        source_owner,
                        imported_owner,
                    )
                )
    return tuple(sorted(violations))


def strongly_connected_components(
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


def _private_imports_from_node(
    source_path: Path,
    source_name: str,
    node: ast.ImportFrom,
) -> tuple[str, ...]:
    module_name = _resolved_from_module(source_path, source_name, node)
    if module_name is None:
        return ()
    imported: list[str] = []
    if any(part.startswith("_") for part in module_name.split(".")):
        imported.append(module_name)
    imported.extend(
        f"{module_name}.{alias.name}"
        for alias in node.names
        if alias.name.startswith("_")
    )
    return tuple(imported)


def private_imports(manifest: dict[str, Any]) -> tuple[PrivateImport, ...]:
    violations: set[PrivateImport] = set()
    for source_path in source_paths():
        source_name = source_module(source_path)
        source_owner = owner_for(source_name, manifest)
        if source_owner is None:
            continue
        tree = parse_source(source_path)
        for node in ast.walk(tree):
            imported_names: tuple[str, ...] = ()
            if isinstance(node, ast.Import):
                imported_names = tuple(
                    alias.name
                    for alias in node.names
                    if any(part.startswith("_") for part in alias.name.split("."))
                )
            elif isinstance(node, ast.ImportFrom):
                imported_names = _private_imports_from_node(
                    source_path,
                    source_name,
                    node,
                )
            for imported_name in imported_names:
                if not imported_name.startswith(f"{PACKAGE_NAME}."):
                    continue
                imported_owner = owner_for(imported_name, manifest)
                if imported_owner is not None and imported_owner != source_owner:
                    violations.add(
                        PrivateImport(
                            source_name,
                            imported_name,
                            source_owner,
                            imported_owner,
                        )
                    )
    return tuple(sorted(violations))


def wildcard_imports() -> tuple[str, ...]:
    violations: list[str] = []
    for source_path in source_paths():
        for node in ast.walk(parse_source(source_path)):
            if isinstance(node, ast.ImportFrom) and any(
                alias.name == "*" for alias in node.names
            ):
                violations.append(f"{source_module(source_path)}:{node.lineno}")
    return tuple(sorted(violations))


def sys_modules_aliases() -> tuple[str, ...]:
    violations: list[str] = []
    for source_path in source_paths():
        for node in ast.walk(parse_source(source_path)):
            if not isinstance(node, ast.Subscript):
                continue
            value = node.value
            if (
                isinstance(value, ast.Attribute)
                and isinstance(value.value, ast.Name)
                and value.value.id == "sys"
                and value.attr == "modules"
                and isinstance(node.ctx, ast.Store)
            ):
                violations.append(f"{source_module(source_path)}:{node.lineno}")
    return tuple(sorted(violations))


def modules_defining(function_name: str) -> tuple[str, ...]:
    return tuple(
        sorted(
            source_module(source_path)
            for source_path in source_paths()
            if function_name in top_level_functions(source_path)
        )
    )


def modules_with_names(names: set[str]) -> tuple[str, ...]:
    modules: list[str] = []
    for source_path in source_paths():
        found = False
        for node in parse_source(source_path).body:
            if isinstance(node, ast.Assign):
                found = any(
                    isinstance(target, ast.Name) and target.id in names
                    for target in node.targets
                )
            elif isinstance(node, ast.AnnAssign):
                found = isinstance(node.target, ast.Name) and node.target.id in names
            if found:
                modules.append(source_module(source_path))
                break
    return tuple(sorted(modules))


def obsolete_paths() -> tuple[str, ...]:
    allowed_root_files = {
        "__init__.py",
        "__main__.py",
        "cli.py",
        "failures.py",
        "settings.py",
    }
    paths = {
        path.relative_to(API_ROOT).as_posix()
        for path in PACKAGE_ROOT.glob("*.py")
        if path.name not in allowed_root_files
    }
    forbidden_root_directories = {
        "core",
        "db",
        "historical_inspection",
        "inspector",
        "repositories",
        "runtime",
        "schemas",
        "services",
        "storage",
    }
    paths.update(
        path.relative_to(API_ROOT).as_posix()
        for path in PACKAGE_ROOT.iterdir()
        if path.is_dir() and path.name in forbidden_root_directories
    )
    legacy_api_paths = (
        PACKAGE_ROOT / "api" / "mutation_policy.py",
        PACKAGE_ROOT / "api" / "v1" / "config_snapshot_mapping.py",
        PACKAGE_ROOT / "api" / "v1" / "log_archive_upload.py",
        PACKAGE_ROOT / "api" / "v1" / "logs_mapping.py",
        PACKAGE_ROOT / "api" / "v1" / "router.py",
        PACKAGE_ROOT / "api" / "v1" / "routers",
        PACKAGE_ROOT / "api" / "v1" / "training_commands.py",
        PACKAGE_ROOT / "api" / "v1" / "training_mapping.py",
    )
    paths.update(
        path.relative_to(API_ROOT).as_posix()
        for path in legacy_api_paths
        if path.exists()
    )
    return tuple(sorted(paths))


def is_transport_framework_import(module_name: str) -> bool:
    return module_name.partition(".")[0] in {"fastapi", "starlette"}


def framework_imports_outside_api() -> tuple[FrameworkImport, ...]:
    violations: set[FrameworkImport] = set()
    for source_path in source_paths():
        source_name = source_module(source_path)
        if source_name == f"{PACKAGE_NAME}.api" or source_name.startswith(
            f"{PACKAGE_NAME}.api."
        ):
            continue
        for imported_module in static_imports(source_path):
            if is_transport_framework_import(imported_module):
                violations.add(FrameworkImport(source_name, imported_module))
    return tuple(sorted(violations))


def _base_expression_name(expression: ast.expr) -> str | None:
    if isinstance(expression, ast.Name):
        return expression.id
    if isinstance(expression, ast.Attribute):
        return expression.attr
    if isinstance(expression, ast.Subscript):
        return _base_expression_name(expression.value)
    return None


def defines_http_contracts(source_path: Path) -> bool:
    contract_bases = {"ApiResponseModel", "BaseModel", "RootModel"}
    for node in parse_source(source_path).body:
        if not isinstance(node, ast.ClassDef):
            continue
        base_names = {
            base_name
            for base in node.bases
            if (base_name := _base_expression_name(base)) is not None
        }
        if base_names & contract_bases:
            return True
        if node.name.endswith(("Request", "Response")) and any(
            base_name.endswith(("Request", "Response", "ResponseModel"))
            for base_name in base_names
        ):
            return True
    return False


def _capability_namespace(module_name: str) -> str:
    parts = module_name.split(".")
    return ".".join(parts[:2])


def http_contract_modules_outside_api() -> tuple[str, ...]:
    candidates = {
        source_module(source_path): source_path
        for source_path in source_paths()
        if source_module(source_path) != f"{PACKAGE_NAME}.api"
        and not source_module(source_path).startswith(f"{PACKAGE_NAME}.api.")
    }
    contracts = {
        module_name
        for module_name, source_path in candidates.items()
        if defines_http_contracts(source_path)
    }
    imports = {
        module_name: static_imports(source_path)
        for module_name, source_path in candidates.items()
    }

    changed = True
    while changed:
        changed = False
        for module_name in tuple(contracts):
            namespace = _capability_namespace(module_name)
            for imported_module in imports[module_name]:
                if (
                    imported_module in candidates
                    and _capability_namespace(imported_module) == namespace
                    and imported_module not in contracts
                ):
                    contracts.add(imported_module)
                    changed = True
        for module_name, source_path in candidates.items():
            if module_name in contracts or source_path.name != "__init__.py":
                continue
            namespace = _capability_namespace(module_name)
            if any(
                imported_module in contracts
                and _capability_namespace(imported_module) == namespace
                for imported_module in imports[module_name]
            ):
                contracts.add(module_name)
                changed = True

    return tuple(sorted(contracts))


def facade_modules() -> tuple[str, ...]:
    return modules_defining("__getattr__")


def compatibility_marker_modules() -> tuple[str, ...]:
    return modules_with_names({"COMPATIBILITY_STATUS", "REPLACEMENT_IMPORT"})


def _assignment_names(node: ast.Assign | ast.AnnAssign) -> tuple[str, ...]:
    if isinstance(node, ast.AnnAssign):
        return (node.target.id,) if isinstance(node.target, ast.Name) else ()
    return tuple(target.id for target in node.targets if isinstance(target, ast.Name))


def _called_name(value: ast.expr | None) -> str | None:
    if not isinstance(value, ast.Call):
        return None
    function = value.func
    if isinstance(function, ast.Name):
        return function.id
    if isinstance(function, ast.Attribute):
        return function.attr
    return None


def app_state_violations() -> tuple[AppStateViolation, ...]:
    state_names = {
        "_DEFAULT_CLIENT",
        "_PROCESS_TOKEN",
        "_blocking_work_limiters",
        "_mutation_limiters",
    }
    state_calls = {
        "ProjectAdapterClient",
        "Semaphore",
        "ThreadPoolExecutor",
        "WeakKeyDictionary",
        "create_app",
        "uuid4",
    }
    violations: set[AppStateViolation] = set()
    for source_path in source_paths():
        module_name = source_module(source_path)
        tree = parse_source(source_path)
        for node in tree.body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                names = _assignment_names(node)
                called_name = _called_name(node.value)
                for name in names:
                    if (
                        module_name == "emperor_workbench.api"
                        and name == "app"
                        and called_name == "create_app"
                    ):
                        # The canonical ASGI Interface intentionally exposes a
                        # resource-free application configuration. Import
                        # purity is enforced independently below.
                        continue
                    if name in state_names or called_name in state_calls:
                        violations.add(AppStateViolation(module_name, name))
            elif (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and isinstance(node.value.func.value, ast.Name)
                and node.value.func.value.id == "atexit"
                and node.value.func.attr == "register"
            ):
                violations.add(AppStateViolation(module_name, "atexit.register"))
            elif isinstance(node, ast.ClassDef):
                for class_node in node.body:
                    if not isinstance(class_node, (ast.Assign, ast.AnnAssign)):
                        continue
                    for name in _assignment_names(class_node):
                        if name == "_metadata_cache":
                            violations.add(
                                AppStateViolation(
                                    module_name,
                                    f"{node.name}.{name}",
                                )
                            )
    return tuple(sorted(violations))


def manifest_records(
    manifest: dict[str, Any],
    key: str,
) -> tuple[dict[str, str], ...]:
    return tuple(
        sorted(
            (
                {str(name): str(value) for name, value in record.items()}
                for record in manifest[key]
            ),
            key=lambda record: tuple(record.values()),
        )
    )

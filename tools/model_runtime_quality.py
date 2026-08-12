from __future__ import annotations

import argparse
import ast
import fnmatch
import json
import os
import subprocess
import sys
import tempfile
import tomllib
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "tests/architecture/model_runtime_quality.toml"
COVERAGE_CONFIG_PATH = PROJECT_ROOT / "tests/architecture/model_runtime_coveragerc"
StructuralFinding = tuple[str, str, str]


def load_manifest() -> dict[str, Any]:
    with MANIFEST_PATH.open("rb") as manifest_file:
        return tomllib.load(manifest_file)


def _python_environment(project_root: Path = PROJECT_ROOT) -> dict[str, str]:
    environment = os.environ.copy()
    python_paths = [
        str(project_root / "src"),
        str(project_root / "tests"),
        str(project_root),
    ]
    existing_python_path = environment.get("PYTHONPATH")
    if existing_python_path:
        python_paths.append(existing_python_path)
    environment["PYTHONPATH"] = os.pathsep.join(python_paths)
    environment["PYTHONSAFEPATH"] = "1"
    environment.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    environment.setdefault("IPYTHONDIR", "/tmp/ipython")
    return environment


def _run(command: Sequence[str], *, environment: Mapping[str, str]) -> int:
    return subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=dict(environment),
        check=False,
    ).returncode


def run_tests(manifest: Mapping[str, Any]) -> int:
    tests = cast(list[str], manifest["coverage"]["tests"])
    return _run(
        [sys.executable, "-m", "unittest", *tests],
        environment=_python_environment(),
    )


def coverage_gate_errors(
    totals: Mapping[str, object],
    coverage_policy: Mapping[str, object],
) -> tuple[str, ...]:
    statement_percent = float(cast(int | float, totals["percent_statements_covered"]))
    branch_percent = float(cast(int | float, totals["percent_branches_covered"]))
    minimum_statement = float(
        cast(int | float, coverage_policy["minimum_statement_percent"])
    )
    minimum_branch = float(cast(int | float, coverage_policy["minimum_branch_percent"]))
    errors: list[str] = []
    if statement_percent < minimum_statement:
        errors.append(
            f"statement coverage {statement_percent:.2f}% is below "
            f"{minimum_statement:.2f}%"
        )
    if branch_percent < minimum_branch:
        errors.append(
            f"branch coverage {branch_percent:.2f}% is below {minimum_branch:.2f}%"
        )
    return tuple(errors)


def mutation_gate_errors(
    results: Sequence[tuple[str, str]],
    selectors: Sequence[str],
) -> tuple[str, ...]:
    if not results:
        return ("no selected mutants were generated and completed",)
    selector_counts = Counter(selectors)
    unique_selectors = tuple(selector_counts)
    errors = [
        f"duplicate mutation selector: {selector}"
        for selector, count in selector_counts.items()
        if count > 1
    ]
    for selector in unique_selectors:
        match_count = sum(fnmatch.fnmatchcase(name, selector) for name, _ in results)
        if match_count == 0:
            errors.append(f"selector matched no mutant: {selector}")
        elif match_count > 1:
            errors.append(f"selector matched {match_count} mutants: {selector}")
    for name, _ in results:
        matching_selectors = [
            selector
            for selector in unique_selectors
            if fnmatch.fnmatchcase(name, selector)
        ]
        if not matching_selectors:
            errors.append(f"mutant matched no selector: {name}")
        elif len(matching_selectors) > 1:
            errors.append(f"mutant matched multiple selectors: {name}")
    errors.extend(f"{status}: {name}" for name, status in results if status != "killed")
    return tuple(errors)


def _structural_finding_label(finding: StructuralFinding) -> str:
    return ":".join(finding)


def structural_finding_errors(
    actual: Sequence[StructuralFinding],
    allowed: Sequence[StructuralFinding],
) -> tuple[str, ...]:
    actual_counts = Counter(actual)
    allowed_counts = Counter(allowed)
    actual_findings = set(actual_counts)
    allowed_findings = set(allowed_counts)
    errors = [
        f"duplicate structural finding: {_structural_finding_label(finding)} "
        f"({count} occurrences)"
        for finding, count in sorted(actual_counts.items())
        if count > 1
    ]
    errors.extend(
        f"duplicate allowed structural finding: "
        f"{_structural_finding_label(finding)} ({count} entries)"
        for finding, count in sorted(allowed_counts.items())
        if count > 1
    )
    errors.extend(
        f"unexpected structural finding: {_structural_finding_label(finding)}"
        for finding in sorted(actual_findings - allowed_findings)
    )
    errors.extend(
        "allowed structural finding is no longer present: "
        + _structural_finding_label(finding)
        for finding in sorted(allowed_findings - actual_findings)
    )
    return tuple(errors)


class _FunctionCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.scope: list[str] = []
        self.spans: list[tuple[str, int, int]] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def _visit_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> None:
        qualified_name = ".".join((*self.scope, node.name))
        start = min(
            [node.lineno, *(decorator.lineno for decorator in node.decorator_list)]
        )
        self.spans.append((qualified_name, start, node.end_lineno or node.lineno))
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)


def _function_spans(source: str) -> tuple[tuple[str, int, int], ...]:
    collector = _FunctionCollector()
    collector.visit(ast.parse(source))
    return tuple(collector.spans)


def _function_at_line(
    spans: Sequence[tuple[str, int, int]],
    row: int,
) -> str:
    containing = [span for span in spans if span[1] <= row <= span[2]]
    if not containing:
        return "<module>"
    return min(
        containing,
        key=lambda span: (span[2] - span[1], -len(span[0])),
    )[0]


def ruff_finding_keys(
    diagnostics: Sequence[Mapping[str, Any]],
    project_root: Path,
) -> tuple[StructuralFinding, ...]:
    spans_by_path: dict[Path, tuple[tuple[str, int, int], ...]] = {}
    findings: list[StructuralFinding] = []
    for diagnostic in diagnostics:
        source_path = Path(str(diagnostic["filename"]))
        if not source_path.is_absolute():
            source_path = project_root / source_path
        source_path = source_path.resolve()
        relative_path = source_path.relative_to(project_root.resolve()).as_posix()
        spans = spans_by_path.get(source_path)
        if spans is None:
            spans = _function_spans(source_path.read_text(encoding="utf-8"))
            spans_by_path[source_path] = spans
        location = cast(Mapping[str, object], diagnostic["location"])
        findings.append(
            (
                relative_path,
                str(diagnostic["code"]),
                _function_at_line(spans, int(cast(int, location["row"]))),
            )
        )
    return tuple(findings)


def _reviewed_module_ceilings(
    policy: Mapping[str, object],
) -> dict[str, int]:
    reviewed_modules = cast(
        Sequence[Mapping[str, object]],
        policy.get("reviewed_modules", ()),
    )
    return {
        str(module["path"]): int(cast(int, module["maximum_lines"]))
        for module in reviewed_modules
    }


def structure_policy_errors(policy: Mapping[str, object]) -> tuple[str, ...]:
    allowed_findings = cast(
        Sequence[Mapping[str, object]],
        policy.get("allowed_findings", ()),
    )
    reviewed_modules = cast(
        Sequence[Mapping[str, object]],
        policy.get("reviewed_modules", ()),
    )
    errors = [
        "allowed structural finding requires a rationale: "
        + _structural_finding_label(
            (str(item["path"]), str(item["code"]), str(item["function"]))
        )
        for item in allowed_findings
        if not str(item.get("rationale", "")).strip()
    ]
    errors.extend(
        f"reviewed module requires a rationale: {item['path']}"
        for item in reviewed_modules
        if not str(item.get("rationale", "")).strip()
    )
    reviewed_counts = Counter(str(item["path"]) for item in reviewed_modules)
    errors.extend(
        f"duplicate reviewed module: {path} ({count} entries)"
        for path, count in sorted(reviewed_counts.items())
        if count > 1
    )
    return tuple(errors)


def _module_structure_errors(
    relative_path: str,
    line_count: int,
    review_threshold: int,
    reviewed_ceiling: int | None,
) -> tuple[str, ...]:
    if line_count <= review_threshold:
        return ()
    if reviewed_ceiling is None:
        return (
            f"{relative_path} has {line_count} lines and requires cohesion "
            f"review above {review_threshold}",
        )
    if line_count > reviewed_ceiling:
        return (
            f"{relative_path} has {line_count} lines, exceeding reviewed "
            f"ceiling {reviewed_ceiling}",
        )
    return ()


def _function_structure_errors(
    relative_path: str,
    source: str,
    maximum_lines: int,
) -> tuple[str, ...]:
    try:
        spans = _function_spans(source)
    except SyntaxError as exc:
        return (f"{relative_path} could not be parsed: {exc}",)
    return tuple(
        f"{relative_path}:{start} {qualified_name} has {end - start + 1} "
        f"lines (maximum {maximum_lines})"
        for qualified_name, start, end in spans
        if end - start + 1 > maximum_lines
    )


def _review_configuration_errors(
    module_lines: Mapping[str, int],
    reviewed_ceilings: Mapping[str, int],
    review_threshold: int,
) -> tuple[str, ...]:
    errors: list[str] = []
    for relative_path in sorted(reviewed_ceilings):
        line_count = module_lines.get(relative_path)
        if line_count is None:
            errors.append(f"reviewed module does not exist in source: {relative_path}")
        elif line_count <= review_threshold:
            errors.append(
                f"{relative_path} is reviewed but has only {line_count} lines "
                f"(review threshold {review_threshold})"
            )
    return tuple(errors)


def _source_symlink_errors(
    project_root: Path,
    source_root: Path,
) -> tuple[str, ...]:
    candidates = (source_root, *source_root.rglob("*"))
    return tuple(
        "structural source contains unsupported symlink: "
        + path.relative_to(project_root).as_posix()
        for path in sorted(candidates)
        if path.is_symlink()
    )


def source_structure_errors(
    project_root: Path,
    policy: Mapping[str, object],
) -> tuple[str, ...]:
    source_root = project_root / str(policy["source_root"])
    maximum_function_lines = int(cast(int, policy["maximum_function_lines"]))
    module_review_threshold = int(cast(int, policy["module_review_threshold"]))
    reviewed_ceilings = _reviewed_module_ceilings(policy)
    if not source_root.is_dir():
        return (f"structural source root does not exist: {source_root}",)

    symlink_errors = _source_symlink_errors(project_root, source_root)
    if symlink_errors:
        return symlink_errors
    errors: list[str] = []
    module_lines: dict[str, int] = {}
    for source_path in sorted(source_root.rglob("*.py")):
        relative_path = source_path.relative_to(project_root).as_posix()
        source = source_path.read_text(encoding="utf-8")
        line_count = len(source.splitlines())
        module_lines[relative_path] = line_count
        errors.extend(
            _module_structure_errors(
                relative_path,
                line_count,
                module_review_threshold,
                reviewed_ceilings.get(relative_path),
            )
        )
        errors.extend(
            _function_structure_errors(
                relative_path,
                source,
                maximum_function_lines,
            )
        )

    errors.extend(
        _review_configuration_errors(
            module_lines,
            reviewed_ceilings,
            module_review_threshold,
        )
    )
    return tuple(errors)


def _allowed_structural_findings(
    policy: Mapping[str, object],
) -> tuple[StructuralFinding, ...]:
    configured = cast(
        Sequence[Mapping[str, object]],
        policy.get("allowed_findings", ()),
    )
    return tuple(
        (str(item["path"]), str(item["code"]), str(item["function"]))
        for item in configured
    )


def _structure_python_files(
    project_root: Path,
    policy: Mapping[str, object],
) -> tuple[Path, ...]:
    source_root = project_root / str(policy["source_root"])
    return tuple(sorted(source_root.rglob("*.py")))


def _structure_ruff_command(
    policy: Mapping[str, object],
    source_paths: Sequence[Path],
) -> list[str]:
    rules = cast(Sequence[str], policy["ruff_rules"])
    return [
        sys.executable,
        "-I",
        "-m",
        "ruff",
        "check",
        "--isolated",
        "--target-version",
        "py311",
        "--no-cache",
        "--ignore-noqa",
        "--output-format=json",
        "--select",
        ",".join(rules),
        "--config",
        f"lint.mccabe.max-complexity={policy['maximum_complexity']}",
        "--config",
        f"lint.pylint.max-branches={policy['maximum_branches']}",
        "--config",
        f"lint.pylint.max-returns={policy['maximum_returns']}",
        "--config",
        f"lint.pylint.max-statements={policy['maximum_statements']}",
        "--config",
        f"lint.pylint.max-args={policy['maximum_arguments']}",
        "--",
        *(str(path) for path in source_paths),
    ]


def run_structure(
    manifest: Mapping[str, Any],
    *,
    project_root: Path = PROJECT_ROOT,
) -> int:
    policy = cast(dict[str, object], manifest["structure"])
    preflight_errors = [
        *structure_policy_errors(policy),
        *source_structure_errors(project_root, policy),
    ]
    if preflight_errors:
        for error in preflight_errors:
            print(f"Model Runtime structure gate failed: {error}", file=sys.stderr)
        return 1

    source_paths = _structure_python_files(project_root, policy)
    completed = subprocess.run(
        _structure_ruff_command(policy, source_paths),
        cwd=project_root,
        env=_python_environment(project_root),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode not in (0, 1):
        print(completed.stdout, end="", file=sys.stderr)
        print(completed.stderr, end="", file=sys.stderr)
        return completed.returncode
    try:
        diagnostics = cast(list[Mapping[str, Any]], json.loads(completed.stdout))
        findings = ruff_finding_keys(diagnostics, project_root)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"Model Runtime structure gate could not parse Ruff output: {exc}")
        return 1

    errors = structural_finding_errors(
        findings,
        _allowed_structural_findings(policy),
    )
    if errors:
        for error in errors:
            print(f"Model Runtime structure gate failed: {error}", file=sys.stderr)
        return 1
    print(
        "Model Runtime structure gate passed: "
        f"{len(findings)} reviewed Ruff findings, no oversized functions."
    )
    return 0


def run_coverage(manifest: Mapping[str, Any]) -> int:
    coverage_policy = cast(dict[str, Any], manifest["coverage"])
    tests = cast(list[str], coverage_policy["tests"])
    coverage_source = str(coverage_policy["source"])
    with tempfile.TemporaryDirectory(prefix="model-runtime-coverage-") as directory:
        temporary_root = Path(directory)
        report_path = temporary_root / "coverage.json"
        environment = _python_environment()
        environment["COVERAGE_FILE"] = str(temporary_root / ".coverage")
        environment["COVERAGE_RCFILE"] = str(COVERAGE_CONFIG_PATH)
        environment["MODEL_RUNTIME_COVERAGE"] = "1"

        for command in (
            [sys.executable, "-m", "coverage", "erase"],
            [
                sys.executable,
                "-m",
                "coverage",
                "run",
                "--branch",
                f"--source={coverage_source}",
                "-m",
                "unittest",
                *tests,
            ],
            [
                sys.executable,
                "-m",
                "coverage",
                "json",
                "-o",
                str(report_path),
            ],
        ):
            return_code = _run(command, environment=environment)
            if return_code != 0:
                return return_code

        payload = json.loads(report_path.read_text(encoding="utf-8"))
        totals = cast(dict[str, object], payload["totals"])
        errors = coverage_gate_errors(totals, coverage_policy)
        report_code = _run(
            [
                sys.executable,
                "-m",
                "coverage",
                "report",
                f"--include=src/{coverage_source.replace('.', '/')}/*",
            ],
            environment=environment,
        )
        if report_code != 0:
            return report_code
        if errors:
            for error in errors:
                print(f"Model Runtime coverage gate failed: {error}", file=sys.stderr)
            return 1
        statement_percent = float(
            cast(int | float, totals["percent_statements_covered"])
        )
        branch_percent = float(cast(int | float, totals["percent_branches_covered"]))
        print(
            "Model Runtime coverage gate passed: "
            f"statements={statement_percent:.2f}%, "
            f"branches={branch_percent:.2f}%."
        )
        return 0


@contextmanager
def _isolated_mutation_project() -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix="model-runtime-mutation-") as directory:
        isolated_root = Path(directory)
        isolated_src = isolated_root / "src"
        isolated_src.mkdir()
        for package_name in ("model_runtime", "emperor", "models"):
            (isolated_src / package_name).symlink_to(
                PROJECT_ROOT / "src" / package_name,
                target_is_directory=True,
            )
        (isolated_root / "tests").symlink_to(
            PROJECT_ROOT / "tests",
            target_is_directory=True,
        )
        (isolated_root / "tools").symlink_to(
            PROJECT_ROOT / "tools",
            target_is_directory=True,
        )
        for relative_path in (
            ".python-version",
            "mise.toml",
            "pyproject.toml",
            "pyrightconfig.json",
        ):
            isolated_path = isolated_root / relative_path
            isolated_path.parent.mkdir(parents=True, exist_ok=True)
            isolated_path.symlink_to(PROJECT_ROOT / relative_path)
        previous_directory = Path.cwd()
        os.chdir(isolated_root)
        try:
            yield isolated_root
        finally:
            os.chdir(previous_directory)


def run_mutation(manifest: Mapping[str, Any], *, max_children: int) -> int:
    mutation_policy = cast(dict[str, Any], manifest["mutation"])
    for variable_name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ[variable_name] = "1"
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("IPYTHONDIR", "/tmp/ipython")
    os.environ["MODEL_RUNTIME_QUALITY_SOURCE_ROOT"] = str(PROJECT_ROOT)

    with _isolated_mutation_project() as isolated_root:
        os.environ["PYTHONPATH"] = os.pathsep.join(
            (
                str(isolated_root / "src"),
                str(isolated_root / "tests"),
                str(isolated_root),
            )
        )
        from mutmut import configuration

        configuration.Config.reset()
        base_config = configuration.Config.get()
        configuration.__dict__["_config"] = replace(
            base_config,
            source_paths=[Path(mutation_policy["source_root"])],
            only_mutate=list(mutation_policy["critical_sources"]),
            do_not_mutate=["*/__init__.py"],
            also_copy=[
                Path(".python-version"),
                Path("src/emperor"),
                Path("src/models"),
                Path("tests"),
                Path("mise.toml"),
                Path("pyproject.toml"),
                Path("pyrightconfig.json"),
                Path("tools"),
            ],
            pytest_add_cli_args_test_selection=list(mutation_policy["tests"]),
            mutate_only_covered_lines=False,
        )

        from mutmut import __main__ as mutmut_main

        selectors = tuple(cast(list[str], mutation_policy["selectors"]))
        run_mutants = cast(
            Callable[[list[str], int], None],
            mutmut_main.__dict__["_run"],
        )
        run_mutants(list(selectors), max_children)
        mutants, _ = mutmut_main.collect_source_file_mutation_data(
            mutant_names=list(selectors)
        )
        results = tuple(
            sorted(
                (
                    mutant_name,
                    (
                        mutmut_main.status_by_exit_code[result]
                        if result is not None
                        else "not checked"
                    ),
                )
                for _, mutant_name, result in mutants
            )
        )
        errors = mutation_gate_errors(results, selectors)
        if errors:
            for error in errors:
                print(error, file=sys.stderr)
            print(
                "Model Runtime mutation gate failed for the selected "
                f"{len(results)} mutants.",
                file=sys.stderr,
            )
            return 1
        print(f"Model Runtime mutation gate passed: {len(results)} mutants killed.")
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run dedicated Model Runtime quality gates."
    )
    parser.add_argument(
        "mode",
        choices=("tests", "coverage", "mutation", "structure"),
    )
    parser.add_argument("--max-children", type=int, default=4)
    arguments = parser.parse_args()
    manifest = load_manifest()
    if arguments.mode == "tests":
        return run_tests(manifest)
    if arguments.mode == "coverage":
        return run_coverage(manifest)
    if arguments.mode == "structure":
        return run_structure(manifest)
    if arguments.max_children < 1:
        parser.error("--max-children must be positive")
    return run_mutation(manifest, max_children=arguments.max_children)


if __name__ == "__main__":
    raise SystemExit(main())

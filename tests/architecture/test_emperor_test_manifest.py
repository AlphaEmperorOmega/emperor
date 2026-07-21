from __future__ import annotations

import ast
import csv
import tomllib
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EMPEROR_ROOT = PROJECT_ROOT / "src" / "emperor"
MANIFEST_PATH = Path(__file__).with_name("emperor_test_manifest.toml")
LEDGER_PATH = (
    PROJECT_ROOT / ".scratch" / "emperor-behavioral-testing" / "module-ledger.csv"
)
VALID_STATUSES = {"pending", "partial", "complete", "blocked"}
VALID_MODULE_STATUSES = VALID_STATUSES | {"not_applicable"}
LEDGER_COLUMNS = (
    "production_path",
    "family",
    "symbols",
    "contract_and_branches",
    "existing_test_evidence",
    "missing_tests",
    "statement_coverage",
    "branch_coverage",
    "mutation_result",
    "reviewer_result",
    "status",
    "justification",
)


def _load_manifest() -> dict[str, object]:
    with MANIFEST_PATH.open("rb") as manifest_file:
        return tomllib.load(manifest_file)


def _load_evidence(relative_path: str) -> dict[str, object]:
    with (PROJECT_ROOT / relative_path).open("rb") as evidence_file:
        return tomllib.load(evidence_file)


def _mutation_evidence_errors(
    family_name: str,
    mutation: dict[str, object],
) -> tuple[str, ...]:
    errors: list[str] = []
    if mutation.get("family") != family_name or mutation.get("result") != "pass":
        errors.append(f"{family_name} mutation evidence did not pass")

    total = mutation.get("total_mutants")
    killed = mutation.get("killed_mutants")
    surviving = mutation.get("surviving_mutants")
    if not all(isinstance(value, int) for value in (total, killed, surviving)):
        errors.append(f"{family_name} mutation evidence has invalid counts")
    elif killed + surviving != total:
        errors.append(f"{family_name} mutation evidence counts do not balance")

    if mutation.get("unreviewed_mutants") != 0:
        errors.append(f"{family_name} mutation evidence has unreviewed mutants")

    equivalents = mutation.get("equivalent", ())
    if not isinstance(equivalents, list):
        errors.append(f"{family_name} mutation equivalents are malformed")
        equivalents = []
    equivalent_names: list[str] = []
    for equivalent in equivalents:
        if not isinstance(equivalent, dict):
            errors.append(f"{family_name} mutation equivalent is malformed")
            continue
        name = equivalent.get("name")
        justification = equivalent.get("justification")
        if not isinstance(name, str) or not name:
            errors.append(f"{family_name} mutation equivalent has no name")
        else:
            equivalent_names.append(name)
        if not isinstance(justification, str) or not justification:
            errors.append(f"{family_name} mutation equivalent has no justification")
    if len(equivalent_names) != len(set(equivalent_names)):
        errors.append(f"{family_name} mutation equivalents contain duplicates")
    if isinstance(surviving, int) and surviving != len(equivalent_names):
        errors.append(f"{family_name} surviving mutants are not fully classified")
    return tuple(errors)


def _family_completion_errors(
    family_name: str,
    family: dict[str, object],
) -> tuple[str, ...]:
    if family["status"] != "complete":
        return ()

    errors: list[str] = []
    evidence_fields = (
        ("coverage_evidence", "coverage"),
        ("mutation_evidence", "mutation"),
        ("review_evidence", "reviewer"),
    )
    evidence: dict[str, dict[str, object]] = {}
    for field_name, label in evidence_fields:
        relative_path = family.get(field_name)
        if not isinstance(relative_path, str) or not relative_path:
            errors.append(f"{family_name} has no {label} evidence")
            continue
        evidence_path = PROJECT_ROOT / relative_path
        if not evidence_path.is_file():
            errors.append(f"{family_name} {label} evidence does not exist")
            continue
        evidence[label] = _load_evidence(relative_path)

    coverage = evidence.get("coverage")
    if coverage is not None and (
        coverage.get("family") != family_name
        or coverage.get("result") != "pass"
        or coverage.get("statements") != 100
        or coverage.get("branches") != 100
    ):
        errors.append(f"{family_name} coverage evidence is not complete")

    mutation = evidence.get("mutation")
    if mutation is not None:
        errors.extend(_mutation_evidence_errors(family_name, mutation))

    review = evidence.get("reviewer")
    if review is not None and (
        review.get("family") != family_name
        or review.get("verdict") != "approved"
        or not review.get("reviewer")
        or review.get("reviewer") == review.get("implementer")
        or sorted(review.get("reviewed_modules", ()))
        != sorted(family.get("modules", ()))
    ):
        errors.append(f"{family_name} reviewer evidence is not independent")

    return tuple(errors)


def _emperor_mock_violations(source: str, test_path: str) -> tuple[str, ...]:
    tree = ast.parse(source, test_path)
    mock_constructors: dict[str, str] = {}
    patch_functions: set[str] = set()
    mock_modules: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "unittest.mock":
            for alias in node.names:
                local_name = alias.asname or alias.name
                if alias.name in {"Mock", "MagicMock"}:
                    mock_constructors[local_name] = alias.name
                elif alias.name == "patch":
                    patch_functions.add(local_name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "unittest.mock":
                    mock_modules.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module == "unittest":
            for alias in node.names:
                if alias.name == "mock":
                    mock_modules.add(alias.asname or alias.name)

    violations: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id in mock_constructors:
            constructor = mock_constructors[node.func.id]
            violations.append(
                (node.lineno, f"{test_path}:{node.lineno} uses {constructor}")
            )
            continue

        is_patch = isinstance(node.func, ast.Name) and node.func.id in patch_functions
        if isinstance(node.func, ast.Attribute):
            is_patch = is_patch or (
                node.func.attr == "patch"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in mock_modules
            )
            is_direct_patch_object = (
                node.func.attr == "object"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in patch_functions
            )
            is_module_patch_object = (
                node.func.attr == "object"
                and isinstance(node.func.value, ast.Attribute)
                and node.func.value.attr == "patch"
                and isinstance(node.func.value.value, ast.Name)
                and node.func.value.value.id in mock_modules
            )
            if is_direct_patch_object or is_module_patch_object:
                violations.append(
                    (node.lineno, f"{test_path}:{node.lineno} uses patch.object")
                )
                continue
        if not is_patch or not node.args:
            continue
        target = node.args[0]
        if (
            isinstance(target, ast.Constant)
            and isinstance(target.value, str)
            and (target.value == "emperor" or target.value.startswith("emperor."))
        ):
            violations.append(
                (
                    node.lineno,
                    f"{test_path}:{node.lineno} patches {target.value}",
                )
            )

    return tuple(message for _, message in sorted(violations))


class EmperorTestManifestTests(unittest.TestCase):
    def test_manifest_registers_every_production_module_exactly_once(self) -> None:
        manifest = _load_manifest()

        registered_modules = [
            module_path
            for family in manifest["families"].values()
            for module_path in family["modules"]
        ]
        production_modules = sorted(
            path.relative_to(PROJECT_ROOT).as_posix()
            for path in EMPEROR_ROOT.rglob("*.py")
        )

        self.assertEqual(len(registered_modules), len(set(registered_modules)))
        self.assertEqual(sorted(registered_modules), production_modules)

    def test_each_family_records_runnable_test_and_environment_metadata(self) -> None:
        manifest = _load_manifest()

        for family_name, family in manifest["families"].items():
            with self.subTest(family=family_name):
                self.assertIn(family["status"], VALID_STATUSES)
                self.assertTrue(family["focused_tests"])
                self.assertTrue(family["mutation_scope"])
                self.assertTrue(family["environment"])
                self.assertIn("coverage_evidence", family)
                self.assertIn("mutation_evidence", family)
                self.assertIn("review_evidence", family)
                self.assertIn("blocked_reason", family)

                for pattern in family["focused_tests"]:
                    self.assertTrue(
                        tuple(PROJECT_ROOT.glob(pattern)),
                        f"{family_name} test pattern matched nothing: {pattern}",
                    )
                for relative_path in family["integration_tests"]:
                    self.assertTrue(
                        (PROJECT_ROOT / relative_path).is_file(),
                        f"{family_name} integration test does not exist: "
                        f"{relative_path}",
                    )

    def test_module_ledger_tracks_every_manifest_entry_with_required_fields(
        self,
    ) -> None:
        manifest = _load_manifest()
        expected_families = {
            module_path: family_name
            for family_name, family in manifest["families"].items()
            for module_path in family["modules"]
        }
        with LEDGER_PATH.open(newline="", encoding="utf-8") as ledger_file:
            reader = csv.DictReader(ledger_file)
            self.assertEqual(tuple(reader.fieldnames or ()), LEDGER_COLUMNS)
            rows = list(reader)

        actual_paths = [row["production_path"] for row in rows]
        self.assertEqual(len(actual_paths), len(set(actual_paths)))
        self.assertEqual(set(actual_paths), set(expected_families))
        for row in rows:
            with self.subTest(module=row["production_path"]):
                self.assertEqual(
                    row["family"],
                    expected_families[row["production_path"]],
                )
                self.assertTrue(row["symbols"])
                self.assertTrue(row["contract_and_branches"])
                self.assertTrue(row["existing_test_evidence"])
                self.assertTrue(row["missing_tests"])
                self.assertTrue(row["statement_coverage"])
                self.assertTrue(row["branch_coverage"])
                self.assertTrue(row["mutation_result"])
                self.assertTrue(row["reviewer_result"])
                self.assertIn(row["status"], VALID_MODULE_STATUSES)
                if row["status"] == "not_applicable":
                    self.assertTrue(row["justification"])

    def test_complete_family_requires_all_acceptance_evidence(self) -> None:
        incomplete_family = {
            "status": "complete",
            "focused_tests": ["tests/unit/test_linears.py"],
            "coverage_evidence": "",
            "mutation_evidence": "",
            "review_evidence": "",
        }

        self.assertEqual(
            _family_completion_errors("linears", incomplete_family),
            (
                "linears has no coverage evidence",
                "linears has no mutation evidence",
                "linears has no reviewer evidence",
            ),
        )

    def test_mutation_evidence_accepts_only_fully_reviewed_equivalents(
        self,
    ) -> None:
        evidence = {
            "family": "linears",
            "result": "pass",
            "total_mutants": 497,
            "killed_mutants": 495,
            "surviving_mutants": 2,
            "unreviewed_mutants": 0,
            "equivalent": [
                {
                    "name": "first",
                    "justification": "Same falsey branch value.",
                },
                {
                    "name": "second",
                    "justification": "Private result is never consumed.",
                },
            ],
        }

        self.assertEqual(
            _mutation_evidence_errors("linears", evidence),
            (),
        )

        evidence["unreviewed_mutants"] = 1
        self.assertEqual(
            _mutation_evidence_errors("linears", evidence),
            ("linears mutation evidence has unreviewed mutants",),
        )

    def test_mock_scanner_rejects_emperor_owned_test_doubles(self) -> None:
        source = """\
from unittest import mock
from unittest.mock import MagicMock, Mock, patch

Mock()
MagicMock()
with patch("emperor.linears._layer.F.linear"):
    pass
with mock.patch.object(object(), "attribute"):
    pass
with patch("external.service"):
    pass
"""

        self.assertEqual(
            _emperor_mock_violations(source, "tests/unit/test_example.py"),
            (
                "tests/unit/test_example.py:4 uses Mock",
                "tests/unit/test_example.py:5 uses MagicMock",
                "tests/unit/test_example.py:6 patches emperor.linears._layer.F.linear",
                "tests/unit/test_example.py:8 uses patch.object",
            ),
        )

    def test_manifest_completion_claims_are_evidenced_and_mock_free(self) -> None:
        manifest = _load_manifest()
        errors: list[str] = []

        for family_name, family in manifest["families"].items():
            if family["status"] == "blocked" and not family["blocked_reason"]:
                errors.append(f"{family_name} is blocked without a reason")
            errors.extend(_family_completion_errors(family_name, family))
            if family["status"] != "complete":
                continue
            for pattern in family["focused_tests"]:
                for test_path in sorted(PROJECT_ROOT.glob(pattern)):
                    relative_path = test_path.relative_to(PROJECT_ROOT).as_posix()
                    errors.extend(
                        _emperor_mock_violations(
                            test_path.read_text(encoding="utf-8"),
                            relative_path,
                        )
                    )

        self.assertEqual(errors, [])

    def test_quality_configuration_uses_manifest_driven_family_commands(
        self,
    ) -> None:
        with (PROJECT_ROOT / "pyproject.toml").open("rb") as project_file:
            project = tomllib.load(project_file)
        with (PROJECT_ROOT / "mise.toml").open("rb") as mise_file:
            mise = tomllib.load(mise_file)

        self.assertEqual(project["tool"]["coverage"]["run"]["source"], ["emperor"])
        mutation = project["tool"]["mutmut"]
        self.assertEqual(mutation["source_paths"], ["src/emperor/"])
        self.assertEqual(mutation["do_not_mutate"], ["*/__init__.py"])
        self.assertNotIn("only_mutate", mutation)
        self.assertNotIn("pytest_add_cli_args_test_selection", mutation)

        expected_tasks = {
            "test:family": "python tools/emperor_test_family.py tests",
            "test:family-coverage": "python tools/emperor_test_family.py coverage",
            "test:family-mutation": "python tools/emperor_test_family.py mutation",
        }
        for task_name, command in expected_tasks.items():
            with self.subTest(task=task_name):
                self.assertTrue(mise["tasks"][task_name]["raw"])
                self.assertEqual(mise["tasks"][task_name]["run"], command)


if __name__ == "__main__":
    unittest.main()

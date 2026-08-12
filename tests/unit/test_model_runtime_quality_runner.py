from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_quality_runner():
    name = "_model_runtime_quality_runner_tests"
    spec = importlib.util.spec_from_file_location(
        name,
        PROJECT_ROOT / "tools/model_runtime_quality.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the Model Runtime quality runner.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


quality = _load_quality_runner()


class ModelRuntimeQualityRunnerTests(unittest.TestCase):
    def test_coverage_gate_checks_statement_and_branch_floors_independently(
        self,
    ) -> None:
        policy = {
            "minimum_statement_percent": 88.0,
            "minimum_branch_percent": 75.0,
        }

        self.assertEqual(
            quality.coverage_gate_errors(
                {
                    "percent_statements_covered": 87.99,
                    "percent_branches_covered": 74.99,
                },
                policy,
            ),
            (
                "statement coverage 87.99% is below 88.00%",
                "branch coverage 74.99% is below 75.00%",
            ),
        )
        self.assertEqual(
            quality.coverage_gate_errors(
                {
                    "percent_statements_covered": 88.0,
                    "percent_branches_covered": 75.0,
                },
                policy,
            ),
            (),
        )

    def test_mutation_gate_rejects_empty_unmatched_and_surviving_results(self) -> None:
        selectors = ("*planning*", "*preflight*")

        self.assertIn(
            "no selected mutants",
            quality.mutation_gate_errors((), selectors)[0],
        )
        errors = quality.mutation_gate_errors(
            (("model_runtime.runs.planning.mutant", "survived"),),
            selectors,
        )
        self.assertTrue(any("selector matched no mutant" in error for error in errors))
        self.assertTrue(any("survived" in error for error in errors))
        self.assertEqual(
            quality.mutation_gate_errors(
                (
                    ("model_runtime.runs.planning.mutant", "killed"),
                    ("model_runtime.inspection.preflight.mutant", "killed"),
                ),
                selectors,
            ),
            (),
        )
        self.assertEqual(
            quality.mutation_gate_errors(
                (
                    ("model_runtime.runs.planning.first", "killed"),
                    ("model_runtime.runs.planning.second", "killed"),
                ),
                ("*planning*",),
            ),
            ("selector matched 2 mutants: *planning*",),
        )
        self.assertEqual(
            quality.mutation_gate_errors(
                (("model_runtime.runs.planning.mutant", "killed"),),
                ("*planning*", "*planning*"),
            ),
            ("duplicate mutation selector: *planning*",),
        )

    def test_structural_finding_gate_requires_an_exact_allowlist(self) -> None:
        allowed = (
            (
                "src/model_runtime/runs/execution.py",
                "PLR0913",
                "execute_runs",
            ),
        )

        self.assertEqual(quality.structural_finding_errors(allowed, allowed), ())
        self.assertEqual(
            quality.structural_finding_errors(
                (
                    *allowed,
                    (
                        "src/model_runtime/inspection/schema.py",
                        "C901",
                        "configuration_schema",
                    ),
                ),
                allowed,
            ),
            (
                "unexpected structural finding: "
                "src/model_runtime/inspection/schema.py:C901:configuration_schema",
            ),
        )
        self.assertEqual(
            quality.structural_finding_errors((), allowed),
            (
                "allowed structural finding is no longer present: "
                "src/model_runtime/runs/execution.py:PLR0913:execute_runs",
            ),
        )
        self.assertEqual(
            quality.structural_finding_errors((*allowed, *allowed), allowed),
            (
                "duplicate structural finding: "
                "src/model_runtime/runs/execution.py:PLR0913:execute_runs "
                "(2 occurrences)",
            ),
        )
        self.assertEqual(
            quality.structural_finding_errors(allowed, (*allowed, *allowed)),
            (
                "duplicate allowed structural finding: "
                "src/model_runtime/runs/execution.py:PLR0913:execute_runs "
                "(2 entries)",
            ),
        )

    def test_source_structure_gate_enforces_function_and_module_limits(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory)
            source_root = project_root / "src/model_runtime"
            source_root.mkdir(parents=True)
            source_path = source_root / "oversized.py"
            source_path.write_text(
                "def oversized():\n"
                "    first = 1\n"
                "    second = 2\n"
                "    third = 3\n"
                "    return first + second + third\n",
                encoding="utf-8",
            )
            policy: dict[str, object] = {
                "source_root": "src/model_runtime",
                "maximum_function_lines": 4,
                "module_review_threshold": 4,
                "reviewed_modules": [],
            }

            self.assertEqual(
                quality.source_structure_errors(project_root, policy),
                (
                    "src/model_runtime/oversized.py has 5 lines and requires "
                    "cohesion review above 4",
                    "src/model_runtime/oversized.py:1 oversized has 5 lines "
                    "(maximum 4)",
                ),
            )

            policy["maximum_function_lines"] = 5
            policy["reviewed_modules"] = [
                {
                    "path": "src/model_runtime/oversized.py",
                    "maximum_lines": 5,
                    "rationale": "One cohesive fixture module.",
                }
            ]
            self.assertEqual(
                quality.source_structure_errors(project_root, policy),
                (),
            )

            policy["reviewed_modules"][0]["maximum_lines"] = 4
            self.assertEqual(
                quality.source_structure_errors(project_root, policy),
                (
                    "src/model_runtime/oversized.py has 5 lines, exceeding "
                    "reviewed ceiling 4",
                ),
            )

    def test_source_structure_gate_rejects_stale_module_reviews(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory)
            source_path = project_root / "src/model_runtime/small.py"
            source_path.parent.mkdir(parents=True)
            source_path.write_text("VALUE = 1\n", encoding="utf-8")

            self.assertEqual(
                quality.source_structure_errors(
                    project_root,
                    {
                        "source_root": "src/model_runtime",
                        "maximum_function_lines": 60,
                        "module_review_threshold": 600,
                        "reviewed_modules": [
                            {
                                "path": "src/model_runtime/small.py",
                                "maximum_lines": 1,
                                "rationale": "No longer needed.",
                            }
                        ],
                    },
                ),
                (
                    "src/model_runtime/small.py is reviewed but has only 1 lines "
                    "(review threshold 600)",
                ),
            )

    def test_source_structure_gate_rejects_symlinked_entries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory)
            source_root = project_root / "src/model_runtime"
            source_root.mkdir(parents=True)
            (source_root / "clean.py").write_text("VALUE = 1\n", encoding="utf-8")
            external = project_root / "external"
            external.mkdir()
            (external / "hidden.py").write_text(
                "def too_many(first, second, third, fourth, fifth, sixth):\n"
                "    return None\n",
                encoding="utf-8",
            )
            (source_root / "linked").symlink_to(
                external,
                target_is_directory=True,
            )

            policy: dict[str, object] = {
                "source_root": "src/model_runtime",
                "maximum_function_lines": 60,
                "module_review_threshold": 600,
                "reviewed_modules": [],
            }
            self.assertEqual(
                quality.source_structure_errors(project_root, policy),
                (
                    "structural source contains unsupported symlink: "
                    "src/model_runtime/linked",
                ),
            )
            with (
                patch.object(quality.subprocess, "run") as run,
                redirect_stderr(StringIO()),
            ):
                self.assertEqual(
                    quality.run_structure(
                        {"structure": policy},
                        project_root=project_root,
                    ),
                    1,
                )
            run.assert_not_called()

    def test_structure_policy_requires_rationales_and_unique_reviews(self) -> None:
        policy = {
            "allowed_findings": [
                {
                    "path": "src/model_runtime/api.py",
                    "code": "PLR0913",
                    "function": "api",
                    "rationale": "",
                }
            ],
            "reviewed_modules": [
                {
                    "path": "src/model_runtime/large.py",
                    "maximum_lines": 700,
                    "rationale": "First review.",
                },
                {
                    "path": "src/model_runtime/large.py",
                    "maximum_lines": 750,
                    "rationale": "",
                },
            ],
        }

        self.assertEqual(
            quality.structure_policy_errors(policy),
            (
                "allowed structural finding requires a rationale: "
                "src/model_runtime/api.py:PLR0913:api",
                "reviewed module requires a rationale: src/model_runtime/large.py",
                "duplicate reviewed module: src/model_runtime/large.py (2 entries)",
            ),
        )

    def test_ruff_findings_resolve_qualified_function_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory)
            source_path = project_root / "src/model_runtime/sample.py"
            source_path.parent.mkdir(parents=True)
            source_path.write_text(
                "class Owner:\n"
                "    def method(self, first, second):\n"
                "        return first + second\n",
                encoding="utf-8",
            )
            diagnostics = [
                {
                    "code": "PLR0913",
                    "filename": str(source_path),
                    "location": {"row": 2},
                }
            ]

            self.assertEqual(
                quality.ruff_finding_keys(diagnostics, project_root),
                (
                    (
                        "src/model_runtime/sample.py",
                        "PLR0913",
                        "Owner.method",
                    ),
                ),
            )

    def test_function_spans_include_decorators_and_nested_qualification(
        self,
    ) -> None:
        source = (
            "@outer_decorator(\n"
            "    1,\n"
            ")\n"
            "def outer():\n"
            "    @inner_decorator\n"
            "    def inner():\n"
            "        return 1\n"
            "    return inner()\n"
        )

        self.assertEqual(
            quality._function_spans(source),
            (
                ("outer", 1, 8),
                ("outer.inner", 5, 7),
            ),
        )

    def test_structure_gate_scans_default_and_gitignored_directories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory)
            source_root = project_root / "src/model_runtime"
            source_root.mkdir(parents=True)
            (project_root / ".gitignore").write_text(
                "src/model_runtime/ignored/\n",
                encoding="utf-8",
            )
            fake_ruff = project_root / "src/ruff"
            fake_ruff.mkdir()
            (fake_ruff / "__init__.py").write_text("", encoding="utf-8")
            (fake_ruff / "__main__.py").write_text(
                'print("[]")\n',
                encoding="utf-8",
            )
            ignored_sources = (
                source_root / ".venv/hidden.py",
                source_root / "node_modules/hidden.py",
                source_root / "dist/hidden.py",
                source_root / "ignored/hidden.py",
            )
            for source_path in ignored_sources:
                source_path.parent.mkdir(parents=True, exist_ok=True)
                source_path.write_text(
                    "def too_many(first, second, third, fourth, fifth, sixth):\n"
                    "    return None\n",
                    encoding="utf-8",
                )
            policy = {
                "source_root": "src/model_runtime",
                "ruff_rules": [
                    "C901",
                    "PLR0911",
                    "PLR0912",
                    "PLR0913",
                    "PLR0915",
                ],
                "maximum_complexity": 10,
                "maximum_branches": 12,
                "maximum_returns": 6,
                "maximum_statements": 50,
                "maximum_arguments": 5,
                "maximum_function_lines": 60,
                "module_review_threshold": 600,
                "allowed_findings": [],
                "reviewed_modules": [],
            }
            errors = StringIO()

            with redirect_stderr(errors):
                result = quality.run_structure(
                    {"structure": policy},
                    project_root=project_root,
                )

            self.assertEqual(result, 1)
            for source_path in ignored_sources:
                relative_path = source_path.relative_to(project_root).as_posix()
                with self.subTest(source=relative_path):
                    self.assertIn(
                        f"{relative_path}:PLR0913:too_many",
                        errors.getvalue(),
                    )

    def test_structure_command_pins_thresholds_and_disables_suppressions(
        self,
    ) -> None:
        policy = quality.load_manifest()["structure"]
        source_paths = quality._structure_python_files(PROJECT_ROOT, policy)

        command = quality._structure_ruff_command(policy, source_paths)

        self.assertIn("--isolated", command)
        self.assertIn("--no-cache", command)
        self.assertIn("--ignore-noqa", command)
        self.assertEqual(command[1:3], ["-I", "-m"])
        source_separator = command.index("--")
        self.assertEqual(
            command[source_separator + 1 :],
            [str(path) for path in source_paths],
        )
        self.assertEqual(
            {
                command[index + 1]
                for index, value in enumerate(command)
                if value == "--config"
            },
            {
                "lint.mccabe.max-complexity=10",
                "lint.pylint.max-branches=12",
                "lint.pylint.max-returns=6",
                "lint.pylint.max-statements=50",
                "lint.pylint.max-args=5",
            },
        )


if __name__ == "__main__":
    unittest.main()

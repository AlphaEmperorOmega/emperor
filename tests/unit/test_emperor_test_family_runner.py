from __future__ import annotations

import fnmatch
import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_emperor_test_family():
    name = "_emperor_test_family_runner_tests"
    spec = importlib.util.spec_from_file_location(
        name,
        PROJECT_ROOT / "tools" / "emperor_test_family.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the Emperor test-family runner.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


emperor_test_family = _load_emperor_test_family()
cached_mutant_names_to_run = emperor_test_family.cached_mutant_names_to_run
classify_mutation_results = emperor_test_family.classify_mutation_results
coverage_include_argument = emperor_test_family.coverage_include_argument
documented_equivalent_mutants = emperor_test_family.documented_equivalent_mutants
load_family = emperor_test_family.load_family
mutant_names_to_run = emperor_test_family.mutant_names_to_run
pending_mutant_names = emperor_test_family.pending_mutant_names
resolve_test_paths = emperor_test_family.resolve_test_paths
resolve_test_module_names = emperor_test_family.test_module_names
reset_mutation_stats_cache = emperor_test_family.reset_mutation_stats_cache
reset_family_mutant_sources = emperor_test_family.reset_family_mutant_sources
restore_cached_exit_codes = emperor_test_family.restore_cached_exit_codes
selected_mutant_names = emperor_test_family.selected_mutant_names
should_reset_mutation_cache = emperor_test_family.should_reset_mutation_cache


class EmperorTestFamilyRunnerTests(unittest.TestCase):
    def test_python_environment_prefers_src_and_disables_implicit_cwd_imports(
        self,
    ) -> None:
        environment = emperor_test_family._python_environment(PROJECT_ROOT)

        self.assertEqual(environment["PYTHONSAFEPATH"], "1")
        self.assertEqual(
            environment["PYTHONPATH"].split(os.pathsep)[:3],
            [
                str(PROJECT_ROOT / "src"),
                str(PROJECT_ROOT / "tests"),
                str(PROJECT_ROOT),
            ],
        )

    def test_family_modules_include_deduplicated_integration_tests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            focused = root / "tests" / "unit" / "test_family.py"
            integration = root / "tests" / "integration" / "test_family_runtime.py"
            focused.parent.mkdir(parents=True)
            integration.parent.mkdir(parents=True)
            focused.touch()
            integration.touch()
            family = {
                "focused_tests": ["tests/unit/test_family.py"],
                "integration_tests": [
                    "tests/unit/test_family.py",
                    "tests/integration/test_family_runtime.py",
                ],
            }

            modules = emperor_test_family._family_test_modules(root, family)

        self.assertEqual(
            modules,
            (
                "tests.integration.test_family_runtime",
                "tests.unit.test_family",
            ),
        )

    def test_resolved_test_files_convert_to_unittest_module_names(self) -> None:
        paths = (
            PROJECT_ROOT / "tests" / "unit" / "test_linears.py",
            PROJECT_ROOT / "tests" / "integration" / "test_linear_monitor_lifecycle.py",
        )

        self.assertEqual(
            resolve_test_module_names(PROJECT_ROOT, paths),
            (
                "tests.unit.test_linears",
                "tests.integration.test_linear_monitor_lifecycle",
            ),
        )

    def test_coverage_include_argument_uses_only_registered_family_modules(
        self,
    ) -> None:
        family = load_family(PROJECT_ROOT, "linears")

        include_argument = coverage_include_argument(family["modules"])

        self.assertEqual(
            include_argument,
            "--include=" + ",".join(family["modules"]),
        )
        self.assertNotIn("src/emperor/attention/", include_argument)

    def test_mutation_classification_allows_only_exact_documented_equivalents(
        self,
    ) -> None:
        results = (
            ("killed-mutant", "killed"),
            ("reviewed-equivalent", "survived"),
            ("unexpected-survivor", "survived"),
        )

        unexpected, stale = classify_mutation_results(
            results,
            {"reviewed-equivalent", "stale-equivalent"},
        )

        self.assertEqual(
            unexpected,
            (("unexpected-survivor", "survived"),),
        )
        self.assertEqual(stale, ("stale-equivalent",))

    def test_mutation_cache_resets_only_for_a_fresh_full_run(self) -> None:
        self.assertTrue(
            should_reset_mutation_cache(
                mutant_names=(),
                resume=False,
            )
        )
        self.assertFalse(
            should_reset_mutation_cache(
                mutant_names=(),
                resume=True,
            )
        )
        self.assertFalse(
            should_reset_mutation_cache(
                mutant_names=("module__mutmut_1",),
                resume=False,
            )
        )
        self.assertFalse(
            should_reset_mutation_cache(
                mutant_names=("module__mutmut_1",),
                resume=True,
            )
        )

    def test_neuron_mutation_excludes_only_empty_private_initializers(self) -> None:
        family = load_family(PROJECT_ROOT, "neuron")

        exclusions = emperor_test_family.mutation_exclusion_patterns(
            family,
            default_patterns=("*/__init__.py",),
        )

        self.assertEqual(
            exclusions,
            (
                "src/emperor/neuron/_cluster/__init__.py",
                "src/emperor/neuron/_monitoring/__init__.py",
            ),
        )
        public_initializer = "src/emperor/neuron/__init__.py"
        self.assertTrue(
            any(
                fnmatch.fnmatchcase(public_initializer, pattern)
                for pattern in family["mutation_scope"]
            )
        )
        self.assertFalse(
            any(
                fnmatch.fnmatchcase(public_initializer, pattern)
                for pattern in exclusions
            )
        )

    def test_default_mutation_exclusions_apply_without_family_override(self) -> None:
        family = load_family(PROJECT_ROOT, "linears")

        exclusions = emperor_test_family.mutation_exclusion_patterns(
            family,
            default_patterns=("*/__init__.py",),
        )

        self.assertEqual(exclusions, ("*/__init__.py",))

    def test_pending_mutant_names_select_only_unchecked_cache_entries(self) -> None:
        results = (
            ("killed", 1),
            ("survived", 0),
            ("unchecked-second", None),
            ("unchecked-first", None),
        )

        self.assertEqual(
            pending_mutant_names(results),
            ("unchecked-first", "unchecked-second"),
        )

    def test_pending_mutant_names_rejects_an_absent_cache(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "No cached mutation data exists",
        ):
            pending_mutant_names(())

    def test_resume_keeps_pending_names_when_no_selector_is_given(self) -> None:
        cached_results = (
            ("already-killed", 1),
            ("still-unchecked", None),
        )

        self.assertEqual(
            cached_mutant_names_to_run(
                cached_results,
                (),
                resume=True,
            ),
            ("still-unchecked",),
        )

    def test_cached_exit_codes_restore_only_still_generated_mutants(self) -> None:
        generated = {
            "same-killed": None,
            "same-survived": None,
            "new-mutant": None,
        }
        cached = {
            "same-killed": 1,
            "same-survived": 0,
            "removed-mutant": 1,
        }

        self.assertEqual(
            restore_cached_exit_codes(generated, cached),
            {
                "same-killed": 1,
                "same-survived": 0,
                "new-mutant": None,
            },
        )

    def test_selected_mutants_are_reset_when_other_cached_results_are_restored(
        self,
    ) -> None:
        generated = {
            "same-killed": None,
            "selected-survivor": None,
            "new-mutant": None,
        }
        cached = {
            "same-killed": 1,
            "selected-survivor": 0,
        }

        self.assertEqual(
            restore_cached_exit_codes(
                generated,
                cached,
                rerun_names={"selected-survivor"},
            ),
            {
                "same-killed": 1,
                "selected-survivor": None,
                "new-mutant": None,
            },
        )

    def test_selected_mutant_names_supports_exact_names_and_globs(self) -> None:
        names = {
            "emperor.module.first__mutmut_1",
            "emperor.module.first__mutmut_2",
            "emperor.module.second__mutmut_1",
        }

        self.assertEqual(
            selected_mutant_names(
                names,
                (
                    "emperor.module.first__mutmut_1",
                    "emperor.module.second*",
                ),
            ),
            {
                "emperor.module.first__mutmut_1",
                "emperor.module.second__mutmut_1",
            },
        )

    def test_mutant_names_to_run_expands_globs_to_exact_cached_names(self) -> None:
        available_names = {
            "emperor.module.first__mutmut_1",
            "emperor.module.first__mutmut_2",
            "emperor.module.second__mutmut_1",
        }

        self.assertEqual(
            mutant_names_to_run(
                available_names,
                ("emperor.module.first*",),
            ),
            (
                "emperor.module.first__mutmut_1",
                "emperor.module.first__mutmut_2",
            ),
        )

    def test_mutant_names_to_run_rejects_an_unmatched_selector(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "No cached mutant matches",
        ):
            mutant_names_to_run(
                {"emperor.module.first__mutmut_1"},
                ("emperor.module.absent*",),
            )

    def test_mutation_stats_reset_preserves_per_mutant_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory)
            mutants_root = project_root / "mutants"
            family_cache = mutants_root / "src" / "emperor" / "module.py.meta"
            family_cache.parent.mkdir(parents=True)
            family_cache.write_text("cached mutant result", encoding="utf-8")
            stats_paths = tuple(
                mutants_root / name
                for name in ("mutmut-stats.json", "mutmut-cicd-stats.json")
            )
            for stats_path in stats_paths:
                stats_path.write_text("stale family mapping", encoding="utf-8")

            reset_mutation_stats_cache(project_root)

            self.assertTrue(family_cache.is_file())
            self.assertTrue(all(not path.exists() for path in stats_paths))

    def test_family_source_reset_preserves_per_mutant_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory)
            generated_source = (
                project_root / "mutants" / "src" / "emperor" / "module.py"
            )
            generated_metadata = Path(f"{generated_source}.meta")
            generated_source.parent.mkdir(parents=True)
            generated_source.write_text("plain copied source", encoding="utf-8")
            generated_metadata.write_text("cached results", encoding="utf-8")

            reset_family_mutant_sources(
                project_root,
                {"modules": ["src/emperor/module.py"]},
            )

            self.assertFalse(generated_source.exists())
            self.assertEqual(
                generated_metadata.read_text(encoding="utf-8"),
                "cached results",
            )

    def test_documented_equivalent_mutants_come_from_family_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory)
            evidence_path = project_root / "mutation.toml"
            evidence_path.write_text(
                """\
family = "linears"

[[equivalent]]
name = "first"
justification = "Same observable result."

[[equivalent]]
name = "second"
justification = "Private value is never consumed."
""",
                encoding="utf-8",
            )

            equivalents = documented_equivalent_mutants(
                project_root,
                {
                    "mutation_evidence": "mutation.toml",
                },
            )

        self.assertEqual(equivalents, {"first", "second"})


if __name__ == "__main__":
    unittest.main()

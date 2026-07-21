from __future__ import annotations

import argparse
import fnmatch
import os
import subprocess
import sys
import tomllib
from dataclasses import replace
from pathlib import Path
from typing import Any

MANIFEST_RELATIVE_PATH = Path("tests/architecture/emperor_test_manifest.toml")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_family(project_root: Path, family_name: str) -> dict[str, Any]:
    with (project_root / MANIFEST_RELATIVE_PATH).open("rb") as manifest_file:
        manifest = tomllib.load(manifest_file)
    try:
        return manifest["families"][family_name]
    except KeyError as error:
        available = ", ".join(sorted(manifest["families"]))
        raise ValueError(
            f"Unknown Emperor test family {family_name!r}. Available: {available}"
        ) from error


def resolve_test_paths(
    project_root: Path,
    patterns: list[str],
) -> tuple[Path, ...]:
    resolved = {
        path.resolve()
        for pattern in patterns
        for path in project_root.glob(pattern)
        if path.is_file()
    }
    return tuple(sorted(resolved))


def test_module_names(
    project_root: Path,
    test_paths: tuple[Path, ...],
) -> tuple[str, ...]:
    return tuple(
        ".".join(path.relative_to(project_root).with_suffix("").parts)
        for path in test_paths
    )


def coverage_include_argument(module_paths: list[str]) -> str:
    return "--include=" + ",".join(module_paths)


def mutation_exclusion_patterns(
    family: dict[str, Any],
    *,
    default_patterns: tuple[str, ...] | list[str],
) -> tuple[str, ...]:
    configured_patterns = family.get("mutation_exclusions")
    if configured_patterns is None:
        return tuple(default_patterns)
    if not isinstance(configured_patterns, list) or any(
        not isinstance(pattern, str) or not pattern for pattern in configured_patterns
    ):
        raise ValueError("mutation_exclusions must be a list of non-empty strings.")
    return tuple(configured_patterns)


def classify_mutation_results(
    results: tuple[tuple[str, str], ...],
    documented_equivalents: set[str],
) -> tuple[tuple[tuple[str, str], ...], tuple[str, ...]]:
    surviving_names = {
        mutant_name for mutant_name, status in results if status == "survived"
    }
    unexpected = tuple(
        (mutant_name, status)
        for mutant_name, status in results
        if status != "killed"
        and not (status == "survived" and mutant_name in documented_equivalents)
    )
    stale = tuple(sorted(documented_equivalents - surviving_names))
    return unexpected, stale


def should_reset_mutation_cache(
    *,
    mutant_names: tuple[str, ...],
    resume: bool,
) -> bool:
    return not mutant_names and not resume


def pending_mutant_names(
    results: tuple[tuple[str, int | None], ...],
) -> tuple[str, ...]:
    if not results:
        raise ValueError(
            "No cached mutation data exists; run mutation mode without --resume first."
        )
    return tuple(sorted(name for name, result in results if result is None))


def restore_cached_exit_codes(
    generated: dict[str, int | None],
    cached: dict[str, int],
    *,
    rerun_names: set[str] | None = None,
) -> dict[str, int | None]:
    rerun_names = rerun_names or set()
    return {
        mutant_name: (
            result if mutant_name in rerun_names else cached.get(mutant_name, result)
        )
        for mutant_name, result in generated.items()
    }


def selected_mutant_names(
    mutant_names: set[str],
    patterns: tuple[str, ...],
) -> set[str]:
    return {
        mutant_name
        for mutant_name in mutant_names
        if any(fnmatch.fnmatchcase(mutant_name, pattern) for pattern in patterns)
    }


def mutant_names_to_run(
    available_names: set[str],
    patterns: tuple[str, ...],
) -> tuple[str, ...]:
    unmatched_patterns = tuple(
        pattern
        for pattern in patterns
        if not any(
            fnmatch.fnmatchcase(mutant_name, pattern) for mutant_name in available_names
        )
    )
    if unmatched_patterns:
        selectors = ", ".join(repr(pattern) for pattern in unmatched_patterns)
        raise ValueError(f"No cached mutant matches selector(s): {selectors}.")
    return tuple(sorted(selected_mutant_names(available_names, patterns)))


def cached_mutant_names_to_run(
    cached_results: tuple[tuple[str, int | None], ...],
    selectors: tuple[str, ...],
    *,
    resume: bool,
) -> tuple[str, ...]:
    if selectors:
        return mutant_names_to_run(
            {mutant_name for mutant_name, _ in cached_results},
            selectors,
        )
    if resume:
        return pending_mutant_names(cached_results)
    return ()


def documented_equivalent_mutants(
    project_root: Path,
    family: dict[str, Any],
) -> set[str]:
    relative_path = family.get("mutation_evidence")
    if not isinstance(relative_path, str) or not relative_path:
        return set()
    evidence_path = project_root / relative_path
    if not evidence_path.is_file():
        raise ValueError(f"Mutation evidence does not exist: {relative_path}")
    with evidence_path.open("rb") as evidence_file:
        evidence = tomllib.load(evidence_file)

    equivalents: set[str] = set()
    for entry in evidence.get("equivalent", ()):
        name = entry.get("name")
        justification = entry.get("justification")
        if not isinstance(name, str) or not name:
            raise ValueError("Equivalent mutation evidence has no name.")
        if not isinstance(justification, str) or not justification:
            raise ValueError(f"Equivalent mutation {name!r} has no justification.")
        equivalents.add(name)
    return equivalents


def _python_environment(project_root: Path) -> dict[str, str]:
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


def _family_test_paths(
    project_root: Path,
    family: dict[str, Any],
) -> tuple[Path, ...]:
    patterns = [
        *family["focused_tests"],
        *family.get("integration_tests", []),
    ]
    paths = resolve_test_paths(project_root, patterns)
    if not paths:
        raise ValueError("The family has no resolvable tests.")
    return paths


def _family_test_modules(
    project_root: Path,
    family: dict[str, Any],
) -> tuple[str, ...]:
    return test_module_names(project_root, _family_test_paths(project_root, family))


def run_tests(project_root: Path, family: dict[str, Any]) -> int:
    command = [
        sys.executable,
        "-m",
        "unittest",
        *_family_test_modules(project_root, family),
    ]
    return subprocess.run(
        command,
        cwd=project_root,
        env=_python_environment(project_root),
        check=False,
    ).returncode


def run_coverage(project_root: Path, family: dict[str, Any]) -> int:
    environment = _python_environment(project_root)
    erase = subprocess.run(
        [sys.executable, "-m", "coverage", "erase"],
        cwd=project_root,
        env=environment,
        check=False,
    )
    if erase.returncode != 0:
        return erase.returncode

    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "coverage",
            "run",
            "--branch",
            "--source=emperor",
            "-m",
            "unittest",
            *_family_test_modules(project_root, family),
        ],
        cwd=project_root,
        env=environment,
        check=False,
    )
    if run.returncode != 0:
        return run.returncode

    combine = subprocess.run(
        [sys.executable, "-m", "coverage", "combine"],
        cwd=project_root,
        env=environment,
        check=False,
    )
    if combine.returncode != 0:
        return combine.returncode

    report = subprocess.run(
        [
            sys.executable,
            "-m",
            "coverage",
            "report",
            coverage_include_argument(family["modules"]),
        ],
        cwd=project_root,
        env=environment,
        check=False,
    )
    return report.returncode


def _reset_family_mutation_cache(
    project_root: Path,
    family: dict[str, Any],
) -> None:
    mutants_root = project_root / "mutants"
    for relative_path in family["modules"]:
        mutant_path = mutants_root / relative_path
        for generated_path in (mutant_path, Path(f"{mutant_path}.meta")):
            if generated_path.is_file():
                generated_path.unlink()
    reset_mutation_stats_cache(project_root)


def reset_mutation_stats_cache(project_root: Path) -> None:
    mutants_root = project_root / "mutants"
    for stats_name in ("mutmut-stats.json", "mutmut-cicd-stats.json"):
        stats_path = mutants_root / stats_name
        if stats_path.is_file():
            stats_path.unlink()


def reset_family_mutant_sources(
    project_root: Path,
    family: dict[str, Any],
) -> None:
    mutants_root = project_root / "mutants"
    for relative_path in family["modules"]:
        mutant_path = mutants_root / relative_path
        if mutant_path.is_file():
            mutant_path.unlink()


def run_mutation(
    project_root: Path,
    family: dict[str, Any],
    *,
    max_children: int,
    mutant_names: tuple[str, ...] = (),
    resume: bool = False,
) -> int:
    if project_root != Path.cwd().resolve():
        raise ValueError("Mutation testing must run from the project root.")

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
    os.environ["PYTHONPATH"] = os.pathsep.join(("src", "tests"))
    tests_path = str(project_root / "tests")
    if tests_path not in sys.path:
        sys.path.insert(0, tests_path)
    if should_reset_mutation_cache(mutant_names=mutant_names, resume=resume):
        _reset_family_mutation_cache(project_root, family)
    else:
        reset_mutation_stats_cache(project_root)

    from mutmut import configuration

    configuration.Config.reset()
    base_config = configuration.Config.get()
    focused_test_paths = _family_test_paths(project_root, family)
    configuration._config = replace(
        base_config,
        source_paths=[Path("src/emperor/")],
        only_mutate=list(family["mutation_scope"]),
        do_not_mutate=list(
            mutation_exclusion_patterns(
                family,
                default_patterns=base_config.do_not_mutate,
            )
        ),
        pytest_add_cli_args_test_selection=[
            path.relative_to(project_root).as_posix() for path in focused_test_paths
        ],
    )

    from mutmut import __main__ as mutmut_main

    _run = mutmut_main._run
    collect_source_file_mutation_data = mutmut_main.collect_source_file_mutation_data
    status_by_exit_code = mutmut_main.status_by_exit_code

    names_to_run = mutant_names
    cached_exit_codes: dict[str, int] = {}
    cached_mutant_names: set[str] = set()
    rerun_mutant_names: set[str] = set()
    preserve_cache = False
    if resume or mutant_names:
        cached_mutants, _ = collect_source_file_mutation_data(mutant_names=[])
        cached_results = tuple(
            (mutant_name, result) for _, mutant_name, result in cached_mutants
        )
        names_to_run = cached_mutant_names_to_run(
            cached_results,
            mutant_names,
            resume=resume,
        )
        if resume and not mutant_names:
            print(f"Resuming {len(names_to_run)} unchecked mutants.")
        if cached_results:
            cached_exit_codes = {
                mutant_name: result
                for mutant_name, result in cached_results
                if result is not None
            }
            cached_mutant_names = {mutant_name for mutant_name, _ in cached_results}
            rerun_mutant_names = set(names_to_run)
            preserve_cache = True

    if names_to_run or not resume:
        if preserve_cache:
            original_create_mutants = mutmut_main.create_mutants

            def create_mutants_preserving_cache(children: int) -> Any:
                reset_family_mutant_sources(project_root, family)
                generation_stats = original_create_mutants(children)
                generated_mutants, generated_by_path = (
                    collect_source_file_mutation_data(mutant_names=[])
                )
                generated_mutant_names = {
                    mutant_name for _, mutant_name, _ in generated_mutants
                }
                if generated_mutant_names != cached_mutant_names:
                    raise ValueError(
                        "Cached mutants no longer match generated mutants; "
                        "run mutation mode without --resume."
                    )
                for mutation_data in generated_by_path.values():
                    mutation_data.exit_code_by_key = restore_cached_exit_codes(
                        mutation_data.exit_code_by_key,
                        cached_exit_codes,
                        rerun_names=rerun_mutant_names,
                    )
                    mutation_data.save()
                return generation_stats

            mutmut_main.create_mutants = create_mutants_preserving_cache
            try:
                _run(list(names_to_run), max_children)
            finally:
                mutmut_main.create_mutants = original_create_mutants
        else:
            _run(list(names_to_run), max_children)
    mutants, _ = collect_source_file_mutation_data(mutant_names=list(mutant_names))
    results = tuple(
        sorted(
            (mutant_name, status_by_exit_code[result])
            for _, mutant_name, result in mutants
        )
    )
    documented_equivalents = documented_equivalent_mutants(
        project_root,
        family,
    )
    if mutant_names:
        documented_equivalents.intersection_update(
            selected_mutant_names(documented_equivalents, mutant_names)
        )
    unexpected, stale = classify_mutation_results(
        results,
        documented_equivalents,
    )
    non_killed = tuple(
        (mutant_name, status) for mutant_name, status in results if status != "killed"
    )
    for mutant_name, status in non_killed:
        classification = (
            " (documented equivalent)" if mutant_name in documented_equivalents else ""
        )
        print(f"{mutant_name}: {status}{classification}")
    for mutant_name in stale:
        print(f"{mutant_name}: stale equivalent evidence")
    return 1 if unexpected or stale else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run manifest-driven Emperor family quality gates."
    )
    parser.add_argument("mode", choices=("tests", "coverage", "mutation"))
    parser.add_argument("family")
    parser.add_argument("--max-children", type=int, default=4)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue a full family mutation run without clearing cached results.",
    )
    parser.add_argument(
        "--mutant",
        action="append",
        default=[],
        help="Rerun one mutant name or glob without clearing the family cache.",
    )
    arguments = parser.parse_args(argv)

    try:
        family = load_family(PROJECT_ROOT, arguments.family)
        if arguments.mode == "tests":
            return run_tests(PROJECT_ROOT, family)
        if arguments.mode == "coverage":
            return run_coverage(PROJECT_ROOT, family)
        return run_mutation(
            PROJECT_ROOT,
            family,
            max_children=arguments.max_children,
            mutant_names=tuple(arguments.mutant),
            resume=arguments.resume,
        )
    except ValueError as error:
        parser.error(str(error))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

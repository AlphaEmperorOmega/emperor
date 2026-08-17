import unittest
from collections.abc import Iterator, Mapping
from dataclasses import fields, is_dataclass
from importlib import import_module
from inspect import isclass

from emperor.layers import RecurrentLayerConfig
from model_runtime.inspection.runtime_defaults import runtime_defaults_spec
from models.catalog import discover_model_packages

EXISTING_RECURRENT_CONTROL_KEYS = frozenset(
    {
        "RECURRENT_MAX_STEPS",
        "RECURRENT_HALTING_THRESHOLD",
    }
)
MINIMUM_AND_PONDER_CONTROL_KEYS = frozenset(
    {
        "RECURRENT_MIN_STEPS",
        "RECURRENT_PONDER_COST_WEIGHT",
    }
)
ITERATION_SCHEDULE_CONTROL_KEYS = frozenset(
    {
        "RECURRENT_INITIAL_ITERATIONS",
        "RECURRENT_GRADIENT_TRANSITION_COUNT",
        "RECURRENT_ITERATION_INCREMENT",
        "RECURRENT_FORWARD_CALLS_BEFORE_ITERATION_INCREMENT",
    }
)
SMOOTH_ITERATION_GROWTH_CONTROL_KEY = "RECURRENT_SMOOTH_ITERATION_GROWTH_FLAG"
NO_GRADIENT_TRANSITION_COUNT_CONTROL_KEY = "RECURRENT_NO_GRADIENT_TRANSITION_COUNT"
RECURRENT_MODEL_PACKAGES = frozenset(
    {
        "bert/expert_linear",
        "bert/expert_linear_adaptive",
        "bert/linear",
        "bert/linear_adaptive",
        "experts/linear",
        "experts/linear_adaptive",
        "gpt/expert_linear",
        "gpt/expert_linear_adaptive",
        "gpt/linear",
        "gpt/linear_adaptive",
        "linears/linear",
        "linears/linear_adaptive",
        "mlp_mixer/expert_linear",
        "mlp_mixer/expert_linear_adaptive",
        "mlp_mixer/linear",
        "mlp_mixer/linear_adaptive",
        "neuron/expert_linear",
        "neuron/expert_linear_adaptive",
        "neuron/linear",
        "neuron/linear_adaptive",
        "transformer/expert_linear",
        "transformer/expert_linear_adaptive",
        "transformer/linear",
        "transformer/linear_adaptive",
        "vit/expert_linear",
        "vit/expert_linear_adaptive",
        "vit/linear",
        "vit/linear_adaptive",
    }
)


def _recurrent_iteration_schedule_values(
    value: object,
) -> Iterator[tuple[object, ...]]:
    seen: set[int] = set()

    def visit(candidate: object) -> Iterator[tuple[object, ...]]:
        candidate_id = id(candidate)
        if candidate_id in seen:
            return
        seen.add(candidate_id)

        recurrent_fields = (
            "recurrent_initial_iterations",
            "recurrent_gradient_transition_count",
            "recurrent_iteration_increment",
            "recurrent_forward_calls_before_iteration_increment",
        )
        nested_fields = (
            "initial_iterations",
            "gradient_transition_count",
            "iteration_increment",
            "forward_calls_before_iteration_increment",
        )
        for field_names in (recurrent_fields, nested_fields):
            if all(hasattr(candidate, name) for name in field_names):
                yield tuple(getattr(candidate, name) for name in field_names)

        if is_dataclass(candidate) and not isinstance(candidate, type):
            for field in fields(candidate):
                yield from visit(getattr(candidate, field.name))
        elif isinstance(candidate, Mapping):
            for item in candidate.values():
                yield from visit(item)
        elif isinstance(candidate, (list, tuple)):
            for item in candidate:
                yield from visit(item)

    yield from visit(value)


def _standard_recurrent_configs(value: object) -> Iterator[RecurrentLayerConfig]:
    seen: set[int] = set()

    def visit(candidate: object) -> Iterator[RecurrentLayerConfig]:
        candidate_id = id(candidate)
        if candidate_id in seen:
            return
        seen.add(candidate_id)

        if isinstance(candidate, RecurrentLayerConfig):
            yield candidate
        if is_dataclass(candidate) and not isinstance(candidate, type):
            for field in fields(candidate):
                yield from visit(getattr(candidate, field.name))
        elif isinstance(candidate, Mapping):
            for item in candidate.values():
                yield from visit(item)
        elif isinstance(candidate, (list, tuple)):
            for item in candidate:
                yield from visit(item)

    yield from visit(value)


def _smooth_iteration_growth_values(value: object) -> Iterator[bool]:
    seen: set[int] = set()

    def visit(candidate: object) -> Iterator[bool]:
        candidate_id = id(candidate)
        if candidate_id in seen:
            return
        seen.add(candidate_id)

        for field_name in (
            "recurrent_smooth_iteration_growth_flag",
            "smooth_iteration_growth_flag",
        ):
            if hasattr(candidate, field_name):
                yield getattr(candidate, field_name)

        if is_dataclass(candidate) and not isinstance(candidate, type):
            for field in fields(candidate):
                yield from visit(getattr(candidate, field.name))
        elif isinstance(candidate, Mapping):
            for item in candidate.values():
                yield from visit(item)
        elif isinstance(candidate, (list, tuple)):
            for item in candidate:
                yield from visit(item)

    yield from visit(value)


def _no_gradient_transition_count_values(value: object) -> Iterator[int | None]:
    seen: set[int] = set()

    def visit(candidate: object) -> Iterator[int | None]:
        candidate_id = id(candidate)
        if candidate_id in seen:
            return
        seen.add(candidate_id)

        for field_name in (
            "recurrent_no_gradient_transition_count",
            "no_gradient_transition_count",
        ):
            if hasattr(candidate, field_name):
                yield getattr(candidate, field_name)

        if is_dataclass(candidate) and not isinstance(candidate, type):
            for field in fields(candidate):
                yield from visit(getattr(candidate, field.name))
        elif isinstance(candidate, Mapping):
            for item in candidate.values():
                yield from visit(item)
        elif isinstance(candidate, (list, tuple)):
            for item in candidate:
                yield from visit(item)

    yield from visit(value)


def _config_builder_type(catalog_key: str) -> type:
    module_name = f"models.{catalog_key.replace('/', '.')}.config_builder"
    module = import_module(module_name)
    builder_types = [
        candidate
        for candidate in vars(module).values()
        if isclass(candidate)
        and candidate.__module__ == module_name
        and candidate.__name__.endswith("ConfigBuilder")
    ]
    if len(builder_types) != 1:
        raise AssertionError(
            f"Expected one public ConfigBuilder in {module_name}, got "
            f"{[candidate.__name__ for candidate in builder_types]}"
        )
    return builder_types[0]


class TestRecurrentRuntimeControlScope(unittest.TestCase):
    def test_no_gradient_transition_count_has_an_explicit_package_scope(self) -> None:
        supported_keys = {
            package.catalog_key: frozenset(
                runtime_defaults_spec(package).supported_keys
            )
            for package in discover_model_packages()
        }

        self.assertEqual(
            {
                catalog_key
                for catalog_key, keys in supported_keys.items()
                if NO_GRADIENT_TRANSITION_COUNT_CONTROL_KEY in keys
            },
            RECURRENT_MODEL_PACKAGES,
        )

    def test_full_gradient_smooth_growth_binds_and_builds_every_package(
        self,
    ) -> None:
        overrides = {
            "recurrent_flag": True,
            "recurrent_max_steps": 3,
            "recurrent_initial_iterations": 2,
            "recurrent_no_gradient_transition_count": 0,
            "recurrent_gradient_transition_count": None,
            "recurrent_iteration_increment": 1,
            "recurrent_forward_calls_before_iteration_increment": 4,
            "recurrent_smooth_iteration_growth_flag": True,
        }

        for package in discover_model_packages():
            if package.catalog_key not in RECURRENT_MODEL_PACKAGES:
                continue
            with self.subTest(catalog_key=package.catalog_key):
                runtime = package.bind_runtime_defaults(overrides)
                self.assertIn(
                    0,
                    set(_no_gradient_transition_count_values(runtime)),
                )

                configuration = _config_builder_type(package.catalog_key)(
                    runtime=runtime
                ).build()
                recurrent_configs = list(_standard_recurrent_configs(configuration))
                self.assertTrue(recurrent_configs)
                matching_configs = [
                    recurrent
                    for recurrent in recurrent_configs
                    if recurrent.max_steps == 3
                    and recurrent.initial_iterations == 2
                    and recurrent.no_gradient_transition_count == 0
                    and recurrent.gradient_transition_count is None
                    and recurrent.iteration_increment == 1
                    and recurrent.forward_calls_before_iteration_increment == 4
                    and recurrent.smooth_iteration_growth_flag is True
                ]
                self.assertTrue(
                    matching_configs,
                    "top-level recurrent config did not receive the explicit "
                    "full-gradient smooth-growth window",
                )
                for recurrent in matching_configs:
                    recurrent_dimension = (
                        recurrent.input_dim
                        or getattr(recurrent.block_config, "input_dim", None)
                        or 4
                    )
                    recurrent_model = recurrent.build(
                        overrides=type(recurrent)(
                            input_dim=recurrent_dimension,
                            output_dim=recurrent_dimension,
                        )
                    )
                    schedule = recurrent_model.recurrent_iteration_schedule
                    schedule.forward_call_progress.fill_(4)
                    self.assertEqual(
                        type(schedule.execution_plan()).__name__,
                        "RecurrentNestedSmoothHandoffExecutionPlan",
                    )

    def test_no_gradient_transition_count_preserves_none_and_rejects_bool(
        self,
    ) -> None:
        for package in discover_model_packages():
            if package.catalog_key not in RECURRENT_MODEL_PACKAGES:
                continue
            with self.subTest(catalog_key=package.catalog_key, value=None):
                runtime = package.bind_runtime_defaults(
                    {"recurrent_no_gradient_transition_count": None}
                )
                values = list(_no_gradient_transition_count_values(runtime))
                self.assertTrue(values)
                self.assertTrue(all(value is None for value in values))
            with self.subTest(catalog_key=package.catalog_key, value=True):
                with self.assertRaisesRegex(
                    TypeError,
                    "recurrent_no_gradient_transition_count",
                ):
                    package.bind_runtime_defaults(
                        {"recurrent_no_gradient_transition_count": True}
                    )

    def test_no_gradient_transition_count_rejects_negative_package_config(
        self,
    ) -> None:
        overrides = {
            "recurrent_flag": True,
            "recurrent_max_steps": 3,
            "recurrent_initial_iterations": 2,
            "recurrent_no_gradient_transition_count": -1,
            "recurrent_gradient_transition_count": None,
            "recurrent_iteration_increment": 1,
            "recurrent_forward_calls_before_iteration_increment": 1,
            "recurrent_smooth_iteration_growth_flag": False,
        }

        package = next(
            package
            for package in discover_model_packages()
            if package.catalog_key == "gpt/linear"
        )
        runtime = package.bind_runtime_defaults(overrides)
        configuration = _config_builder_type(package.catalog_key)(
            runtime=runtime
        ).build()
        matching_configs = [
            recurrent
            for recurrent in _standard_recurrent_configs(configuration)
            if recurrent.no_gradient_transition_count == -1
        ]
        self.assertTrue(matching_configs)
        for recurrent in matching_configs:
            with self.assertRaisesRegex(
                ValueError,
                "no_gradient_transition_count",
            ):
                recurrent.build()

    def test_smooth_iteration_growth_has_an_explicit_package_scope(self) -> None:
        supported_keys = {
            package.catalog_key: frozenset(
                runtime_defaults_spec(package).supported_keys
            )
            for package in discover_model_packages()
        }

        self.assertEqual(
            {
                catalog_key
                for catalog_key, keys in supported_keys.items()
                if SMOOTH_ITERATION_GROWTH_CONTROL_KEY in keys
            },
            RECURRENT_MODEL_PACKAGES,
        )

    def test_smooth_iteration_growth_binds_and_builds_every_recurrent_config(
        self,
    ) -> None:
        overrides = {
            "recurrent_flag": True,
            "recurrent_max_steps": 3,
            "recurrent_initial_iterations": 2,
            "recurrent_gradient_transition_count": 2,
            "recurrent_iteration_increment": 1,
            "recurrent_forward_calls_before_iteration_increment": 4,
            "recurrent_smooth_iteration_growth_flag": True,
        }

        for package in discover_model_packages():
            if package.catalog_key not in RECURRENT_MODEL_PACKAGES:
                continue
            with self.subTest(catalog_key=package.catalog_key):
                runtime = package.bind_runtime_defaults(overrides)
                smooth_runtime_values = set(_smooth_iteration_growth_values(runtime))
                self.assertIn(True, smooth_runtime_values)

                configuration = _config_builder_type(package.catalog_key)(
                    runtime=runtime
                ).build()
                recurrent_configs = list(_standard_recurrent_configs(configuration))
                self.assertTrue(recurrent_configs)
                self.assertTrue(
                    any(
                        recurrent.max_steps == 3
                        and recurrent.gradient_transition_count == 2
                        and recurrent.initial_iterations == 2
                        and recurrent.iteration_increment == 1
                        and recurrent.forward_calls_before_iteration_increment == 4
                        and recurrent.smooth_iteration_growth_flag is True
                        for recurrent in recurrent_configs
                    ),
                    "top-level recurrent config did not receive smooth iteration growth",
                )

    def test_iteration_schedule_controls_cover_every_recurrent_package(self) -> None:
        supported_keys = {
            package.catalog_key: frozenset(
                runtime_defaults_spec(package).supported_keys
            )
            for package in discover_model_packages()
        }

        self.assertEqual(
            {
                catalog_key
                for catalog_key, keys in supported_keys.items()
                if ITERATION_SCHEDULE_CONTROL_KEYS <= keys
            },
            RECURRENT_MODEL_PACKAGES,
        )

    def test_iteration_schedule_values_bind_in_every_recurrent_package(self) -> None:
        expected = (2, 2, 3, 4)
        overrides = {
            "recurrent_initial_iterations": expected[0],
            "recurrent_gradient_transition_count": expected[1],
            "recurrent_iteration_increment": expected[2],
            "recurrent_forward_calls_before_iteration_increment": expected[3],
        }

        for package in discover_model_packages():
            if package.catalog_key not in RECURRENT_MODEL_PACKAGES:
                continue
            with self.subTest(catalog_key=package.catalog_key):
                runtime = package.bind_runtime_defaults(overrides)
                self.assertIn(
                    expected,
                    set(_recurrent_iteration_schedule_values(runtime)),
                )

    def test_iteration_schedule_builds_in_every_recurrent_package(self) -> None:
        overrides = {
            "recurrent_flag": True,
            "recurrent_max_steps": 10,
            "recurrent_initial_iterations": 2,
            "recurrent_gradient_transition_count": 2,
            "recurrent_iteration_increment": 3,
            "recurrent_forward_calls_before_iteration_increment": 4,
        }

        for package in discover_model_packages():
            if package.catalog_key not in RECURRENT_MODEL_PACKAGES:
                continue
            with self.subTest(catalog_key=package.catalog_key):
                runtime = package.bind_runtime_defaults(overrides)
                configuration = _config_builder_type(package.catalog_key)(
                    runtime=runtime
                ).build()
                recurrent_configs = list(_standard_recurrent_configs(configuration))
                self.assertTrue(
                    any(
                        recurrent.max_steps == 10
                        and recurrent.gradient_transition_count == 2
                        and recurrent.initial_iterations == 2
                        and recurrent.iteration_increment == 3
                        and recurrent.forward_calls_before_iteration_increment == 4
                        for recurrent in recurrent_configs
                    ),
                    "top-level recurrent config did not receive the iteration schedule",
                )

    def test_iteration_schedule_starts_at_two_iterations_in_every_package(
        self,
    ) -> None:
        for package in discover_model_packages():
            if package.catalog_key not in RECURRENT_MODEL_PACKAGES:
                continue
            with self.subTest(catalog_key=package.catalog_key):
                runtime = package.bind_runtime_defaults({"recurrent_flag": True})
                configuration = _config_builder_type(package.catalog_key)(
                    runtime=runtime
                ).build()
                recurrent_configs = list(_standard_recurrent_configs(configuration))
                self.assertTrue(recurrent_configs)
                self.assertTrue(
                    all(
                        recurrent.gradient_transition_count is None
                        and recurrent.initial_iterations == 2
                        for recurrent in recurrent_configs
                    )
                )

    def test_minimum_and_ponder_controls_have_an_explicit_package_scope(self) -> None:
        supported_keys = {
            package.catalog_key: frozenset(
                runtime_defaults_spec(package).supported_keys
            )
            for package in discover_model_packages()
        }
        recurrent_packages = {
            catalog_key
            for catalog_key, keys in supported_keys.items()
            if EXISTING_RECURRENT_CONTROL_KEYS <= keys
        }

        self.assertEqual(recurrent_packages, RECURRENT_MODEL_PACKAGES)
        self.assertEqual(
            {
                catalog_key
                for catalog_key in recurrent_packages
                if MINIMUM_AND_PONDER_CONTROL_KEYS <= supported_keys[catalog_key]
            },
            {"gpt/expert_linear_adaptive"},
        )
        for catalog_key in recurrent_packages - {"gpt/expert_linear_adaptive"}:
            with self.subTest(catalog_key=catalog_key):
                self.assertTrue(
                    MINIMUM_AND_PONDER_CONTROL_KEYS.isdisjoint(
                        supported_keys[catalog_key]
                    )
                )


if __name__ == "__main__":
    unittest.main()
